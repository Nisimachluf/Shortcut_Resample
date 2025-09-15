import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Flax pads before downsampling, PyTorch Conv2d with padding=1 is equivalent for stride=2
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, groups=32, use_nin_shortcut=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.use_nin_shortcut = use_nin_shortcut if use_nin_shortcut is not None else (in_channels != out_channels)
        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return x + residual

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_head_channels=None, num_groups=32):
        super().__init__()
        self.channels = channels
        self.num_head_channels = num_head_channels or channels
        self.num_heads = channels // self.num_head_channels
        self.group_norm = nn.GroupNorm(num_groups, channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.group_norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        def transpose_for_scores(proj):
            new_shape = proj.shape[:-1] + (self.num_heads, -1)
            proj = proj.view(*new_shape)
            proj = proj.permute(0, 2, 1, 3)
            return proj
        query = transpose_for_scores(query)
        key = transpose_for_scores(key)
        value = transpose_for_scores(value)
        scale = 1 / (self.channels / self.num_heads) ** 0.25
        attn_weights = torch.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = F.softmax(attn_weights, dim=-1)
        x = torch.einsum("...kc,...qk->...qc", value, attn_weights)
        x = x.permute(0, 2, 1, 3).reshape(B, H * W, C)
        x = self.proj_attn(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + residual

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, resnet_groups=32, add_downsample=True, dropout=0.0):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels if i == 0 else out_channels,
                out_channels,
                groups=resnet_groups,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        self.add_downsample = add_downsample
        if add_downsample:
            self.downsampler = Downsample2D(out_channels)

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.add_downsample:
            x = self.downsampler(x)
        return x

class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, resnet_groups=32, add_upsample=True, dropout=0.0):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels if i == 0 else out_channels,
                out_channels,
                groups=resnet_groups,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        self.add_upsample = add_upsample
        if add_upsample:
            self.upsampler = Upsample2D(out_channels)

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.add_upsample:
            x = self.upsampler(x)
        return x

class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, resnet_groups=32, num_attention_heads=1, dropout=0.0, num_layers=1):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        # First resnet
        self.resnets.append(ResnetBlock2D(in_channels, in_channels, groups=resnet_groups, dropout=dropout))
        # Then alternating attention and resnet
        for _ in range(num_layers):
            self.attentions.append(AttentionBlock(in_channels, num_head_channels=in_channels // num_attention_heads, num_groups=resnet_groups))
            self.resnets.append(ResnetBlock2D(in_channels, in_channels, groups=resnet_groups, dropout=dropout))

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, down_block_types=None, block_out_channels=(128, 256, 512, 512), layers_per_block=2, norm_num_groups=32, act_fn='silu', double_z=True):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block
                )
            )
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            num_attention_heads=1,
            dropout=0.0,
            num_layers=1
        )
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1])
        self.act_fn = F.silu if act_fn == 'silu' else F.relu
        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels * 2 if double_z else latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.act_fn(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, up_block_types=None, block_out_channels=(128, 256, 512, 512), layers_per_block=2, norm_num_groups=32, act_fn='silu'):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            num_attention_heads=1,
            dropout=0.0,
            num_layers=1
        )
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, out_channel in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final_block
                )
            )
            prev_output_channel = output_channel
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0])
        self.act_fn = F.silu if act_fn == 'silu' else F.relu
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.act_fn(x)
        x = self.conv_out(x)
        return x

class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=None,
        up_block_types=None,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn='silu',
        latent_channels=4,
        norm_num_groups=32,
        sample_size=256,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True
        )
        self.decoder = Decoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn
        )
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, C, H, W)
        hidden_states = self.encoder(x)
        moments = self.quant_conv(hidden_states)
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample() * self.scaling_factor
        return latents

    def decode(self, latents):
        latents = latents / self.scaling_factor
        latents = self.post_quant_conv(latents)
        x = self.decoder(latents)
        return x

    def forward(self, x):
        latents = self.encode(x)
        recon = self.decode(latents)
        return recon