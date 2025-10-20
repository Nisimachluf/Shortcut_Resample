import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_2d_sincos_pos_embed(dummy, embed_dim, length):
    # Match JAX implementation: embed_dim = D, length = H*W
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)  # (1, H*W, D)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TrainConfig:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    def kern_init(self, name='default', zero=False):
        if zero or 'bias' in name:
            return nn.init.constant_
        return nn.init.xavier_uniform_
    def default_config(self):
        return {}

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, tc, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.tc = tc
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, t):
        x_e = self.timestep_embedding(t)
        x1 = self.fc1(x_e[0])
        x = F.silu(x1)
        x = self.fc2(x)
        return x, *x_e, x1
    
    def timestep_embedding(self, t, max_period=10000):
        t = t.float()
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        embedding1 = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = embedding1.to(self.tc.dtype)
        return embedding, t, dim, half, freqs, embedding1

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, tc):
        super().__init__()
        self.embed = nn.Embedding(num_classes + 1, hidden_size)
    def forward(self, labels):
        return self.embed(labels)

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, hidden_size, tc, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
    def forward(self, x):
        B, H, W, C = x.shape
        # x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.conv(x)  # (B, hidden_size, H//patch, W//patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        return x

class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, tc, out_dim=None, dropout_rate=None, train=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim if out_dim is not None else input_dim)
        self.dropout_rate = dropout_rate
        self._train = train
    def forward(self, inputs):
        x = self.fc1(inputs)
        
        x = F.gelu(x)
        if self.dropout_rate:
            x = F.dropout(x, self.dropout_rate, self._train)
        x = self.fc2(x)
        
        if self.dropout_rate:
            x = F.dropout(x, self.dropout_rate, self._train)
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, tc, mlp_ratio=4.0, dropout=0.0, train=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tc = tc
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self._train = train
        self.ln1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp = MlpBlock(hidden_size, int(hidden_size * mlp_ratio), tc, dropout_rate=dropout, train=train)
        self.fc_mod = nn.Linear(hidden_size, 6 * hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_attn = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, c):
        c = F.silu(c)
        cact = c.clone()
        c = self.fc_mod(c)
        
        # print("c2", c.shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(c, 6, dim=-1)
        # print("shift_msa", shift_msa.shape)
        x_norm = self.ln1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        # print("x_modulated", x_modulated.shape)
        channels_per_head = self.hidden_size // self.num_heads
        # print("channels_per_head", channels_per_head)
        k = self.fc_k(x_modulated)
        # print("k", k.shape)
        q = self.fc_q(x_modulated)
        v = self.fc_v(x_modulated)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, channels_per_head)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, channels_per_head)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, channels_per_head)
        q = q / q.shape[3]
        w = torch.einsum('bqhc,bkhc->bhqk', q, k)
        w = F.softmax(w.float(), dim=-1)
        y = torch.einsum('bhqk,bkhc->bqhc', w, v)
        # print("y", y.shape)
        # print("x.shape", x.shape)
        y = y.reshape(x.shape)
        # print("y2", y.shape)
        attn_x = self.fc_attn(y)
        x = x + (gate_msa.unsqueeze(1) * attn_x)
        x_norm2 = self.ln2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        # print("x_modulated2", x_modulated2.shape)
        mlp_x = self.mlp(x_modulated2)
        x = x + (gate_mlp.unsqueeze(1) * mlp_x)
        return x, x_norm, x_modulated, shift_msa, scale_msa, c.clone(), cact

class FinalLayer(nn.Module):
    def __init__(self, patch_size, out_channels, hidden_size, tc):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.fc_mod = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
    def forward(self, x, c):
        c = F.silu(c)
        c = self.fc_mod(c)
        shift, scale = torch.chunk(c, 2, dim=-1)
        x = self.ln(x)
        x = modulate(x, shift, scale)
        x = self.fc_out(x)
        return x

class DiT(nn.Module):
    def __init__(self, patch_embed_in_channels, patch_size, hidden_size, depth, num_heads, mlp_ratio, out_channels, class_dropout_prob, num_classes, ignore_dt=False, dropout=0.0, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        self.dtype = dtype
        self.tc = TrainConfig(dtype=dtype)
        self.patch_embed = PatchEmbed(patch_embed_in_channels, patch_size, hidden_size, self.tc)
        self.time_embed = TimestepEmbedder(hidden_size, self.tc)
        self.dt_embed = TimestepEmbedder(hidden_size, self.tc)
        self.label_embed = LabelEmbedder(num_classes, hidden_size, self.tc)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, self.tc, mlp_ratio, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(patch_size, out_channels, hidden_size, self.tc)
        self.pos_embed = None  # Will be initialized in forward
        self.device = device

    def forward(self, x, t, dt, y, train=False, return_activations=False):
        activations = {}
        batch_size = x.shape[0]
        input_size = x.shape[-1]
        in_channels = x.shape[1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size

        if self.ignore_dt:
            dt = torch.zeros_like(t)

        if self.pos_embed is None:
            self.pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, num_patches).to(x.device)
        true_input_x = x.clone()
        x = self.patch_embed(x)
        x_patch_embed = x.clone()
        x = x + self.pos_embed
        xpos_embed = x.clone()
        activations['pos_embed'] = xpos_embed
        
        x = x.to(dtype=self.dtype)
        te, *_ = self.time_embed(t)
        dte, *_ = self.dt_embed(dt)
        ye = self.label_embed(y)
        activations['time_embed'] = te.clone()
        activations['dt_embed'] = dte.clone()
        activations['label_embed'] = ye.clone()
        c = te + ye + dte
        interms = []
        for i, block in enumerate(self.blocks):
            x, *_ = block(x, c)
            interms.append(x.clone())
            activations[f'dit_block_{i}'] = x.clone()
            
        x = self.final_layer(x, c)
        activations['final_layer'] = x.clone()
        x = x.view(batch_size, num_patches_side, num_patches_side, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, input_size, input_size, self.out_channels)
        x = x.permute(0, 3, 1, 2)

        t_discrete = torch.floor(t * 256).int()
        logvars = nn.Embedding(256, 1).to(self.device)
        logvars_out = logvars(t_discrete) * 100

        if return_activations:
            return x, logvars_out, activations
        return x , self.pos_embed, te, dte, ye, *interms, true_input_x, x_patch_embed, xpos_embed
    
    if __name__ == "__main__":
        patch = PatchEmbed(4,2,768,TrainConfig(dtype=torch.float32))
        patch.to("cuda")
        arr = torch.rand((1, 32, 32, 4), dtype=torch.float32, device="cuda")
        c = torch.rand((1, 768), dtype=torch.float32, device="cuda")
        
        output = patch(arr)
        
        block = DiTBlock(768,12,TrainConfig(dtype=torch.float32),4, 0, False).to("cuda")
        output2 = block(output, c)