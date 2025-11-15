from ldm.util import instantiate_from_config
from stablediffusion.ldm.util import instantiate_from_config as instantiate_from_config_sd
import yaml
import torch

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# def load_model_from_config(config, ckpt, train=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt)#, map_location="cpu")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     _, _ = model.load_state_dict(sd, strict=False)
    
#     model.cuda()

#     if train:
#       model.train()
#     else:
#       model.eval()

#     return model
  
def load_model_from_config(config, ckpt, device=torch.device("cuda"), model_class="ldm", verbose=False):
  print(f"Loading model from {ckpt}")
  pl_sd = torch.load(ckpt, map_location="cpu")
  if "global_step" in pl_sd:
      print(f"Global Step: {pl_sd['global_step']}")
  sd = pl_sd["state_dict"]
  if model_class == "ldm":
    model = instantiate_from_config(config.model)
  elif model_class == "stable_diffusion":
    model = instantiate_from_config_sd(config.model)
  else:
    raise ValueError(f"Unsupported model class {model_class}")
  m, u = model.load_state_dict(sd, strict=False)
  if len(m) > 0 and verbose:
      print("missing keys:")
      print(m)
  if len(u) > 0 and verbose:
      print("unexpected keys:")
      print(u)

  if device == torch.device("cuda"):
      model.cuda()
  elif device == torch.device("cpu"):
      model.cpu()
      model.cond_stage_model.device = "cpu"
  else:
      raise ValueError(f"Incorrect device name. Received: {device}")
  model.eval()
  return model