import torch
import torch.nn.functional as F

def CEloss(cfg):
    smoothing = cfg.get('smoothing', 0.0)
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.class_weights).to(cfg.device), label_smoothing=smoothing)
           

