import torch
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def get_device(cpu):
    if cpu or not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device('cuda')
