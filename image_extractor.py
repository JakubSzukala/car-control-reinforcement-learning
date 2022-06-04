from PIL import Image

import torch
import torchvision.transforms as T

import gym
import numpy as np

# Composition of transformations to apply
resize = T.Compose([
    T.ToPILImage(),
    T.Resize((96, 96), interpolation=Image.CUBIC), # Proposed in example size
    T.ToTensor(), # tensor with shape (CxHxW) in range(0, 1) care for type!
    ])

cvt_grayscale = T.Grayscale()

def get_screen(env):
    """
    Image captured is of size (400, 600, 3) so it has to be converted into
    torch order (CHW):
    (400, 600, 3) -> (3, 400, 600)
    """
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # ascontiguousarray makes sure that array is continous in memory
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen) # Tensor from np array
    
    out = resize(screen)
    #out = cvt_grayscale(out)
     
    # Unsqueeze is for adding batch dim at 0 position (new axis) but why?
    return out.unsqueeze(0)



