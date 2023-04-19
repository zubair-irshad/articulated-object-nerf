import torch
from matplotlib.cm import get_cmap
import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import OpenEXR
import minexr
import Imath
from PIL import Image
import torch
from torchvision import transforms as T
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map
    Parameters
    ----------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map
    Returns
    -------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map
    """
    inv_depth = 1. / depth.clamp(min=1e-6)
    inv_depth[depth <= 0.] = 0.
    return inv_depth

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.
    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization
    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        inv_depth = inv_depth.squeeze(0).squeeze(0)
        # Squeeze if depth channel exists
        # if len(inv_depth.shape) == 3:
        #     inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    print("inv_depth", inv_depth.shape)
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    print("inv depth", inv_depth.shape)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

from PIL import Image
import OpenEXR
import Imath
import numpy

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def exr_to_array(exrfile):
    file = OpenEXR.InputFile(exrfile)
    dw = file.header()['dataWindow']

    channels = file.header()['channels'].keys()
    channels_list = list()
    for c in ('R', 'G', 'B', 'A'):
        if c in channels:
            channels_list.append(c)

    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    color_channels = file.channels(channels_list, FLOAT)
    channels_tuple = [numpy.fromstring(channel, dtype='f') for channel in color_channels]
    res = numpy.dstack(channels_tuple)
    return res.reshape(size + (len(channels_tuple),))
  
  
def encode_to_srgb(image_array):
    a = 0.055
    return numpy.where(image_array <= 0.0031308,
                       image_array * 12.92,
                       (1 + a) * pow(image_array, 1 / 2.4) - a)
    
    
def exr_to_srgb(exrfile):
    array = exr_to_array(exrfile)
    result = encode_to_srgb(array) * 255.
    present_channels = ["R", "G", "B", "A"][:result.shape[2]]
    channels = "".join(present_channels)
    return Image.fromarray(result.astype('uint8'), channels)


def exr2numpy(exr_path, chanel_name):
    '''
    See:
    https://excamera.com/articles/26/doc/intro.html
    http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    '''
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    channel_str = file.channel(chanel_name, Float_Type)
    
    channel = np.fromstring(channel_str, dtype = np.float32).reshape(size[1],-1)
    
    return(channel)

def load_hdr_as_tensor(img_path):
    """Converts OpenEXR image to torch float tensor."""

    # Read OpenEXR file
    if not OpenEXR.isOpenExrFile(img_path):
        raise ValueError(f'Image {img_path} is not a valid OpenEXR file')
    src = OpenEXR.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # Read into tensor
    tensor = torch.zeros((3, size[1], size[0]))
    for i, c in enumerate('RGB'):
        rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
        tensor[i, :, :] = torch.from_numpy(rgb32f.reshape(size[1], size[0]))
        
    return tensor 

def EncodeToSRGB(v):
    return(np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055))
if __name__ == '__main__':
    exr_rgb = '/home/zubair/Downloads/google_valts/00007/00000.exr'

    channels = []
    channel_names = ['R','G','B']
    for channel_name in channel_names:
        channel = exr2numpy(exr_rgb, channel_name)
        channels.append(EncodeToSRGB(channel))
        
    RGB = np.dstack(channels)
    img = np.clip(RGB*255, 0, 255).astype(np.uint8)
    plt.imshow(RGB)
    plt.show()

    plt.imshow(img)
    plt.show()
    # print(RGB)