# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rotate0(x):
    return torch.rot90(x, 0, [2,3])

def rotate90(x):
    return torch.rot90(x, 1, [2,3])

def rotate180(x):
    return torch.rot90(x, 2, [2,3])

def rotate270(x):
    return torch.rot90(x, 3, [2,3])

def fliph(x):
    return x.flip(2)

def flipv(x):
    return x.flip(3)

def fliphv(x):
    return x.flip(2).flip(3)


def rand_filter(images, affine=None):
    ratio = 0.25
   
    
    _, Hz_fbank = affine
    Hz_fbank = Hz_fbank.to(images.device)
    imgfilter_bands = [1,1,1,1]
    batch_size, num_channels, height, width = images.shape
    device = images.device
    num_bands = Hz_fbank.shape[0]
    assert len([1,1,1,1]) == num_bands
    expected_power = constant(np.array([10, 1, 1, 1]) / 13, device=device) # Expected power spectrum (1/f).

    # Apply amplification for each band with probability (imgfilter * strength * band_strength).
    g = torch.ones([batch_size, num_bands], device=device) # Global gain vector (identity).
    for i, band_strength in enumerate(imgfilter_bands):
        t_i = torch.exp2(torch.randn([batch_size], device=device) * 1)
        t_i = torch.where(torch.rand([batch_size], device=device) < ratio * band_strength, t_i, torch.ones_like(t_i))
#         if debug_percentile is not None:
#             t_i = torch.full_like(t_i, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * 1)) if band_strength > 0 else torch.ones_like(t_i)
        t = torch.ones([batch_size, num_bands], device=device)                  # Temporary gain vector.
        t[:, i] = t_i                                                           # Replace i'th element.
        t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
        g = g * t                                                               # Accumulate into global gain.

    # Construct combined amplification filter.
    Hz_prime = g @ Hz_fbank                                    # [batch, tap]
    Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])   # [batch, channels, tap]
    Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1]) # [batch * channels, 1, tap]

    # Apply filter.
    p = Hz_fbank.shape[1] // 2
    images = images.reshape([1, batch_size * num_channels, height, width])
    images = torch.nn.functional.pad(input=images, pad=[p,p,p,p], mode='reflect')
    images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size*num_channels)
    images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size*num_channels)
    images = images.reshape([batch_size, num_channels, height, width])
    return images

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)
    
def rand_hue(images, affine=None):
    batch_size, num_channels, height, width = images.shape
    device = images.device
    I_4 = torch.eye(4, device=device)
    C = I_4
    v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device) # Luma axis.

    # Apply hue rotation with probability (hue * strength).
    if num_channels > 1:
        theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * 1
        theta = torch.where(torch.rand([batch_size], device=device) < 0.5, theta, torch.zeros_like(theta))
#         if debug_percentile is not None:
#             theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * 1)
        C = rotate3d(v, theta) @ C # Rotate around v.

    # Apply saturation with probability (saturation * strength).
#     if self.saturation > 0 and num_channels > 1:
#         s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
#         s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
#         if debug_percentile is not None:
#             s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
#         C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

    # ------------------------------
    # Execute color transformations.
    # ------------------------------

    # Execute if the transform is not identity.
    if C is not I_4:
        images = images.reshape([batch_size, num_channels, height * width])
        if num_channels == 3:
            images = C[:, :3, :3] @ images + C[:, :3, 3:]
        elif num_channels == 1:
            C = C[:, :3, :].mean(dim=1, keepdims=True)
            images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
        else:
            raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
        images = images.reshape([batch_size, num_channels, height, width])
    return images

def rand_crop(x, affine=None):
    b, _, h, w = x.shape
    x_large = torch.nn.functional.interpolate(x, scale_factor=1.2, mode='bicubic')
    _, _, h_large, w_large = x_large.size()
    h_start, w_start = random.randint(0, (h_large - h)), random.randint(0, (w_large - w))
    x_crop = x_large[:, :, h_start:h_start+h, w_start:w_start+w]
    assert x_crop.size() == x.size()
    output = torch.where(torch.rand([b, 1, 1, 1], device=x.device) < 0.2, x_crop, x)
    return output

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def rand_erase_ratio(x, ratio=0.5, affine=None):
    ratio_x = random.randint(int(x.size(2)*0.2), int(x.size(2)*0.7))
    ratio_y = random.randint(int(x.size(3)*0.2), int(x.size(3)*0.7))
    if random.random() < 0.3:
#         cutout_size = int(x.size(2) * ratio_x + 0.5), int(x.size(3) * ratio_y + 0.5)
        cutout_size = ratio_x, ratio_y
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'rotate':[rotate90],
    'flip':[rotate90, rotate180, rotate270],
    'filter': [rand_filter],
    'erase_ratio': [rand_erase_ratio],
    'hue': [rand_hue],
    'crop': [rand_crop],
    'translation': [rand_translation, rand_cutout],
    'cutout': [rand_cutout],
}