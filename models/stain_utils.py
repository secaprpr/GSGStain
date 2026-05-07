import torch
import torch.nn.functional as F


def hed_from_rgb(device=None, dtype=torch.float32):
    rgb_from_hed = torch.tensor([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ], dtype=dtype, device=device)
    return torch.linalg.inv(rgb_from_hed)


def dab_od_from_rgb(rgb_tensor):
    rgb_tensor = torch.clamp(rgb_tensor, min=1e-6, max=1.0)
    od = -torch.log(rgb_tensor)
    stains = torch.matmul(
        od.reshape(-1, 3),
        hed_from_rgb(rgb_tensor.device, rgb_tensor.dtype)
    )
    stains = stains.reshape(*rgb_tensor.shape)
    dab_od = torch.clamp(stains[..., 2], 0.0, 1.0)
    return dab_od


def dab_semantic_map(image, positive_threshold=0.15):
    dab_od = dab_od_from_rgb(image.permute(0, 2, 3, 1)).unsqueeze(1)
    positive = torch.sigmoid((dab_od - positive_threshold) * 12.0)
    local_mean = F.avg_pool2d(dab_od, kernel_size=3, stride=1, padding=1)
    local_mean_sq = F.avg_pool2d(dab_od * dab_od, kernel_size=3, stride=1, padding=1)
    local_std = torch.sqrt(torch.clamp(local_mean_sq - local_mean * local_mean, min=1e-6))
    return torch.cat([local_mean, local_std, positive], dim=1)
