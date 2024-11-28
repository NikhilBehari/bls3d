import torch
import torch.nn.functional as F

def snr_shifting(x, scaling_a, scaling_b):
    x = x.item() if isinstance(x, torch.Tensor) else x
    return max(scaling_a * x + scaling_b, 0.001)

def image_snr(img):
    if img.ndim == 5:
        img = img.squeeze(0).squeeze(0) 
    elif img.ndim == 4:
        img = img.squeeze(0)  
    
    if img.shape[0] == 3: 
        img = torch.mean(img, dim=0)   

    laplacian_filter = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=img.dtype, device=img.device)
    laplacian = F.conv2d(img.unsqueeze(0).unsqueeze(0), laplacian_filter, padding=1).squeeze()

    signal = torch.mean(img**2)
    noise = torch.mean(laplacian**2)
    return 10 * torch.log10(signal / noise) if noise > 0 else float('-inf')

def batch_histogram(timing_data, weights, t_min, t_max, num_bins):
    bin_width = (t_max - t_min) / num_bins
    timing_quant = ((timing_data - t_min) / bin_width)
    
    valid_mask = (timing_data >= t_min) & (timing_data < t_max)
    timing_quant = timing_quant * valid_mask.float() 
    lower_bin = timing_quant.floor().long()
    upper_bin = (lower_bin + 1).clamp(0, num_bins - 1)
    weight_upper = timing_quant - lower_bin.float()
    weight_lower = 1 - weight_upper

    hist = torch.zeros((num_bins, *timing_quant.shape[1:]), dtype=timing_data.dtype, device=timing_data.device)    
    hist.scatter_add_(0, lower_bin, weights * weight_lower * valid_mask.float())
    hist.scatter_add_(0, upper_bin, weights * weight_upper * valid_mask.float())
    
    return hist


def normalize_hist(hist):
    hist += 1e-16
    return hist / (hist.sum(dim=0, keepdim=True))

def total_variation_loss(depth):
    depth = depth.squeeze(0) 
    diff_h = torch.abs(depth[:, 1:] - depth[:, :-1])
    diff_w = torch.abs(depth[1:, :] - depth[:-1, :])
    weighted_diff_h = diff_h ** 2
    weighted_diff_w = diff_w ** 2
    return weighted_diff_h.mean() + weighted_diff_w.mean()

def convolve_histograms(hists: torch.Tensor, pulse: torch.Tensor) -> torch.Tensor:
    hists = hists.view(128, -1).transpose(0, 1).unsqueeze(1)
    hists = torch.nn.functional.pad(hists, (128, 128))
    pulse = pulse.view(1, 1, -1)
    conv_result = torch.nn.functional.conv1d(hists, pulse, padding=0)
    conv_result = conv_result[:, :, :128].squeeze(1).transpose(0, 1).view(128, 8, 8)
    return conv_result

def convolve_histograms_64(hists: torch.Tensor, pulse: torch.Tensor) -> torch.Tensor:
    hists = hists.view(128, -1).transpose(0, 1).unsqueeze(1)
    hists = torch.nn.functional.pad(hists, (128, 128))
    pulse = pulse.view(1, 1, -1)
    conv_result = torch.nn.functional.conv1d(hists, pulse, padding=0)
    conv_result = conv_result[:, :, :128].squeeze(1).transpose(0, 1).view(128, 64)
    return conv_result
