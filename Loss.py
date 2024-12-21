import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, interpolate=True):
        # Add epsilon to avoid log(0) or division by zero
        eps = 1e-6
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        # Use log(input + eps) to prevent log(0) from occurring
        g = torch.log(input + eps) - torch.log(target + eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        # Also add an epsilon here in sqrt() to prevent sqrt(0) from becoming unstable
        return 10 * torch.sqrt(Dg + eps)


class BerHuLoss(nn.Module):
    def __init__(self, threshold_ratio=0.2):              #The threshold ratio determines the point at which the loss switches from L1 to L2.
        super(BerHuLoss, self).__init__()
        self.threshold_ratio = threshold_ratio

    def forward(self, input, target):
        # Calculate the absolute difference
        eps = 1e-6  # Small value to avoid numerical issues
        diff = torch.abs(input - target)

        # Calculate the threshold value 'c' based on the ratio and the max error
        max_diff = torch.max(diff).item()
        c = max(self.threshold_ratio * max_diff, eps)  # Add epsilon to avoid c being zero

        # Apply the BerHu loss calculation
        mask = diff <= c
        l1_part = diff[mask]  # L1 for smaller errors
        l2_part = (diff[~mask] ** 2 + c ** 2) / (
                    2 * c + eps)  # Add epsilon in denominator to avoid division by zero

        # Combine L1 and L2 losses
        loss = torch.cat([l1_part, l2_part]).mean()
        return loss


class EdgeLoss(nn.Module):
    def forward(self, edge_map_pred, edge_map_gt):
        eps = 1e-6
        # Add epsilon to both prediction and ground truth to ensure no zero-values lead to NaNs
        return F.mse_loss(edge_map_pred + eps, edge_map_gt + eps)




import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming SILogLoss, BerHuLoss, and EdgeLoss classes are defined as above

def test_loss_function(loss_fn, input_tensor, target_tensor, name):
    try:
        # Forward pass through the loss function
        loss_value = loss_fn(input_tensor, target_tensor)

        # Check if the output contains NaN or is not finite
        if torch.isnan(loss_value) or not torch.isfinite(loss_value):
            print(f"[ERROR] Loss function '{name}' produced NaN or invalid value.")
        else:
            print(f"[SUCCESS] Loss function '{name}' output: {loss_value.item()}")
    except Exception as e:
        print(f"[ERROR] Exception occurred in loss function '{name}': {e}")





def edge_loss(pred_depth, gt_depth):
    grad_pred = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]) + \
                torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
    grad_gt = torch.abs(gt_depth[:, :, 1:, :] - gt_depth[:, :, :-1, :]) + \
              torch.abs(gt_depth[:, :, :, 1:] - gt_depth[:, :, :, :-1])
    edge_loss = F.l1_loss(grad_pred, grad_gt)
    return edge_loss