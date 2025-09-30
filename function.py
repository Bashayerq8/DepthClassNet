'''
Copyright (c) 2025 Bashayer Abdallah
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Commercial use is prohibited.
'''


import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn
import torch.nn as nn


# Training loop with early stopping and checkpoint saving
class earlyStoppingWithCheckpoint:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

        print(f"Checkpoint saved: validation loss = {self.best_loss:.4f}")




# Evaluate the model on the validation dataset
def evaluate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, text_tokens in dataloader:
            images = images.to(device)
            text_tokens = text_tokens.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, text_tokens)

            # Get predictions
            image_preds = torch.argmax(logits_per_image, dim=1)
            text_preds = torch.argmax(logits_per_text, dim=1)

            # Calculate accuracy
            correct_predictions += (image_preds == text_preds).sum().item()
            total_predictions += images.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


def compute_errors(gt, pred):
    eps = 1e-6  # Small value to avoid log(0)
    gt = np.clip(gt, eps, 1.0)
    pred = np.clip(pred, eps, 1.0)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)



# a gradient-based edge detection method to generate the Edge map
class scharrEdgeDetector(nn.Module):                     
    def __init__(self):
        super(ScharrEdgeDetector, self).__init__()
        self.scharr_x = None
        self.scharr_y = None

    def forward(self, image):
        in_channels = image.size(1)  # Get the number of input channels

        if self.scharr_x is None or self.scharr_x.in_channels != in_channels:
            # Adjusted Scharr kernels for multi-channel input
            scharr_kernel_x = torch.tensor([[3., 0., -3.],
                                            [10., 0., -10.],
                                            [3., 0., -3.]], dtype=torch.float32)
            scharr_kernel_y = torch.tensor([[3., 10., 3.],
                                            [0.,  0., 0.],
                                            [-3., -10., -3.]], dtype=torch.float32)

            # Expand kernels to match the number of input channels
            scharr_kernel_x = scharr_kernel_x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
            scharr_kernel_y = scharr_kernel_y.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]

            # Repeat the kernel for each input channel
            scharr_kernel_x = scharr_kernel_x.repeat(in_channels, 1, 1, 1)  # Shape: [in_channels, 1, 3, 3]
            scharr_kernel_y = scharr_kernel_y.repeat(in_channels, 1, 1, 1)  # Shape: [in_channels, 1, 3, 3]

            # Define convolution layers with appropriate in_channels and out_channels
            self.scharr_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            self.scharr_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

            # Set the weights of the convolution layers
            self.scharr_x.weight = nn.Parameter(scharr_kernel_x, requires_grad=False)
            self.scharr_y.weight = nn.Parameter(scharr_kernel_y, requires_grad=False)

            # Move to GPU if necessary
            if image.is_cuda:
                self.scharr_x = self.scharr_x.cuda()
                self.scharr_y = self.scharr_y.cuda()

        # Apply the Scharr filter to the input image
        grad_x = self.scharr_x(image)
        grad_y = self.scharr_y(image)

        # Calculate gradient magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # Adding epsilon to avoid sqrt(0)

        # Normalise the magnitude to [0, 1]
        magnitude_min = magnitude.view(magnitude.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        magnitude_max = magnitude.view(magnitude.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        magnitude = (magnitude - magnitude_min) / (magnitude_max - magnitude_min + 1e-6)  # Adding epsilon for stability

        return magnitude



