
# This code is to test the model and save the predicted DM with its corresponding GT in a directory   (The accurate final outputs)

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from swin_edge_FFM import DualEncoderModel        # './models_checkpoints/swin_edge_FMM/checkpoint_epoch_10_2024-11-05.pth'   # >> depth maps saved in the folder
# from DataLoad import prepare_dataset
from loadData_official import prepare_dataset

from Functions import EarlyStoppingWithCheckpoint, compute_errors, ScharrEdgeDetector
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def save_predictions(predictions, ground_truths, save_folder="depth_maps"):
    """
    Save each predicted and ground truth depth map in the specified folder with the 'viridis' colormap.
    """
    # Create the folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save each depth map with 'viridis' colormap
    for idx, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
        # Remove extra dimensions if they exist
        prediction = prediction.squeeze()
        ground_truth = ground_truth.squeeze()

        # Save predicted depth map
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.imshow(prediction, cmap='plasma')
        pred_save_path = os.path.join(save_folder, f"depth_prediction_{idx}.png")
        plt.savefig(pred_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved predicted depth map {idx} to {pred_save_path} with 'plasma' colormap")

        # Save ground truth depth map
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.imshow(ground_truth, cmap='plasma')
        gt_save_path = os.path.join(save_folder, f"depth_ground_truth_{idx}.png")
        plt.savefig(gt_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved ground truth depth map {idx} to {gt_save_path} with 'viridis' colormap")




def test(model, dataloader, device):
    model.eval()
    predictions, ground_truths = [], []
    edge_detector = ScharrEdgeDetector()  # Instantiate the edge detector once

    print("Starting testing...")
    with torch.no_grad():
        for rgb, depth, edge in tqdm(dataloader, desc="Testing", leave=False):
            # Move data to the correct device
            rgb, depth, edge = rgb.to(device), depth.to(device), edge.to(device)

            # Generate edges using the edge detector for the input RGB
            greyScale_trans = transforms.Grayscale(num_output_channels=1)
            grayRGB = greyScale_trans(rgb)
            edge_rgb = edge_detector.forward(grayRGB)

            # Forward pass through the model using both RGB and edge
            output = model(rgb, edge_rgb)


            # Store predictions and ground truths for further evaluation
            predictions.append(output.cpu().numpy())
            ground_truths.append(depth.cpu().numpy())

    # Compute errors (optional, if applicable)
    metrics = compute_errors(np.array(ground_truths), np.array(predictions))
    print("Error Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
     # Save the predicted and ground truth depth maps in combined figures
    save_predictions(predictions, ground_truths, save_folder="depth_maps_NEWWW")

    return predictions, ground_truths


if __name__ == "__main__":
    # Hyperparameters for testing
    batch_size = 1  # Set to 1 for testing each sample individually
    num_workers = 4

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset root directory
    root_dir = 'Data'

    # Prepare dataset
    _, _, test_dataset = prepare_dataset(root_dir)


    # Create data loader for the test set
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers= 1)

    # Initialize the model
    model = DualEncoderModel().to(device)

    # Load the best model checkpoint for testing
    # checkpoint_path = './models_checkpoints_2024/final_model.pth'    # swin_Ndecoder (saved maps)
    checkpoint_path = './models_checkpoints_2024/swin_edge_FMM/checkpoint_epoch_10_2024-11-05.pth'

    try:
        checkpoint = torch.load(checkpoint_path)
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint file not found at '{checkpoint_path}'. Please ensure the path is correct.")
        exit()

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters())  # Initialize optimizer to load its state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore epoch and loss values if needed
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    # Set model to evaluation mode
    model.eval()

    # Testing the model using the provided test() function
    predictions, ground_truths = test(model, test_loader, device)
    print("Testing completed.")
