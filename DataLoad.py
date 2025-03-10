

# Define the dataset class and prepare dataset function

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image

class ColonDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_transform = depth_transform
        self.rgb_images = []
        self.depth_images = []
        self.edge_images = []

        # Load image paths from all subdirectories
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.startswith("FrameBuffer_") and file_name.endswith(".png"):
                    rgb_path = os.path.join(root, file_name)
                    depth_name = file_name.replace("FrameBuffer_", "Depth_")
                    depth_path = os.path.join(root, depth_name)
                    edge_name = file_name.replace("FrameBuffer_", "Edge_")
                    edge_path = os.path.join(root, edge_name)

                    if os.path.isfile(rgb_path) and os.path.isfile(depth_path) and os.path.isfile(edge_path):
                        self.rgb_images.append(rgb_path)
                        self.depth_images.append(depth_path)
                        self.edge_images.append(edge_path)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        # Load RGB, depth, and edge images
        rgb_image = Image.open(self.rgb_images[idx]).convert('RGB')
        depth_image = Image.open(self.depth_images[idx])
        edge_image = Image.open(self.edge_images[idx])

        # Apply transformations to convert PIL Images to tensors
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
            edge_image = self.depth_transform(edge_image)

        # Ensure all returned items are tensors
        if not isinstance(rgb_image, torch.Tensor):
            raise TypeError(f"RGB image after transformation is not a tensor, got {type(rgb_image)}")
        if not isinstance(depth_image, torch.Tensor):
            raise TypeError(f"Depth image after transformation is not a tensor, got {type(depth_image)}")
        if not isinstance(edge_image, torch.Tensor):
            raise TypeError(f"Edge image after transformation is not a tensor, got {type(edge_image)}")

        return rgb_image, depth_image, edge_image

def prepare_dataset(root_dir):
    # Define appropriate transforms for RGB and depth images for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to larger size first
        transforms.RandomCrop((224, 224)),  # Random crop to final size for augmentation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
        transforms.RandomRotation(10),  # Random rotation for augmentation
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    train_depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match RGB image
        transforms.RandomCrop((224, 224)),  # Random crop to match RGB crop
        transforms.RandomHorizontalFlip(),  # Random flip to match RGB flip
        transforms.RandomRotation(10),  # Random rotation to match RGB rotation
        transforms.ToTensor(),  # Convert to PyTorch tensor & Normalize depth values to [0, 1]
    ])

    # Define appropriate transforms for RGB and depth images for validation/testing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize directly to the required input size
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_test_depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match RGB image
        transforms.ToTensor(),  # Convert to PyTorch tensor & Normalize depth values to [0, 1]
    ])

    # Load the raw dataset without initial transformations
    print("Loading dataset...")
    full_dataset = ColonDepthDataset(root_dir=root_dir)

    # Split dataset into train, validation, and test sets >> training (80%), validation (10%), and testing (10%)
    print("Splitting dataset into training, validation, and test sets...")
    train_size = int(0.80 * len(full_dataset))
    val_size = int(0.10 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Training set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")

    # Assign appropriate transformations for each split
    # Training dataset with augmentation
    train_dataset.dataset.transform = train_transform
    train_dataset.dataset.depth_transform = train_depth_transform

    # Validation and test datasets without augmentation
    val_dataset.dataset.transform = val_test_transform
    val_dataset.dataset.depth_transform = val_test_depth_transform

    test_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.depth_transform = val_test_depth_transform

    return train_dataset, val_dataset, test_dataset
