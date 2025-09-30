
# Copyright (c) 2025 Bashayer Abdallah
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Commercial use is prohibited.

'''
The DataLoader creates batches by combining multiple samples, returning the RGB, Depth, Edge images, and Class Labels for the batch.
The dataset split follows SoftEnNet (https://arxiv.org/abs/2301.08157)
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import re



# Extract main and sub classes from the base name
def extract_classes(base_name):
    pattern = r'C_(T[123])_(L[1-5])'
    match = re.search(pattern, base_name)
    if match:
        main_class = match.group(1)
        sub_class = match.group(2)
        combined_class = f"{main_class}_{sub_class}"
        # print(f"DEBUG: Extracted class: {combined_class} from {base_name}")
        return combined_class
    # print(f"DEBUG: Failed to extract class from: {base_name}")
    return None


# Define the ColonDepthDataset with class labels
class colonDepthDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_transform = depth_transform
        self.rgb_images = []
        self.depth_images = []
        self.edge_images = []
        self.classes = []

        # Build file paths and extract class labels
        for base_file, index in file_list:
            rgb_path = os.path.join(root_dir, f"{base_file}_FrameBuffer_{index}.png")
            depth_path = os.path.join(root_dir, f"{base_file}_Depth_{index}.png")
            edge_path = os.path.join(root_dir, f"{base_file}_Edge_{index}.png")

            if os.path.isfile(rgb_path) and os.path.isfile(depth_path) and os.path.isfile(edge_path):
                combined_class = extract_classes(base_file)
                if combined_class and combined_class in class_map:
                    self.rgb_images.append(rgb_path)
                    self.depth_images.append(depth_path)
                    self.edge_images.append(edge_path)
                    self.classes.append(class_map[combined_class])
                else:
                    print(f"DEBUG: Invalid class or missing mapping for file: {base_file}")
            # else:
        #         print(f"DEBUG: Missing files for base: {base_file}, index: {index}")
        #         print(f"  RGB Path Exists: {os.path.isfile(rgb_path)}")
        #         print(f"  Depth Path Exists: {os.path.isfile(depth_path)}")
        #         print(f"  Edge Path Exists: {os.path.isfile(edge_path)}")
        #
        # print(f"DEBUG: Initialized dataset with {len(self.rgb_images)} samples")


    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_images[idx]).convert('RGB')
        depth_image = Image.open(self.depth_images[idx])
        edge_image = Image.open(self.edge_images[idx])
        class_label = self.classes[idx]

        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
            edge_image = self.depth_transform(edge_image)

        return rgb_image, depth_image, edge_image, class_label



# Utility to load file list from text files
def load_file_list(file_path):
    """
    Load file list from text file, parsing base filenames and indices.
    Each line in the file should have two parts: base_file and index.
    """
    file_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_list.append((parts[0], parts[1]))
            else:
                print(f"DEBUG: Skipping malformed line: {line.strip()}")
    # print(f"DEBUG: Loaded {len(file_list)} entries from {file_path}")
        return file_list



# Prepare train, validation, and test datasets
def prepare_dataset(root_dir, train_file, val_file, test_file):
    train_files = load_file_list(train_file)
    val_files = load_file_list(val_file)
    test_files = load_file_list(test_file)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = ColonDepthDataset(root_dir=root_dir, file_list=train_files, transform=train_transform, depth_transform=train_depth_transform)
    val_dataset = ColonDepthDataset(root_dir=root_dir, file_list=val_files, transform=val_test_transform, depth_transform=val_test_depth_transform)
    test_dataset = ColonDepthDataset(root_dir=root_dir, file_list=test_files, transform=val_test_transform, depth_transform=val_test_depth_transform)

    # print(f"DEBUG: Train dataset size: {len(train_dataset)}")
    # print(f"DEBUG: Validation dataset size: {len(val_dataset)}")
    # print(f"DEBUG: Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset





