# Old Dataset Splits with classes extraction 




import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch


def extract_label(image_name):
    """
    Given an image name like "C_T1_L1_1_resized", extract the class (e.g., T1)
    and subclass (e.g., L1) identifiers and return a descriptive label.
    """
    parts = image_name.split('_')
    if len(parts) < 3:
        raise ValueError("Image name does not follow the expected format.")
    texture_class = parts[1]  # e.g., "T1"
    lighting_condition = parts[2]  # e.g., "L1"
    label = f"The predicted depth Map has Texture {texture_class} under the lighting condition {lighting_condition}"
    return label


def generate_file_list(root_dir):
    """
    Walks through the root directory and generates a list of (base_file, index) pairs.
    It looks for files that contain 'FrameBuffer_' in their name and assumes the name format is:
      <base_file>_FrameBuffer_<index>.png
    """
    file_list = []
    for dirpath, dirnames, files in os.walk(root_dir):
        for file_name in files:
            if "FrameBuffer_" in file_name and file_name.endswith(".png"):
                # Split the file name into base_file and index.
                # Example: "C_T1_L3_5_resized_FrameBuffer_0429.png"
                parts = file_name.split("_FrameBuffer_")
                if len(parts) == 2:
                    base_file = parts[0]
                    index = parts[1].replace(".png", "")
                    # Optionally, check that corresponding depth and edge files exist.
                    rgb_path = os.path.join(dirpath, f"{base_file}_FrameBuffer_{index}.png")
                    depth_path = os.path.join(dirpath, f"{base_file}_Depth_{index}.png")
                    edge_path = os.path.join(dirpath, f"{base_file}_Edge_{index}.png")
                    if os.path.isfile(rgb_path) and os.path.isfile(depth_path) and os.path.isfile(edge_path):
                        file_list.append((base_file, index))
                    else:
                        print(f"DEBUG: Missing corresponding files in {dirpath} for {base_file} and index {index}")
    return file_list


class ColonDepthDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_transform = depth_transform
        self.rgb_images = []
        self.depth_images = []
        self.edge_images = []
        self.classes = []

        # Build file paths and extract labels.
        for base_file, index in file_list:
            # Assuming all files are in the same directory structure (or subdirectories)
            rgb_path = os.path.join(self.root_dir, f"{base_file}_FrameBuffer_{index}.png")
            depth_path = os.path.join(self.root_dir, f"{base_file}_Depth_{index}.png")
            edge_path = os.path.join(self.root_dir, f"{base_file}_Edge_{index}.png")

            if os.path.isfile(rgb_path) and os.path.isfile(depth_path) and os.path.isfile(edge_path):
                label = extract_label(base_file)
                self.rgb_images.append(rgb_path)
                self.depth_images.append(depth_path)
                self.edge_images.append(edge_path)
                self.classes.append(label)
            else:
                print(f"DEBUG: Missing files for base: {base_file}, index: {index}")
                print(f"  Expected RGB: {rgb_path}")
                print(f"  Expected Depth: {depth_path}")
                print(f"  Expected Edge: {edge_path}")

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_images[idx]).convert('RGB')
        depth_image = Image.open(self.depth_images[idx]).convert('L')
        edge_image = Image.open(self.edge_images[idx]).convert('L')
        class_label = self.classes[idx]

        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
            edge_image = self.depth_transform(edge_image)

        return rgb_image, depth_image, edge_image, class_label


def prepare_dataset(root_dir):
    # Generate full file list from the root directory.
    full_file_list = generate_file_list(root_dir)
    print(f"Total entries found: {len(full_file_list)}")

    # Build the full dataset without any transformations.
    full_dataset = ColonDepthDataset(root_dir=root_dir, file_list=full_file_list)

    # Define transformations.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize first
        transforms.RandomCrop((224, 224)),  # Random crop for augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_depth_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize to required size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_test_depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Split dataset into training (80%), validation (10%), and testing (10%)
    print("Splitting dataset into training, validation, and test sets...")
    total_samples = len(full_dataset)
    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Training set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")

    # Assign transformations for each split.
    train_dataset.dataset.transform = train_transform
    train_dataset.dataset.depth_transform = train_depth_transform

    val_dataset.dataset.transform = val_test_transform
    val_dataset.dataset.depth_transform = val_test_depth_transform

    test_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.depth_transform = val_test_depth_transform

    return train_dataset, val_dataset, test_dataset


# # Example usage.
# if __name__ == "__main__":
#     # Set the root directory where your image files are stored.
#     root_dir = "Data/data"  # Adjust this path to your data location
#
#     train_dataset, val_dataset, test_dataset = prepare_dataset(root_dir)
#
#     # Optionally create DataLoaders.
#     train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
#
#     print("Total images in train dataset:", len(train_dataset))
#     print("Total images in validation dataset:", len(val_dataset))
#     print("Total images in test dataset:", len(test_dataset))
#
#     # Check the first sample.
#     rgb, depth, edge, label = train_dataset[0]
#     print("First sample label:", label)
