
'''

Copyright (c) 2025 Bashayer Abdallah
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Commercial use is prohibited.

'''




import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL

def extract_label(image_name):
    """
    Given an image name like "sigmoid_t3_b_0073",
    extract the class (e.g., "sigmoid") and texture condition (e.g., "t3")
    and return a descriptive label.
    """
    parts = image_name.split('_')
    if len(parts) < 3:
        raise ValueError("Image name does not follow the expected format.")

    origin_class = parts[0]
    texture_condition = parts[1]
    texture_sub = parts[2]
    return f"The predicted depth Map from {origin_class} with Texture {texture_condition}{texture_sub}"

class c3VD_Dataset(Dataset):

    def __init__(self, data_dir, image_size=518):
        self.data_dir = data_dir
        self.image_size = image_size

        # Lists for file paths
        self.images = []
        self.depths = []
        self.normals = []
        self.labels = []

        # Discover and pair files
        for sub in sorted(os.listdir(self.data_dir)):
            folder = os.path.join(self.data_dir, sub)
            if not os.path.isdir(folder):
                continue
            pngs = sorted(glob.glob(os.path.join(folder, '*.png')))
            for img_path in pngs:
                filename = os.path.basename(img_path)
                base = os.path.splitext(filename)[0]
                sample_id = base[:-6] if base.endswith('_color') else base
                depth_path = os.path.join(folder, sample_id + '_depth.tiff')
                normal_path = os.path.join(folder, sample_id + '_normals.tiff')
                # Validate existence
                if not os.path.isfile(depth_path):
                    raise FileNotFoundError(f"Missing depth map: {depth_path}")
                if not os.path.isfile(normal_path):
                    raise FileNotFoundError(f"Missing normal map: {normal_path}")
                # Record
                self.images.append(img_path)
                self.depths.append(depth_path)
                self.normals.append(normal_path)
                # Label
                combined = f"{sub}_{sample_id}"
                self.labels.append(extract_label(combined))

        # Sanity check
        assert len(self.images)==len(self.depths)==len(self.normals)==len(self.labels), \
            f"Counts mismatch: img={len(self.images)}, dep={len(self.depths)}, norm={len(self.normals)}, lbl={len(self.labels)}"
        self.classes = sorted(set(self.labels))

        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(self.image_size)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load images
        img  = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.images[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        raw = cv2.imread(self.depths[idx], cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Failed to read depth: {self.depths[idx]}")
        depth = np.clip(raw.astype(np.float32)/65535.0,0,1)
        norm = cv2.imread(self.normals[idx], cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        if norm is None:
            raise RuntimeError(f"Failed to read normal: {self.normals[idx]}")
        norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB).astype(np.float32)
        normal = norm/32767.5 -1.0
        mask = ((depth>=0)&(depth<=1)).astype(np.float32)
        # To tensor and resize
        image_t  = self.resize(self.to_tensor(img))
        depth_t  = self.resize(self.to_tensor(depth))[0:1]
        normal_t = self.resize(self.to_tensor(normal))
        mask_t   = self.resize(self.to_tensor(mask))[0:1]
        label    = self.labels[idx]
        return image_t, depth_t, mask_t, normal_t, label

