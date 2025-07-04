import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.transforms as T
import time
import json

import torch.utils.data as data

RGB_TRAIN_MEAN = np.load("utilities/rgb_train_mean.npy")
RGB_TRAIN_STD  = np.load("utilities/rgb_train_std.npy") 

class DFC18(data.Dataset):
    def __init__(self, split="train", as_tensor=False, normalize=False, standardize=False, augment=None, add_blur_channels=False, blur_kernel=10.0):
        """
        Args:
            split (str): One of ['train', 'val', 'test']
            as_tensor (bool): Convert to torch.Tensor
            normalize (bool): Normalize to [0,1]
            standardize (bool): Standardize with train mean & var
            augment (callable): Albumentations-style transform
            add_blur_channels (bool): Add blurred channels
            blur_kernel (float): Kernel size

        """
        self.root = os.path.join('data/patches/', split)
        self.as_tensor = as_tensor
        self.normalize = normalize
        self.standardize = normalize
        self.augment = augment
        self.add_blur_channels = add_blur_channels
        self.blur_kernel = blur_kernel
        self.n_classes = 21  

        # Class Frequencies
        self.freq = np.array([0, 1.90555569e-02, 6.57992732e-02, 1.36151029e-03, 2.67724482e-02,
                    9.97414111e-03, 8.04829999e-03, 4.61384373e-04, 7.72459578e-02,
                    4.54671975e-01, 8.62569455e-02, 5.70884499e-02, 3.10250849e-03,
                    9.54861768e-02, 2.00251306e-02, 1.39215162e-02, 2.25124844e-02,
                    2.54609286e-04, 1.32041993e-02, 1.08724781e-02, 1.38849544e-02])

        # Class info
        self.class_info = class_info = {
            0:  ('Unclassified',            (0,   0,   0  )),    # Black / No data
            1:  ('Healthy grass',           (0,  255,   0  )),    # Bright green
            2:  ('Stressed grass',          (128, 128,  0  )),    # Olive / dull green
            3:  ('Artificial turf',         (0,  200,  0  )),    # Medium green
            4:  ('Evergreen trees',         (0,  100,   0  )),    # Dark green
            5:  ('Deciduous trees',         (34, 139,  34  )),    # Forest green
            6:  ('Bare earth',              (210, 180, 140 )),    # Tan / sandy brown
            7:  ('Water',                   (0,    0, 255  )),    # Blue
            8:  ('Residential buildings',  (255,  0,    0  )),    # Red
            9:  ('Non-residential buildings',(139, 0,    0  )),   # Dark red
            10: ('Roads',                   (128, 128, 128 )),    # Gray
            11: ('Sidewalks',               (192, 192, 192 )),    # Light gray
            12: ('Crosswalks',              (255, 255, 255 )),    # White
            13: ('Major thoroughfares',     (255, 165,   0  )),    # Orange
            14: ('Highways',                (255, 140,   0  )),    # Dark orange
            15: ('Railways',                (139, 69,   19  )),    # Saddle brown
            16: ('Paved parking lots',      (169, 169, 169 )),    # Dark gray
            17: ('Unpaved parking lots',    (205, 133,  63  )),    # Peru brown
            18: ('Cars',                    (255, 20,    147 )),   # Deep pink
            19: ('Trains',                  (75,  0,   130  )),    # Indigo / purple
            20: ('Stadium seats',           (255, 215,   0  )),    # Gold / yellow
        }

        # Find all image-label pairs
        self.pairs = []
        for file in os.listdir(self.root):
            if file.endswith("_img.npy"):
                base = file.replace("_img.npy", "")
                label_path = os.path.join(self.root, base + "_label.npy")
                if os.path.exists(label_path):
                    self.pairs.append((os.path.join(self.root, file), label_path))

        print(f"Loaded {len(self.pairs)} samples from {split} set")

        # Load majority classes JSON (assuming you have this file)
        json_path = f"utilities/{split}_majority_class_per_image.json"
        with open(json_path, "r") as f:
            majority_classes = json.load(f)

        assert len(majority_classes) == len(self.pairs), \
            "Mismatch between majority classes count and samples count"

        self.majority_classes = majority_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        image = np.load(img_path).astype(np.float32) 
        label = np.load(label_path).astype(np.int64) 
        majority_class = self.majority_classes[idx]

        if self.normalize:
            image = np.clip(image, 0, 255) / 255.0

        elif self.standardize:
            mean = RGB_TRAIN_MEAN.astype(np.float32)
            std  = RGB_TRAIN_STD.astype(np.float32)
            # broadcast mean/std: (3,) -> (3, 1, 1)
            mean = mean[:, None, None]
            std  = std[:, None, None]
            image = (image - mean) / std

        if self.augment:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC for augmentation
            augmented = self.augment(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            image = np.transpose(image, (2, 0, 1))  # back to CHW for PyTorch

        if self.as_tensor:
            image = torch.tensor(image)
            label = torch.tensor(label, dtype=torch.long)

            # add blurred channels
            if self.add_blur_channels:
                blur = T.GaussianBlur(kernel_size=self.blur_kernel, sigma=(self.blur_kernel - 1) / 6)
                blurred = blur(image)
                image = torch.cat([image, blurred], dim=0) # should always be torch at this point

        return {
            'image': image,
            'label': label,
            'id': os.path.basename(img_path).replace("_img.npy", ""),
            'majority_class': majority_class
        }

