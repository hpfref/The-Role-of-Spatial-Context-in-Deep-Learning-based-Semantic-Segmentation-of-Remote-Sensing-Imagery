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


### inspiration: https://github.com/schmitt-muc/SEN12MS/blob/master/classification/dataset.py, https://github.com/lukasliebel/dfc2020_baseline?tab=readme-ov-file

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]
S2_BANDS_ALL = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# valid classes
DFC20_LABEL_MAP = {
    1: 0,   # Forest
    2: 1,   # Shrubland
    4: 2,   # Grassland
    5: 3,   # Wetlands
    6: 4,   # Croplands
    7: 5,   # Urban/Built-up
    9: 6,   # Barren
    10: 7   # Water
}

S2_TRAIN_MEAN = np.load("utilities/s2_train_mean.npy")  # shape (13,)
S2_TRAIN_STD  = np.load("utilities/s2_train_std.npy")   # shape (13,)

# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read([1, 2])  # VV = band 1, VH = band 2

    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)

    # default to normalization 
    s1 = np.clip(s1, -25, 0)
    s1 = (s1 + 25) / 25

    return s1

# util function for reading s2 data
def load_s2(path, use_s2_RGB, use_s2_hr, use_s2_all, use_s2_hr_mr, normalize, standardize):
    # band selection 
    bands=[]
    if use_s2_RGB: bands = S2_BANDS_RGB
    elif use_s2_hr: bands = S2_BANDS_HR
    elif use_s2_all : bands = S2_BANDS_ALL
    elif use_s2_hr_mr: bands = S2_BANDS_HR + S2_BANDS_MR

    with rasterio.open(path) as data:
        s2 = data.read(bands)
        s2 = s2.astype(np.float32)

    if normalize:
      #s2 = np.clip(s2 / np.max(s2), 0, 1)
      s2 = np.clip(s2, 0, 10000) 
      s2 /= 10000

    elif standardize:
      # pick selected bands
      mean = S2_TRAIN_MEAN[np.array(bands) - 1].astype(np.float32)
      std  = S2_TRAIN_STD[np.array(bands) - 1].astype(np.float32)
      # broadcast mean/std: (13,) -> (13, 1, 1)
      mean = mean[:, None, None]
      std  = std[:, None, None]
      s2 = (s2 - mean) / std

    return s2

# util function for reading dfc data
def load_dfc(path):

    # load labels
    with rasterio.open(path) as data:
        dfc = data.read(1)

    # convert to zero-based labels and set ignore mask
    #lc -= 1
    #lc[lc == -1] = 255
    return dfc

# util function for reading data from single sample
def load_sample(sample, use_s1, use_s2_RGB, use_s2_hr, use_s2_all, use_s2_hr_mr, normalize, standardize):
    #total_start = time.time()
    #times = {}

    use_s2 = use_s2_RGB or use_s2_hr or use_s2_all or use_s2_hr_mr

    # load s1/s2
    #t0 = time.time()
    if use_s1:
        img = load_s1(sample["s1"])
        #times['load_s1s2'] = 0
    elif use_s2:
        img = load_s2(sample["s2"], use_s2_RGB, use_s2_hr, use_s2_all, use_s2_hr_mr, normalize, standardize)
        #times['load_s1s2'] = time.time() - t0
    else:
      img = None

    # load labels
    #t1 = time.time()
    dfc = load_dfc(sample["dfc"])
    #times['load_dfc'] = time.time() - t1

    # remap labels
    #t2 = time.time()
    dfc = np.vectorize(DFC20_LABEL_MAP.get)(dfc).astype(np.float32)
    #times['remap'] = time.time() - t2

    #total_time = time.time() - total_start
    #print(f"Total: {total_time:.4f}s | load_s1s2: {times['load_s1s2']:.4f}s | load_dfc: {times['load_dfc']:.4f}s | remap: {times['remap']:.4f}s")

    return {'image': img, 'label': dfc, 'id': sample["id"]}


#  calculate number of input channels  
def get_ninputs(use_s1, use_s2_RGB, use_s2_hr, use_s2_all, use_s2_hr_mr):
    n_inputs = 0
    if use_s2_hr:
        n_inputs = len(S2_BANDS_HR)
    elif use_s2_all:
        n_inputs = len(S2_BANDS_ALL)
    elif use_s1:
        n_inputs = 2
    elif use_s2_RGB :
        n_inputs = 3
    elif use_s2_hr_mr:
        n_inputs = len(S2_BANDS_HR) + len(S2_BANDS_MR)
        
    return n_inputs




# class SEN12MS..............................
class DFC20(data.Dataset):
    """PyTorch dataset class for the DFC20 dataset"""
    # expects dataset dir as:
    #       -

    def __init__(self, path, subset="train", use_s1=False, use_s2_RGB=False, use_s2_hr=False, use_s2_all=False, use_s2_hr_mr=False, as_tensor=False, 
                 normalize=False, standardize=False, augment=None, add_blur_channels=False, blur_kernel=10.0, in_memory=False):
        """Initialize the dataset"""

        # inizialize
        super(DFC20, self).__init__()

        # make sure input parameters are okay
        if not (use_s1 or use_s2_RGB or use_s2_hr or use_s2_all or use_s2_hr_mr):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        
        self.use_s1 = use_s1
        self.use_s2_RGB = use_s2_RGB
        self.use_s2_hr = use_s2_hr 
        self.use_s2_hr_mr = use_s2_hr_mr
        self.use_s2_all = use_s2_all 
        self.as_tensor = as_tensor
        self.normalize = normalize
        self.standardize = standardize
        self.augment = augment
        self.add_blur_channels = add_blur_channels
        self.blur_kernel = blur_kernel

        self.in_memory = in_memory
        assert subset in ["train", "val", "test"]
        
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2_RGB, use_s2_hr, use_s2_hr_mr, use_s2_all) # excluding blur channels

        # make sure parent dir exists
        assert os.path.exists(path)

        # number of classes 
        self.n_classes = 8

        # classnames with colors
        self.class_info = {
            0: ("Forest", "#009900"),
            1: ("Shrubland", "#c6b044"),
            #3: ("Savanna", "#fbf113"), #not in data
            2: ("Grassland", "#b6ff05"),
            3: ("Wetlands", "#27ff87"),
            4: ("Croplands", "#c24f44"),
            5: ("Urban/Built-up", "#a5a5a5"),
            #8: ("Snow/Ice", "#69fff8"), #not in data
            6: ("Barren", "#f9ffa4"),
            7: ("Water", "#1c0dff")
        }

        # Class Frequencies
        self.freq = np.array([23.1, 6.0, 11.6, 7.0, 17.0, 11.0, 2.3, 22.0])

        # get samples
        self.samples = []

        if subset == "train":
            sample_dir = os.path.join(path, "train")
            with open("utilities/train_majority_class_per_image.json", "r") as f:
                self.image_majority_class = json.load(f)
        elif subset == "val":
            sample_dir = os.path.join(path, "val")
            with open("utilities/val_majority_class_per_image.json", "r") as f:
                self.image_majority_class = json.load(f)
        else:
            sample_dir = os.path.join(path, "test")
            with open("utilities/test_majority_class_per_image.json", "r") as f:
                self.image_majority_class = json.load(f)

        # Get all s2 patches in the corresponding subset (train/val/test)
        s2_files = glob.glob(os.path.join(sample_dir, "s2", "*.tif"))

        # Set up progress bar for loading samples
        pbar = tqdm(total=len(s2_files), desc="[Load]")

        # Loop over the list of s2 files and find corresponding s1 & dfc files
        for s2_loc in s2_files:
            # Extract the sample ID from the filename
            s2_id = os.path.basename(s2_loc)
            #mini_name = s2_id.split("_")

            # Build paths for s1 and dfc based on s2_id
            s1_loc = s2_loc.replace("s2", "s1")
            dfc_loc = s2_loc.replace("s2", "dfc")

            pbar.update()

            # Append the sample info to the samples list
            self.samples.append({"s1": s1_loc, "s2": s2_loc, "dfc": dfc_loc, "id": s2_id})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        # Preload the data into memory
        if self.in_memory:
            self.preloaded_data = []
            for sample in self.samples:
                sample_data = load_sample(sample, self.use_s1, self.use_s2_RGB, self.use_s2_hr, self.use_s2_hr_mr, self.use_s2_all, self.normalize, self.standardize)
                self.preloaded_data.append(sample_data)

        print("loaded", len(self.samples), "samples from the DFC20 subset", subset)
        
        

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        if self.in_memory:
            # Retrieve the preloaded sample data
            sample = self.preloaded_data[index]
        else:
            # get and load sample from index file
            sample = self.samples[index]
            # Load the sample and apply the transformations
            sample = load_sample(sample, self.use_s1, self.use_s2_RGB, self.use_s2_hr, self.use_s2_hr_mr, self.use_s2_all, self.normalize, self.standardize)

        # Apply augmentation if provided
        if self.augment:
            image = sample['image']
            image = np.transpose(image, (1, 2, 0)) # for rotation, ...
            label = sample['label']
            augmented = self.augment(image=image, mask=label)
            image = augmented['image']
            image = np.transpose(image, (2, 0, 1))
            label = augmented['mask']
        else:
            image = sample['image']
            label = sample['label']

        # add majority label
        majority_class = self.image_majority_class[index]

        # convert to tensor
        if self.as_tensor:
            image = torch.tensor(image)
            label = torch.tensor(label, dtype=torch.long)
            majority_class = torch.tensor(majority_class, dtype=torch.long)

            # add blurr channels
            if self.add_blur_channels:
                blur = T.GaussianBlur(kernel_size=self.blur_kernel, sigma=(self.blur_kernel - 1) / 6)
                blurred = blur(image)
                image = torch.cat([image, blurred], dim=0) # should always be torch at this point
            
        return {'image': image, 'label': label, 'id': sample["id"], 'majority_class': majority_class}


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)          
    

if __name__ == "__main__":
    
    path = "./data"

    ds = DFC20(path, subset="test", use_s1=False, use_s2_RGB=True, use_s2_hr=False, use_s2_all=False)
    patch = ds.__getitem__(100)
    print("id:", patch["id"], "\n",
          "input shape:", patch["image"].shape, "\n",
          "number of classes", ds.n_classes)
    

