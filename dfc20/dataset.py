import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import torch.utils.data as data


### inspiration: https://github.com/schmitt-muc/SEN12MS/blob/master/classification/dataset.py, https://github.com/lukasliebel/dfc2020_baseline?tab=readme-ov-file

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]
S2_BANDS_ALL = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# util function for reading s2 data
def load_s2(path, use_s2_RGB, use_s2_hr, use_s2_all, as_tensor):
    # band selection 
    bands=[]
    if use_s2_RGB: bands = S2_BANDS_RGB
    elif use_s2_hr: bands = S2_BANDS_HR
    elif use_s2_all : bands = S2_BANDS_ALL

    with rasterio.open(path) as data:
        s2 = data.read(bands)

    # Normalization - maybe standardization with band means better? -> esp. for brightness?
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000

    if as_tensor:
        s2 = to_tensor(s2)

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
def load_sample(sample, use_s1, use_s2_RGB, use_s2_hr, use_s2_all, as_tensor):

    use_s2 = use_s2_RGB or use_s2_hr or use_s2_all

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2_RGB, use_s2_hr, use_s2_all, as_tensor)

    # load s1 data
    #if use_s1:
    #    if use_s2:
    #        img = np.concatenate((img, load_s1(sample["s1"])), axis=0)
    #    else:
    #        img = load_s1(sample["s1"])

    # load label
    dfc = load_dfc(sample["dfc"])

    return {'image': img, 'label': dfc, 'id': sample["id"]}


#  calculate number of input channels  
def get_ninputs(use_s1, use_s2_RGB, use_s2_hr, use_s2_all):
    n_inputs = 0
    if use_s2_hr:
        n_inputs = len(S2_BANDS_HR)
    elif use_s2_all:
        n_inputs = len(S2_BANDS_ALL)
    elif use_s1:
        n_inputs = 2
    elif use_s2_RGB :
        n_inputs = 3
        
    return n_inputs


def to_tensor(sample):
        img, label, sample_id = sample['image'], sample['label'], sample['id']
        
        sample = {'image': torch.tensor(img), 'label':label, 'id':sample_id}
        return sample



# class SEN12MS..............................
class DFC20(data.Dataset):
    """PyTorch dataset class for the DFC20 dataset"""
    # expects dataset dir as:
    #       -

    def __init__(self, path, subset="train", use_s1=False, use_s2_RGB=False, use_s2_hr=False, use_s2_all=False, as_tensor=False):
        """Initialize the dataset"""

        # inizialize
        super(DFC20, self).__init__()

        # make sure input parameters are okay
        if not (use_s1 or use_s2_RGB or use_s2_hr or use_s2_all):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        
        self.use_s1 = use_s1
        self.use_s2_RGB = use_s2_RGB
        self.use_s2_hr = use_s2_hr 
        self.use_s2_all = use_s2_all 
        self.as_tensor = as_tensor
        
        assert subset in ["train", "val", "test"]
        
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2_RGB, use_s2_hr, use_s2_all)

        # provide number of classes 
        self.n_classes = 10

        # make sure parent dir exists
        assert os.path.exists(path)

        # classnames with colors
        self.class_info = {
            1: ("Forest", "#009900"),
            2: ("Shrubland", "#c6b044"),
            3: ("Savanna", "#fbf113"),
            4: ("Grassland", "#b6ff05"),
            5: ("Wetlands", "#27ff87"),
            6: ("Croplands", "#c24f44"),
            7: ("Urban/Built-up", "#a5a5a5"),
            8: ("Snow/Ice", "#69fff8"),
            9: ("Barren", "#f9ffa4"),
            10: ("Water", "#1c0dff")
        }

        # get samples
        self.samples = []

        if subset == "train":
            sample_dir = os.path.join(path, "train")
        elif subset == "val":
            sample_dir = os.path.join(path, "val")
        else:
            sample_dir = os.path.join(path, "test")

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

        print("loaded", len(self.samples), "samples from the DFC20 subset", subset)
        
        

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        return load_sample(sample, self.use_s1, self.use_s2_RGB, self.use_s2_hr, self.use_s2_all, self.as_tensor)


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)          
    



        
        
# DEBUG usage examples
if __name__ == "__main__":
    
    path = "./data"

    ds = DFC20(path, subset="test", use_s1=False, use_s2_RGB=True, use_s2_hr=False, use_s2_all=False)
    patch = ds.__getitem__(100)
    print("id:", patch["id"], "\n",
          "input shape:", patch["image"].shape, "\n",
          "number of classes", ds.n_classes)
    

"""
bands_mean_train = {'s1_mean': [x, x],
                    's2_mean': [1275.56366148,1038.57734603,949.87916508,814.60094421,1049.14282086,1747.35075034,
                                2033.31146565,1991.47800801,2195.7438094,800.6378601,12.03797621,1521.99609528,970.35119174]}

bands_std_train = {'s1_mean': [x, x],
                   's2_mean': [ 203.10894865,269.65605412,309.13100577,482.80068554,490.5296078,928.52234092,1171.08669927,
                               1181.02504077,1297.944547,500.73803514,7.11942401,990.00060658,765.27912456]}
                               
                               
                            

class Normalize(object): # actually standardiziation
    def __init__(self, bands_mean, bands_std):
        
        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']

        self.bands_s2_mean = bands_mean['s2_mean']
        self.bands_s2_std = bands_std['s2_std']
        
        self.bands_RGB_mean = bands_mean['s2_mean'][0:3]
        self.bands_RGB_std = bands_std['s2_std'][0:3]
        
        self.bands_all_mean = self.bands_s2_mean + self.bands_s1_mean
        self.bands_all_std = self.bands_s2_std + self.bands_s1_std

    def __call__(self, rt_sample):

        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']

        # different input channels
        if img.size()[0] == 12:
            for t, m, s in zip(img, self.bands_all_mean, self.bands_all_std):
                t.sub_(m).div_(s) 
        elif img.size()[0] == 10:
            for t, m, s in zip(img, self.bands_s2_mean, self.bands_s2_std):
                t.sub_(m).div_(s)          
        elif img.size()[0] == 5:
            for t, m, s in zip(img, 
                               self.bands_RGB_mean + self.bands_s1_mean,
                               self.bands_RGB_std + self.bands_s1_std):
                t.sub_(m).div_(s)                                
        elif img.size()[0] == 3:
            for t, m, s in zip(img, self.bands_RGB_mean, self.bands_RGB_std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(img, self.bands_s1_mean, self.bands_s1_std):
                t.sub_(m).div_(s)            
        
        return {'image':img, 'label':label, 'id':sample_id}

class ToTensor(object):
    """"""Convert ndarrays in sample to Tensors.""""""

    def __call__(self, rt_sample):
        
        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']
        
        rt_sample = {'image': torch.tensor(img), 'label':label, 'id':sample_id}
        return rt_sample
"""