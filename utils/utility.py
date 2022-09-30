import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path


class SegmentationDataset(Dataset):
    
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.images
        mask_path = row.masks

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        # mask = preprocess_mask(mask)
        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        # (h, w c) -> (c, h, w)
        image = np.transpose(image, (2, 1, 0)).astype(np.float32)
        mask = np.transpose(mask, (2, 1, 0)).astype(np.float32)

        image = torch.Tensor(image) / 255.0 # image values from 0 to 1
        mask = torch.round(torch.Tensor(mask) / 255.0) # mask values 0 or 1
        #mask = torch.round(mask) #/ 255.0)

        return image, mask 


class SimpleDataset(torch.utils.data.Dataset):
    """Single Folder with images dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = list(Path(root_dir).glob(r'*.[jpb][pnm][jgp]'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.cpu().numpy())

        image = cv2.imread(str(self.images[idx]))
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]

        # (h, w c) -> (c, h, w)
        image = np.transpose(image, (2, 1, 0)).astype(np.float32)
        image = torch.Tensor(image) / 255.0

        return image, str(self.images[idx]), height, width


def get_train_augs(args):

    crop_width = args.imgsz #608
    crop_height = args.imgsz #608

    return A.Compose([
        A.PadIfNeeded(min_height=crop_height, min_width=crop_width),
        A.CropNonEmptyMaskIfExists(height=crop_height, width=crop_width, p=0.9),
        A.RandomCrop(height=crop_height, width=crop_width),
        # A.Resize(height=resize_height, width=resize_width),
        # A.RandomCrop(height=crop_height, width=crop_width),
        A.RandomRotate90(p=0.5),
        #A.Rotate(270),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0), # new
        # A.Perspective(p=0.1),
        # A.ToGray(p=0.1),
        # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3)
        #A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.3),
        #A.JpegCompression(quality_lower=0, quality_upper=1, p=0.5),
        #A.Cutout(num_holes=30, max_h_size=30, max_w_size=30, fill_value=128, p=0.5)
        #A.ElasticTransform(), # new
        #A.GridDistortion(), # new
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2()
    ])

def get_val_augs():
    return A.Compose([
                    A.PadIfNeeded(min_height=None, 
                                  min_width=None,
                                  pad_height_divisor=32,
                                  pad_width_divisor=32),
    ])