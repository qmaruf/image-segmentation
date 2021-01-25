import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import config
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pathlib import Path
from torchvision import transforms as tvt
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
  def __init__(self, images, masks, transforms):
    self.images = images
    self.masks = masks  
    self.transforms_img = tvt.Compose([
        # tvt.CenterCrop(config.CENTER_CROP_SIZE),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
    self.transforms_lbl = tvt.Compose([        
        tvt.ToTensor()])   

  def __len__(self):
    return len(self.images)

  def __getitem__(self, item):
    sample = dict()
    image = Image.open(self.images[item])
    sample['original_image'] = tvt.ToTensor()(image)
    sample["image"] = self.transforms_img(image)    
    
    mask = Image.open(self.masks[item]) 
    mask = np.array(mask)
    mask = (mask==config.TARGET_CLASS_ID).astype(np.float32)
    sample["mask"] = self.transforms_lbl(mask)    
    return sample
    
def create_data_loader(images, masks, batch_size):
    dataset = SegmentationDataset(images = images, masks = masks, transforms = None)
    dataloader =  DataLoader(dataset, batch_size=batch_size)
    return dataloader

def read_data():
    masks = glob(config.LABELS_DIR + '*.png')
    images = [config.IMGS_DIR + Path(mask).stem.split('_')[0] + '.jpg' for mask in masks]
    images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.1,random_state=config.RANDOM_SEED)
    images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=0.2,random_state=config.RANDOM_SEED)
    return images_train, masks_train, images_val, masks_val, images_test, masks_test

def get_data_loader():
    images_train, masks_train, images_val, masks_val, images_test, masks_test = read_data()
    train_dl = create_data_loader(images_train, masks_train, config.TRAIN_BATCH_SIZE)
    val_dl = create_data_loader(images_val, masks_val, config.VAL_BATCH_SIZE)
    test_dl = create_data_loader(images_test, masks_test, config.TEST_BATCH_SIZE)    
    return train_dl, val_dl, test_dl

