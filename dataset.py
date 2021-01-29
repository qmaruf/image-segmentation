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
import albumentations as A
from albumentations.pytorch import ToTensorV2



class SegmentationDataset(Dataset):
	def __init__(self, images, masks, training):
		self.images = images
		self.masks = masks  
		self.training = training
		# self.transforms_img = tvt.Compose([        
		#     tvt.ToTensor(),
		#     tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
		# self.transforms_lbl = tvt.Compose([        
		#     tvt.ToTensor()])   
		
		
		self.semantic_aug = A.Compose([    
			A.HorizontalFlip(p=0.5),		
		])
		self.train_aug = A.Compose([
			A.HorizontalFlip(p=0.5),
			A.OneOf([
				A.IAAAdditiveGaussianNoise(),
				A.GaussNoise(),
				A.MotionBlur(),
				A.MedianBlur(),
				A.Blur(),
				A.CLAHE(),
				A.IAASharpen(),
				A.IAAEmboss(),
				A.RandomBrightnessContrast(),            
				A.HueSaturationValue()
			]),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))        
		])
		
		self.eval_aug = A.Compose([
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))        
		])
		
		self.to_tensor = ToTensorV2()


	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		sample = dict()
	
		image = Image.open(self.images[item])
		mask = Image.open(self.masks[item]) 
		image, mask = np.array(image), np.array(mask)
		sample['original_image'] = self.to_tensor(image=image)['image']
	
		if self.training:
			augmented = self.semantic_aug(image=image, mask=mask)
			image, mask = augmented['image'], augmented['mask']
			image = self.train_aug(image=image)['image']    	
		else:
			image = self.eval_aug(image=image)['image']    
		
		image = self.to_tensor(image=image)['image']			
		sample["image"] = image    		
		mask = (mask==config.TARGET_CLASS_ID).astype(np.float32)
		sample["mask"] = self.to_tensor(image=mask)['image']
		
		return sample
	
def create_data_loader(images, masks, batch_size, training):
	dataset = SegmentationDataset(images = images, masks = masks, training=training)
	dataloader =  DataLoader(dataset, batch_size=batch_size, num_workers=12)
	return dataloader

def read_data():    
	masks = glob(config.LABELS_DIR + '*.png')
	if config.SAMPLE:
		masks = masks[:10]
	images = [config.IMGS_DIR + Path(mask).stem.split('_')[0] + '.jpg' for mask in masks]
	images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.1,random_state=config.RANDOM_SEED)
	images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=0.2,random_state=config.RANDOM_SEED)
	return images_train, masks_train, images_val, masks_val, images_test, masks_test

def get_data_loader():
	images_train, masks_train, images_val, masks_val, images_test, masks_test = read_data()
	train_dl = create_data_loader(images_train, masks_train, config.TRAIN_BATCH_SIZE, training=True)
	val_dl = create_data_loader(images_val, masks_val, config.VAL_BATCH_SIZE, training=False)
	test_dl = create_data_loader(images_test, masks_test, config.TEST_BATCH_SIZE, training=False)    
	return train_dl, val_dl, test_dl

