# Libraries
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from PIL import Image
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

import os, glob
        
class SelfDatasetGenerator(Dataset):
    def __init__(self, data, targets, transform = None, num_classes = 1):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.img_dim = (256, 256)
        self.num_classes = num_classes
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.open(self.data[index])
            x = x.resize(self.img_dim)
            x = self.transform(x)
            
            y = Image.open(self.targets[index])
            y = y.resize(self.img_dim)
            y = np.array(y)
            masks = []
            for i in range(self.num_classes):
                cls_mask = np.where(y == i, 255, 0)
                cls_mask = cls_mask.astype('float')
                cls_mask = cv.resize(cls_mask, (256, 256))

                masks.append(cls_mask[:,:,0] / 255)

            masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        return x, masks
    
    def __len__(self):
        return len(self.data)

class SelfDrivingDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # Input Data
        self.root = "assets/self_driving/"

        sub_dir = []

        for file in os.scandir(self.root):
            if file.is_dir():
                sub_dir.append(file.name)

        self.images = []
        self.masks = []

        for dir in sub_dir:
            images_path = os.path.join(self.root, dir, dir, "CameraRGB/*.png")
            images_path = glob.glob(images_path)
            
            masks_path = os.path.join(self.root, dir, dir, "CameraSeg/*.png")
            masks_path = glob.glob(masks_path)
            
            for img_path, mask_path in zip(images_path, masks_path):
                self.images.append(img_path)
                self.masks.append(mask_path)
        
    def datasetloader(self):
        # set aside 20% of train and test data for evaluation
        x_train, x_test, y_train, y_test = train_test_split(self.images, self.masks,
                                                            test_size = 0.2, shuffle = True, random_state = 8)

        # Use the same function above for the validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                        test_size = 0.25, random_state = 8)

        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = SelfDatasetGenerator(x_train, y_train, transform, num_classes = 13)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)

        validation_dataset = SelfDatasetGenerator(x_val, y_val, transform, num_classes = 13)
        validation_dataloader = DataLoader(validation_dataset, self.batch_size, shuffle=True)

        test_dataset = SelfDatasetGenerator(x_test, y_test, transform, num_classes = 13)
        test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle = False)
        
        return train_dataloader, validation_dataloader, test_dataloader
    
class AerialDatasetGenerator(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.img_dim = (256, 256)
        
        self.BGR_classes = {'Water' : [ 41, 169, 226],
                            'Land' : [246,  41, 132],
                            'Road' : [228, 193, 110],
                            'Building' : [152,  16,  60], 
                            'Vegetation' : [ 58, 221, 254],
                            'Unlabeled' : [155, 155, 155]} # in BGR

        self.bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.open(self.data[index])
            x = x.resize(self.img_dim)
            x = self.transform(x)
            
            y = cv.imread(self.data[index].replace('images', 'masks').replace('.jpg', '.png'))
            cls_mask = np.zeros(y.shape)
            cls_mask[y == self.BGR_classes['Water']] = self.bin_classes.index('Water')
            cls_mask[y == self.BGR_classes['Land']] = self.bin_classes.index('Land')
            cls_mask[y == self.BGR_classes['Road']] = self.bin_classes.index('Road')
            cls_mask[y == self.BGR_classes['Building']] = self.bin_classes.index('Building')
            cls_mask[y == self.BGR_classes['Vegetation']] = self.bin_classes.index('Vegetation')
            cls_mask[y == self.BGR_classes['Unlabeled']] = self.bin_classes.index('Unlabeled')
            cls_mask = cls_mask[:,:,0] 

            cls_mask = cv.resize(cls_mask, self.img_dim) 

            masks = torch.tensor(cls_mask, dtype=torch.int64)
        
        return x, masks
    
    def __len__(self):
        return len(self.data)
    
class AerialImagingDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # Input Data
        self.root = "assets/aerial_imaging/"

        sub_dir = []

        for file in os.scandir(self.root):
            if file.is_dir():
                sub_dir.append(file.name)

        self.images = []
        self.masks = []

        for dir in sub_dir:
            images_path = os.path.join(self.root, dir, "images/*.jpg")
            images_path = glob.glob(images_path)
            
            masks_path = os.path.join(self.root, dir, "masks/*.png")
            masks_path = glob.glob(masks_path)
            
            for img_path, mask_path in zip(images_path, masks_path):
                self.images.append(img_path)
                self.masks.append(mask_path)
        
    def datasetloader(self):
        # set aside 20% of train and test data for evaluation
        x_train, x_test, y_train, y_test = train_test_split(self.images, self.masks,
                                                            test_size = 0.1, shuffle = True, random_state = 8)

        # Use the same function above for the validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                        test_size = 18/90, random_state = 8)

        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = AerialDatasetGenerator(x_train, y_train, transform)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)

        validation_dataset = AerialDatasetGenerator(x_val, y_val, transform)
        validation_dataloader = DataLoader(validation_dataset, self.batch_size, shuffle=True)

        test_dataset = AerialDatasetGenerator(x_test, y_test, transform)
        test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle = False)
        
        return train_dataloader, validation_dataloader, test_dataloader
    