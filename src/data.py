import os
from pathlib import Path
import xml.etree.ElementTree as ET # for parsing XML
from PIL import Image # to read images
import glob
from tqdm import tqdm

import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils


def get_transforms(image_size):
    
    # this normalizes pixel values between [-1,1]
    # https://www.kaggle.com/jesucristo/gan-introduction565419
    # GANHACK #1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    random_transforms = [transforms.ColorJitter(), 
                         transforms.RandomRotation(degrees=1)]
    random_cropper = [torchvision.transforms.CenterCrop(image_size), torchvision.transforms.RandomCrop(image_size)]


    # First preprocessing of data
    transform1 = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(random_transforms, p=0.3),
        transforms.ToTensor(),
        normalize])

    return transform1, transform2 
    

class DogDataset(Dataset):
    def __init__(self, img_dir, args, transform1, transform2=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.imgs = []
        self.labels = []
        for img_name in tqdm(self.img_names):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            annotation_basename = os.path.splitext(os.path.basename(img_path))[0]
            annotation_dirname = next(dirname for dirname in os.listdir(args.root_annots) if dirname.startswith(annotation_basename.split('_')[0]))
            annotation_filename = os.path.join(args.root_annots, annotation_dirname, annotation_basename)
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            bboxes = []
            for o in objects:
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
                bbox = (xmin, ymin, xmin+w, ymin+w)
                img_crop = img.crop(bbox)
                
                self.imgs.append(self.transform1(img_crop))
                self.labels.append(annotation_dirname.split('-')[1].lower())
                
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        
        if self.transform2 is not None:
            img = self.transform2(img)
        
        return img, label

    def __len__(self):
        return len(self.imgs)