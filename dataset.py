import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch
import random
import cv2


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(image).type(torch.LongTensor)


# Dataset class for segmentation
class BreastCancerSegmentation(data.Dataset):
    def __init__(self, masks, img_width, img_height, is_augment=True):
        self.masks = masks
        self.img_width = img_width
        self.img_height = img_height
        self.is_augment = is_augment

    def __getitem__(self, index):
        images = self.masks[index].replace("labels", "images")
        image = Image.open(images).convert("RGB").resize((self.img_width, self.img_height))
        label = Image.open(self.masks[index]).resize((self.img_width, self.img_height))
        label = np.array(label, dtype=np.float32)
        image = np.array(image)
        img_new = np.float32(image)
        img_new = img_new /127.5 -1

        if self.is_augment == True:
            flipCode = random.choice([-1, 0, 1, 2, 3])
            if flipCode == 2:
                height, width = self.img_height, self.img_width
                center = (width / 2, height / 2)
                degree = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                M = cv2.getRotationMatrix2D(center, degree, 1.0)
                img_new = cv2.warpAffine(img_new, M, (height, width))
                label = cv2.warpAffine(label, M, (height, width))
            elif flipCode != 3:
                img_new = cv2.flip(img_new, flipCode)
                label = cv2.flip(label, flipCode)

        tfms = transforms.Compose([
            transforms.ToTensor()
        ])

        y_transform = transforms.Compose([
            ToLabel(),
        ])
        img_new = tfms(img_new)
        label = y_transform(label)

        return img_new, label

    def __len__(self):
        return len(self.masks)


# Dataset class for segmentation
class BUSISegmentation(data.Dataset):
    def __init__(self, images, masks, img_width, img_height, is_augment=True):
        self.images = images
        self.masks = masks
        self.img_width = img_width
        self.img_height = img_height
        self.is_augment = is_augment

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB").resize((self.img_width, self.img_height))
        label = Image.open(self.masks[index]).resize((self.img_width, self.img_height))
        label = np.array(label, dtype=np.float32)
        image = np.array(image)
        img_new = np.float32(image)
        img_new = img_new /127.5 -1

        if self.is_augment == True:
            flipCode = random.choice([-1, 0, 1, 2, 3])
            if flipCode == 2:
                height, width = self.img_height, self.img_width
                center = (width / 2, height / 2)
                degree = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                M = cv2.getRotationMatrix2D(center, degree, 1.0)
                img_new = cv2.warpAffine(img_new, M, (height, width))
                label = cv2.warpAffine(label, M, (height, width))
            elif flipCode != 3:
                img_new = cv2.flip(img_new, flipCode)
                label = cv2.flip(label, flipCode)

        tfms = transforms.Compose([
            transforms.ToTensor()
        ])

        y_transform = transforms.Compose([
            ToLabel(),
        ])
        img_new = tfms(img_new)
        label = y_transform(label)

        return img_new, label

    def __len__(self):
        return len(self.images)
