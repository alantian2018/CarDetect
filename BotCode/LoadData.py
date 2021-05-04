from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from torch.utils.data import *
from os import listdir
import numpy as np
import os
import cv2


class getpaths (Dataset):

    def __init__(self,path):
        self.dataset = [path + '\\' + f for f in listdir(path)]
        self.img_paths = []
        self.numtoca=[""]*len(self.dataset)
        for j in range(len(self.dataset)):
            self.img_dir=self.dataset[j]
            for i in range (len(self.img_dir)):
                self.img_paths+= [[self.img_dir + '\\' + f,j] for f in listdir(self.img_dir)]
                self.numtoca[j]=self.img_dir.split("\\")[-1]

    def item(self):
        return self.img_paths
    def numtocar (self):

        return self.numtoca


class LoadData():
    def __init__(self,img_dir,label,transform=None):
        self.paths = img_dir
        self.label = label
        self.transform=transform

    def __len__(self):
        return (len(self.paths))
    def __getitem__(self):
        img = cv2.imread(self.paths)

        label = self.label
        if self.transform:
            img = self.transform(img)
#        img = img.numpy().transpose(1, 2, 0)
#        print (type(img))

#        cv2.imshow("image", img)
#        cv2.waitKey(0)
        return img, label

def toDataset(data,transform):
    t = []
    for i in range(len(data)):
        c = LoadData(data[i][0], data[i][1], transform)
        if (i % 100 == 0):
            print("Done with {}/{}".format(i, len(data)))
        # print (sys.getsizeof(c.__getitem__))
        t .append (c.__getitem__())
    return t