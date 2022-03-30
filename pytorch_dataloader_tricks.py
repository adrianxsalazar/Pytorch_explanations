import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pathlib
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torchvision.transfrom as transforms
torch.manual_seed(0)

############ 1.0 UNDERSTANDING THE DATALOADER FUNCTION ############
print('\n 1.0 UNDERSTANDING THE DATALOADER FUNCTION \n')
#Data we are going to use in the experiments. X attributes and y labels.
t_x=torch.rand(6,6)*6
t_y=torch.rand(6)*3
print(f'Attributes: {t_x}','\n',f' labels: {t_y}')


#DataLoader(): Standard data loader of the attribute tensor
data_loader=DataLoader(t_x)
print(f'\n Standard DataLoader \n of the attributes')
for i,batch in enumerate(data_loader):
    print(f'batch  {i}:', batch)

#DataLoader(batch_size=<number>, drop_last=True/False). We can choose the
#batch size and decide whether we drop the last uncomplete batch.
data_loader=DataLoader(t_x, batch_size=2, drop_last=False)
print(f'\n DataLoader of the attributes: batch = 2')
for i,batch in enumerate(data_loader):
    print(f'batch  {i}:', batch)

#Epochs: if we want to go throigh the dataset several times we need to write
#and additional loop.
#DataLoader(shuffle=True/false) is to suffle the batches
print(f'\n DataLoader of the attributes: batch = 3 and two epochs with suffle')
data_loader=DataLoader(t_x, batch_size=3, drop_last=False, shuffle=True)
epochs=2

for epoch in range(epochs):                     #epoch loop
    for i,batch in enumerate(data_loader):      #batch loop
        print(f'epoch: {epoch}, batch  {i}:', batch)

del t_x, t_y, data_loader, epoch

############ 1.1 MERGING TENSOR WITH DATALOADER ############
print('\n 1.1 MERGING TENSOR WITH DATALOADER \n')

#We might need to get two tensors at the same time with the dataloader. This is
#useful we want to fetch the attributes (X) and the labels (y) at the same time
#without loosing the right indexation.
#We have to create a class that inherits the dataset object from Pytorch
#Such class needs to include methods such as __init__, and __getitem__

#JointDataset: class that inherits the Dataset class. We use this to merge tnesors
class JointDataset(Dataset):
    #Constructor of the class
    def __init__(self,t_attributes,t_labels):
        self.x=t_attributes
        self.y=t_labels

    #__len__ indicates the dimension of the dataset to the loader.
    def __len__(self):
        return len(self.x)

    #__getitem__ indicates what is returned in each batch. In this case, two tensors
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

#Attributes and class
t_x=torch.normal(mean=3,std=5,size=(6,2))
t_y=torch.normal(mean=0,std=1,size=(6,1))
print(f'Attributes: {t_x}','\n',f' labels: {t_y} \n')

#Create the data loader using the class we used
joint_dataset=JointDataset(t_x,t_y)
data_loader=DataLoader(joint_dataset, batch_size=2,shuffle=True)

#The loop goes through the batches
for i,batch in enumerate(data_loader):      #batch loop
    print(f' batch  {i}: attributes: {batch[0]}, label {batch[1]} \n')


#More information a https://pytorch.org/docs/stable/data.html.

############ 1.2 CREATE A DATASET WITH IMAGES IN A FOLDER ############
print('\n 1.2 CREATE A DATASET WITH IMAGES IN A FOLDER \n')
#transforms.Compose([transforms.ToTensor(),transforms.Resize((,))])
#transfrom.ToTensor takes an image matrix. transforms.Resize resizes the tensor.
transform= transforms.Compose([transforms.ToTensor(),transforms.Resize((100,100))])

#We will need to use transform from torchvisions to open the images with tensors.
#Then, apply implement the transform in the JointDataset class.

class ImageDataset(dataset):
    def __init__(self,imagelist,label_list):
        self.imagelist()

    def __len__(self):

    def __getitem__(self,idx):

        return image,label



############ 1.3 LOAD AND USE IMAGE DATASET FROM TORCHVISION ############
print('\n 1.3 LOAD AND USE AN IMAGE DATASET FROM TORCHVISION \n')
