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
import torchvision.transforms as transforms
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

# first, we need to get the image path and the labels.
imagelist=[]
labels=[]

#directory where we have the images
folder_dir = "./images_demo/"

# os.listdir gets all the files in a directory and the loop goes through all
#the files.
for image in os.listdir(folder_dir):
    if image.endswith(".jpg"):                        #Check whether is an image
        imagelist.append(os.path.join(folder_dir,image))

        #Add a label based on what is in the image (based on the image name)
        if 'robot' in image:
            labels.append(0)

        elif 'car' in image:
            labels.append(1)

print(f'The image paths are: {imagelist}. \n The labels are: {labels} \n')


#transforms.Compose([transforms.ToTensor(),transforms.Resize((,))])
#transfrom.ToTensor takes an image matrix. transforms.Resize resizes the tensor.
transform= transforms.Compose([transforms.ToTensor(),transforms.Resize((100,100))])

#We will need to use transform from torchvisions to open the images with tensors.
#Then, apply implement the transform in the JointDataset class.

#JointDataset class to fetch images and labels in a batch.
class ImageDataset(Dataset):
    def __init__(self,imagelist,label_list,transforms):
        self.imagelist=imagelist
        self.labels=label_list
        self.transforms=transforms

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self,idx):
        #get the label
        label=self.labels[idx]
        #get the image into an array format
        image=Image.open(self.imagelist[idx])

        if self.transforms is not None:
            image=self.transforms(image)

        return image,label

#Build the class
imagedataset=ImageDataset(imagelist,labels,transform)
data_loader=DataLoader(imagedataset,batch_size=2, shuffle=True) #create the dataloader

#Check the tensors in the dataset
for i,batch in enumerate(data_loader):
    print (f'Bach {i}: Number of items in the batch {len(batch)}, labels {batch[1]}, shape tensor {batch[0][0].shape}')


############ 1.3 LOAD AND USE IMAGE DATASET FROM TORCHVISION ############
print('\n 1.3 LOAD AND USE AN IMAGE DATASET FROM TORCHVISION \n')

#We can 
