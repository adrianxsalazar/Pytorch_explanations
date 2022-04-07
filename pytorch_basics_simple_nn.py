import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os


############ 1.0 VANILLA REGRESSION WITH PYTORCH ############
print('\n 1.0 VANILLA REGRESSION WITH PYTORCH \n')
torch.manual_seed(0)
t_x=torch.normal(mean=4,std=6,size=(10,1))
t_y=torch.rand(10,1)

tensor_dataset=TensorDataset(t_x,t_y)
data_loader=DataLoader(tensor_dataset,batch_size=1,shuffle=True)

#function with the predictive model. Matrix multiplication between att and weights
#plus the bias.
def model(attr,weights,bias):
    return attr @ weights + bias

#function with the calculation of the loss.
def loss_regression(input, target):
    return (input-target).pow(2).mean()

#Initialize key variables, such as learning rate and weights
learning_rate=0.0001
weights=torch.randn(1)
weights.requires_grad_()
bias=torch.zeros(1, requires_grad=True)

#Main epoch loop.
for epoch in range(400):
    #batch loop
    for i,batch in enumerate(data_loader):
        prediction=model(batch[0],weights,bias)             #prediction
        loss=loss_regression(prediction,batch[1])           #loop
        # <tensor>.backward() calculates and accumulates gradients
        # We need to zero_ .grad attributes or set them to None before calling it.
        loss.backward()

    #Apply updates to the weight and bias. We need torch.no_grad to avoid
    #updating the gradients calculated with the .backaward() function.
    with torch.no_grad():
        weights -= weights.grad * learning_rate     #update weights
        bias -= bias.grad * learning_rate           #update biases
        weights.grad.zero_()                        #re-start gradients in weights
        bias.grad.zero_()                           #re-start gradients in bias

    if epoch % 50 == 0:
        print (loss)

del t_x, t_y

############ 1.1 PYTORCH: TORCH.NN ############
print('\n 1.1 PYTORCH: TORCH.NN \n')
#Data
torch.manual_seed(0)
t_x=torch.normal(mean=4,std=6,size=(10,1))
t_y=torch.rand(10)

#Using the tools from toch.nn we can easily create networs

#Variables for the experiments.
learning_rate=0.0001
input_size=1
output_size=1
epochs=400

#nn.Linear(input_size,output_size,bias=False/True)
model=nn.Linear(input_size,output_size,bias=True)

#nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#The loss is (x=y)^2. Reduction indicates how the losses are accumulated.
loss_network=nn.MSELoss(reduction='mean')

#nn.troch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0,
# weight_decay=0, nesterov=False, *, maximize=False). The function is the loss
#we use and the parameters that accepts.
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

#Create the dataloader
tensor_dataset=TensorDataset(t_x,t_y)
data_loader=DataLoader(tensor_dataset,batch_size=1,shuffle=True)

#Epoch loop
print (f'training of network using nn.Linear, torch.optim.SGD, and nn.MSELoss')
for epoch in range(epochs):
    #batch loop
    for i, batch in enumerate(data_loader):
        prediction=model(batch[0])[:,0]             #prediction

        if epoch == 1 and i == 1:
            print (f'\n Visualisation of the shape of the prediction:\n{model(batch[0])}\n')

        loss=loss_network(prediction,batch[1])
        loss.backward()                     #Calculate gradients
        optimizer.step()                    #update the model parameters
        optimizer.zero_grad()               #re-start the acumulated gradients

    if epoch % 50 == 0 and epoch != 0:
        print (f'Epoch: {epoch}, loss: {loss.item():.4f}')

#We can access the numeric information of the weights and bias with:
#model.weight.item() and model.bias.item()


############ 1.2 PYTORCH: TORCH.NN: 2 LAYER NETWORK ############
print('\n 1.2 PYTORCH: TORCH.NN: 2 LAYER NETWORK \n')
torch.manual_seed(0)

#We are going to use the classic iris dataset to test our 2 layer neural network.
iris_data=load_iris()
x=iris_data['data']                         #attributes
y=iris_data['target']                       #target

#Split the dataset.
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.2, random_state=0)
xtrain=(xtrain-np.mean(xtrain))/np.std(xtrain)         #normalise the attributes

#ataset into tensors, create the combined tensor dataset, create data loader.
t_x=torch.from_numpy(xtrain).float()
t_y=torch.from_numpy(ytrain)
tensor_dataset=TensorDataset(t_x,t_y)
data_loader=DataLoader(tensor_dataset,1, shuffle=True)

#Here we create a class for the model that inherits the characteristics from
#the nn.Module class. The key classes are the contructor and the forward class.

#Create the class of the predictive model
class Model(nn.Module):
    def __init__(self,input_size,middle_size,output_size):
        super().__init__()  #super() gives access to methods of a parent class
        self.layer1=nn.Linear(input_size,middle_size)
        self.layer2=nn.Linear(middle_size,output_size)

    def forward(self,x):
        x=self.layer1(x)
        x=nn.Sigmoid()(x)
        x=self.layer2(x)
        x=nn.Softmax(dim=1)(x)
        return x


#create the model
model=Model(t_x.shape[1],18,len(set(y)))

#set up the optimiser and the loss
lossnn=nn.CrossEntropyLoss()                                #Loss
optimiser=torch.optim.Adam(model.parameters(), lr=0.001)    #Optimiser

#training loop.
for epoch in range(401):                                    #400 Epochs
    for attributes,label in data_loader:
        prediction=model(attributes)
        loss=lossnn(prediction,label)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    if epoch % 50 == 0:
        print (f'Epoch: {epoch}, Loss: {loss.item():.4f}')


############ 1.3 EVALUATE THE  2 LAYER NETWORK ############
print('\n 1.3 EVALUATE THE  2 LAYER NETWORK \n')
#Evaluate the model in the testing set.
xtest=(xtest-np.mean(xtest))/np.std(xtest)                  #Nomralisation data
t_xtest=torch.from_numpy(xtest).float()
t_ytest=torch.from_numpy(ytest)
prediction_test=model(t_xtest)

#Check the predictions that the same as the ground truth. It will return 1 if
#is the same and 0 otherwise. Then, we will average the ones to know the accuracy
correct_predictions=(torch.argmax(prediction_test,dim=1)==t_ytest).float()
accuracy=correct_predictions.mean()
print (f'The accuracy is: {accuracy:.4f}')



############ 1.4 SAVE AND LOAD THE MODEL ############
print('\n 1.4 SAVE AND LOAD THE MODEL \n')

#Save and load the model
#First we could create a directory to save the model.
directory='./saved_models_simple_nn'

if not os.path.exists(directory):           #Check that the directory exist
    os.makedirs(directory)                  #Create the directory

#Decide the model name in the directory and save the model
model_path=os.path.join(directory,'two_layer_nn_iris.pt')   #Model path
#torch.save(model, path to save the model)
torch.save(model,model_path)                                #Save model

#torch.load(): Load the model
loaded_model=torch.load(model_path)

#check the model characteristics
print (loaded_model.eval())
