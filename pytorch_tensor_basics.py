import numpy as np
import torch

#Basic instructions for pytorch v.01 by Adrian Salazar Gomez.
#To create tensor, we could directly user the torch tools or transfor a numpy
#array into tensors


##### 1.0 CREATING TENSORS #####
print ('\n 1.0 CREATING TENSORS \n')
#Using pytorch tensor
t_x=torch.tensor([4,3,2,1])

#Using numpy tensor
z=np.array([8,7,5,4], dtype=np.int32)
t_z=torch.from_numpy(z)

#Get the shape of a tensor
shape_tensor_z=t_z.shape

#Like in numpy, we can create tensors with a specific shape.
#This one creates a rank 2 tensor where all the numbers are 1
t_o=torch.ones(3,3)
t_n=torch.zeros(3,3)

#torch.rand creates a tensor of a specific shpae filled with random numbers.
t_r=torch.rand(4,2)

#torch.normal allows you to create a tensor that sollows a noral distribution.
t_r=torch.normal(mean=8,std=4,size=(2,3))
del t_r

##### 1.1 MODIFYING TENSORS AND TENSOR OPERATIONS #####
print ('\n 1.1 MODIFYING TENSORS AND TENSOR OPERATIONS \n')
#transpose.
#torch.transpose(input,first dimension to transpose,second dimension to transpose)
t_a=torch.zeros(2,3)
t_a_t=torch.transpose(t_a,0,1)
del t_a_t

#Squeeze. Squeze can delete the dimensions with
t_a_o=torch.ones(1,2,1,3,4,1)

#torch.squeeze: We can just remove the
t_a_s=torch.squeeze(t_a_o)
print (t_a_s.shape[:])
del t_a_s
del t_a_o

#tensor.reshape: Reshape a tensor
t_a_r=t_a.reshape(6,1)
print(f'The shape of the tensor was {t_a.shape[0],t_a.shape[1]}',
    f'now is {t_a_r.shape[0],t_a_r.shape[1]} ')
del t_a_r


#####   1.2 MERGE TENSORS #####
print ('\n 1.2 MERGE TENSORS \n')
torch.manual_seed(2)
t_a=torch.normal(mean=4,std=4,size=(6,4))
t_b=torch.normal(mean=4,std=4,size=(7,4))

#torch.chunks is to split the tensor in multiple parts with the same size.
t_a_chunks=torch.chunk(t_a,4)

print(f'Original tensor: \n {t_a}', '\n')
for i,t_a_chuck in enumerate(t_a_chunks):
    print(f'chunck {i}:', t_a_chuck, '\n')

#torch.split splits the tensor based on the dimension we choose.
#Here we choose to splot the tensor into 3 row tensors
t_a_splits=torch.split(t_a,3)

print(f'Original tensor: \n {t_a}', '\n')
for i,t_a_split in enumerate(t_a_splits):
    print(f'split {i}:', t_a_split, '\n')

#torch.stack and torch.cat are to concatenate tensors.
t_c=torch.cat((t_a, t_b), 0)
print(f'Concatenated tensor: \n',t_c,'\n')

del t_a, t_b, t_c, t_a_splits, t_a_chunks

##### 1.3 MATHEMATICAL OPERATIONS WITH TENSORS. We can easily  sum, rest, multiply
# shape_tensors. #####
print ('\n 1.3 MATHEMATICAL OPERATIONS WITH TENSORS. \n')
t_a=torch.ones(2,3)
print ('Original tensor t_a \n',t_a, '\n')

#tensor* number: multiply all the elements of the tensor.
t_multi=t_a*6
print ('Tensor multiplication \n',t_multi, '\n')

#tensor + number: sum a number to all the elements in the tensor
t_sum=t_a+2
print ('Tensor sum \n',t_multi, '\n')

#torch.manual_seed is to fix the random processes.
torch.manual_seed(1)
t_r=torch.normal(mean=8,std=4,size=(2,3))
print ('Second tensor t_r \n',t_r, '\n')

#torch.mean(): Get the mean of one of the dimensions of the tensor
t_r_m=torch.mean(t_r, axis=0)
print('Second tensor mean axis=0 \n',t_r_m, '\n')

t_r_m=torch.mean(t_r, axis=1)
print('Second tensor mean axis=1 \n',t_r_m, '\n')

#torch.multiply and torch.matmul are to multiply tensor.
#torch.mulltiply is for for intem wise multiplications
t_axr=torch.multiply(t_a,t_r)
print('torch.multiply of t_a and t_r \n',t_axr, '\n')

#torch.matmul is a matrix multiplication
print(t_a.shape, torch.transpose(t_r,0,1).shape)
t_axr=torch.matmul(torch.transpose(t_a,0,1),t_r)
print('torch.mm of t_a and t_r \n',t_axr, '\n')
