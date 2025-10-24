import torch

# torch.rand() is used to create tensor of any dimension
# shape attribute show how elements are arranged inside a tensor
# ndim attribute returns dimension of a tensor.


# Create a random tensor of 0 dim (Scalar)
random_tensor = torch.rand([])
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.item(), random_tensor.dtype)

# Create a random tensor of 1 dim (Vector) with 4 elements
random_tensor = torch.rand(4)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)

# Create a random tensor of 2 dim (MATRIX) with 2x4 
random_tensor = torch.rand(2,4)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)


# Create a random tensor of 3 dim (TENSOR) with 1X2x4 
random_tensor = torch.rand(1,2,4)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)


# torch.rand(1,2,4) is similar to torch.rand(size=(1,2,4))

# Create a random tensor of 3 dim (TENSOR) with 1X2x4 showing explicit usage paramter
random_tensor = torch.rand(size=(1,2,4))
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)

# Tensor with Zeros or Ones

# Create a tensor of Zeros having 3 dim (TENSOR) with 1X2x4 showing
random_tensor = torch.zeros(size=(1,3,4))
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)


# Create a tensor of Ones having 3 dim (TENSOR) with 1X2x4 showing
random_tensor = torch.ones(size=(1,3,4))
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)

# Create a 1-D tensor in  defined range
# pit fall - datatype will not be float.32 by default if int is used in start, end and step
#https://docs.pytorch.org/docs/stable/generated/torch.arange.html
random_tensor = torch.arange(start=2,end=5,step=2)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)


#Sometimes you might want one tensor of a certain type with the same shape as another tensor.
ones_tensor = torch.rand(size=(1,3,4))
random_tensor = torch.zeros_like(input=ones_tensor) # will have same shape
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)

random_tensor = torch.ones_like(input=ones_tensor) # will have same shape
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype)


# Datatypes : There are many different tensor datatypes available in PyTorch.
# Some are specific for CPU and some are better for GPU.
#https://docs.pytorch.org/docs/stable/tensors.html#data-types
#default is torch.float32
#An integer is a flat round number like 7 whereas a float has a decimal 7.0. The reason for all of these is to do with precision in computing.
#create tensors with specific datatypes. We can do so using the dtype parameter.
#if we pass int values , datatype will be int64, to force convert to float16 we need to mention it.
# torch gives priority to precision , so if one value is float it will convert all values as float
random_tensor = torch.tensor([3, 6, 9])
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)



# Once you have a tensor, you can get the most important values like shape, dtype, device from it.
# These three properties of a tensor are major pitfalls, so we need to look into them carefully.
random_tensor = torch.tensor([3, 6])
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)

# Tensors don't change unless reassigned . What this means, below is an example
#random_tensor will not have its value change after + oand * , instead it returns a new tensor.

tensor = torch.tensor([1, 2, 3])
print("-------------------------------------------------------------------------")
print(tensor, tensor.ndim, tensor.shape, tensor.dtype, tensor.device)
random_tensor = tensor + 10
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)

#Alternatively we can also use torch.multiply(tensor, 10) and tensor.add(tensor, 10) for multiply and add
tensor = torch.tensor([1, 2, 3])
print("-------------------------------------------------------------------------")
print(tensor, tensor.ndim, tensor.shape, tensor.dtype, tensor.device)
random_tensor = torch.multiply(tensor , 10)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)

#Element wise multiplication where we multiply two tensor
tensor = torch.tensor([[1, 2, 3],[2,6,8]])
print("-------------------------------------------------------------------------")
print(tensor, tensor.ndim, tensor.shape, tensor.dtype, tensor.device)
random_tensor = tensor * tensor
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)

#Element wise addition where we multiply two tensor
tensor = torch.tensor([[1, 2, 3],[2,6,8]])
print("-------------------------------------------------------------------------")
print(tensor, tensor.ndim, tensor.shape, tensor.dtype, tensor.device)
random_tensor = torch.tensor(tensor + tensor)
print("-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)

#MATRIX MULTIPLICATION is widely used in Machine learning. Matrix Multiplication rules applied
# torch.matmul() is the fastest way in torch to do matrix multiplication
# M1(r1xc1) and M2(r2xc2) = M3(r1xc2) provided c1 = r2 , r and c stands for row and column
#You can also use torch.mm() which is a short for torch.matmul()
#A matrix multiplication like this is also referred to as the dot product of two matrices.
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]])

print(tensor_A.shape, tensor_B.shape)  # Matrix multiplication rules are not satisfying.
#torch.matmul(tensor_A, tensor_B)
# one way to resolve the issue to Transponse the matrix to match the mat mul rule.
# We transponse a matrix by in pYTorch in 2 ways:
# 1. torch.transpose(input, dim0, dim1) - where input is the desired tensor to transpose and dim0 and dim1 are the dimensions to be swapped.
# 2. tensor.T - where tensor is the desired tensor to transpose.
random_tensor = torch.matmul(tensor_A, tensor_B.T)
print("T-------------------------------------------------------------------------")
print(random_tensor, random_tensor.ndim, random_tensor.shape, random_tensor.dtype, random_tensor.device)


# tensor  aggregation functions min, max, sum, mean etc
# tensor.argmax() and tensor.argmin() gives the index where max or min occured.
random_tensor = torch.tensor([[1, 1],
                         [1, 101],
                         [100, -1]],dtype=torch.float32)

print("Aggregation-------------------------------------------------------------------------")
print(random_tensor.min(), random_tensor.max(), random_tensor.mean(),random_tensor.sum(),random_tensor.argmax(),random_tensor.argmin())

# tensor.type(data_type) returns tensor with datatype as data_type.
random_tensor = torch.tensor([[1, 1],
                         [1, 101],
                         [100, -1]],dtype=torch.float32)
print("----------------------------------------------------------------------")
print(random_tensor.type(torch.float16), random_tensor.dtype)
print(random_tensor.dtype)

# https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/

# reshape a tensor using torch.reshape()
random_tensor = torch.tensor([[1, 1],
                         [1, 101],
                         [100, -1]],dtype=torch.float32)
print("----------------------------------------------------------------------")
random_tensor_mod = torch.reshape(random_tensor,(3,1,2))
print(random_tensor.shape, random_tensor_mod.shape, random_tensor_mod)

# Remove single dimensionality from tensor using torch.squeeze(). Unsqueeze does opposite.
x_original = torch.rand(size=(1, 1, 3))
print("----------------------------------------------------------------------")
print(x_original.shape) #size = 1,1,3
print(x_original.squeeze().shape) # size= 3


# Indexing : Getting data from tensor
x_original = torch.rand(size=(3, 5, 3))
print("----------------------------------------------------------------------")
print(x_original) 
print(x_original[1,0,1]) 


#The two main methods you'll want to use for NumPy to PyTorch (and back again) are:
#torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
#torch.Tensor.numpy() - PyTorch tensor -> NumPy array.



#Reproducibility (trying to take the random out of random)
# Create two random tensors
#torch.manual_seed(seed=42)
torch.random.manual_seed(seed=42)
random_tensor_A = torch.rand(3, 4)
torch.random.manual_seed(seed=42) # need manual seed for each rand() function
random_tensor_B = torch.rand(3, 4)
print("----------------------------------------------------------------------")
print(random_tensor_A == random_tensor_B)


# check for gpu
import torch
print("----------------------------------------------------------------------")
print(torch.cuda.is_available())
# device can be cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# You can count the number of GPUs PyTorch has access to using
print(torch.cuda.device_count())
# checking device which the tensor is on
random_tensor_A = torch.rand(3, 4)
print(random_tensor_A.device)
#Putting tensors (and models) on the GPU
print(random_tensor_A.to(device))

#What if we wanted to move the tensor back to CPU?
#For example, you'll want to do this if you want to interact with your tensors with NumPy 
# (NumPy does not leverage the GPU).