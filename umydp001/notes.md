# PyTorch Fundamentals:

- Machine learning can be used for anything from universe, provided we can turn that 'anything' into numbers and program it to find patterns.

- For complex problems, human can't find all the rules efficiently.

- Do we always need Machine Learning to Solve problems ?
    1. Not always, As said in the [Google's Machine Learning Rule # 1](https://developers.google.com/machine-learning/guides/rules-of-ml). In simple words, if you can find all the rules to your problem, which humans can for simple problems, Machine learning is not needed.
    2. Problems where we don't have enough data.

- Where to Use ML
    1. Problems with long lists of Rules, where traditional approach fails
    2. Everchanging Environment/Scenario 
    3. Discovering insights within large dataset


- Learning Material for this course    - 
    - [Deep Learning Github](https://github.com/mrdbourke/pytorch-deep-learning/)
    - [Learn](https://www.learnpytorch.io/)

- What are Tensors ?
    - tensor represent n dimensional number in pytorch. tensor holds number with similar datatype (dtype)

    - Type of tensors based on dimension : scalar (0 dimensional tensor) , vector (1 dimensional tensor), matrix(2 dimensional tensor), n-dimensional tensor.
    - <iframe width="560" height="315" src="https://www.youtube.com/watch?reload=9&v=f5liqUk0ZTw&t=33s" frameborder="0" allowfullscreen></iframe>
    - [![Video Title](https://www.youtube.com/watch?v=f5liqUk0ZTw&t=33s)](https://www.youtube.com/watch?v=f5liqUk0ZTw&t=33s)


- Creating tensors
    - torch.tensor(7) # scalar
    - torch.tensor([4,3,5]) # vector
    - torch.tensor([[1,2,3],[4,5,6]]) # matrix
    - torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]]) # n-dimensional tensor
    - torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]]) # n-dimensional tensor

    - Checking dimension and shape of a tensor
        - tensor.ndim
        - tensor.shape
    
    - Checking datatype of tensor
        - tensor.dtype , default is float32


- Random tensors
    - why we need random tensors ?
        - Random numbers are important because most of the neural network start with random tensor and then adjust the random values to better represent the data.

        - Start with random number -> look at data -> update random numbers -> look at data -> update random numbers -> look at data and so on.... 

    - torch.rand(), torch.rand(1) , torch.rand(1,2), torch(1,3,4) creates scalar, vector , matrix, n-dimensional tensor respectively. rand() function expects size parameter which is default.

    - tensor with zeros , torch.zeroes(size=(1,2))

    - tensor with ones , torch.ones(size=(1,2))
