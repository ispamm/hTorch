# Quaternion PyTorch

This repository contains code to extend PyTorch for use in quaternion-valued applications. We provide quaternion-valued tensors, layers, and examples. Code is designed to be as inter-operable as possible with basic real-valued PyTorch tensors and operations.

This code draws in large part from Titouan Parcollet's [code](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks), which inspired this library.

## Installation

After cloning the repository, simply run:

```
python setup.py install 
```

## Using the library

See the [Basic notebook](notebooks/basic.ipynb) for an introduction to the basic concepts in the library.

## References

<a id="1">[1]</a> Chase Gaudet, Anthony Maida (2018). [Deep Quaternion Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489651&tag=1). 2018 International Joint Conference on Neural Networks (IJCNN).

<a id="2">[2]</a> Chiheb Trabelsi et al. (2017). [Deep Complex Networks](https://arxiv.org/abs/1705.09792). 

<a id="3">[3]</a> Titouan Parcollet, Mohamed Morchid, G. Linarès (2019). [A survey of quaternion neural networks](https://link.springer.com/article/10.1007/s10462-019-09752-1). Artificial Intelligence Review.