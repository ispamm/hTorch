# Quaternion PyTorch
<p align="center">
  <img src="https://github.com/giorgiozannini/hTorch/blob/main/image.png" width = 300>
</p>

![](coverage.svg)

> :warning: **this is still heavily experimental**. Use at your own discretion!

This repository contains code to extend PyTorch for use in quaternion-valued applications. We provide quaternion-valued tensors, layers, and examples. Code is designed to be as inter-operable as possible with basic real-valued PyTorch tensors and operations.

This code draws in large part from Titouan Parcollet's [code](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks), which inspired this library.

## Installation

After cloning the repository, install the requirements as:

```
pip install -r requirements.txt
```

If you want to run the unit tests, you will also need the [pyquaternion](http://kieranwynn.github.io/pyquaternion/) library. Then, install the library by running:

```
pip install -e .
```

## Using the library

The basic unit of the library is the `QuaternionTensor`, an extension of the PyTorch's `tensor` class to handle quaternion-valued elements. You can initialize a quaternion tensor by specifying the four components, or by providing a (..., 4)-dimensional tensor:

```python
# A vector with two quaternions
x = quaternion.QuaternionTensor(torch.rand(2, 4, requires_grad=True))
```

We provide a number of operations from quaternion algebra and inter-operability with PyTorch:

```python
x = x * torch.rand(2) # Multiply with real-valued scalars
x.norm.sum().backward() # Take the absolute value, sum, and take the gradient
```

We also provide layers and utilities to work with PyTorch modules, e.g.:

```python
model = torch.nn.Sequential(
    layers.QLinear(10, 20),
    torch.nn.ReLU(),
    layers.QLinear(20, 10),
    layers.QuaternionToReal(10), # Take the absolute value in output
)
```

 See the [Basic notebook](notebooks/basic.ipynb) for an introduction to the basic concepts in the library, and the [Training notebook](notebooks/training.ipynb) for an example of training a quaternion-valued CNN.


## Code organization

1. The `QuaternionTensor` class is defined in [htorch/quaternion.py].
2. Layers for building quaternion-valued NN are found in [htorch/layers.py].
3. A few utilities to load real-valued datasets or convert existing real-valued models can be found in [htorch/utils.py].

Most operations are documented in the [example notebooks](./notebooks/).

## Testing

To manually run the unit tests:

```
python -m unittest discover -s ./tests -p *_test.py
```

If you have [coverage](https://coverage.readthedocs.io/en/latest/) installed:

```
coverage run -m unittest discover -s ./tests -p *_test.py
```

To generate again the coverage badge (not automated yet), install [coverage-badge](https://pypi.org/project/coverage-badge/), then run:

```
coverage-badge -o coverage.svg -f
```

## References

<a id="1">[1]</a> Chase Gaudet, Anthony Maida (2018). [Deep Quaternion Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489651&tag=1). 2018 International Joint Conference on Neural Networks (IJCNN).

<a id="2">[2]</a> Chiheb Trabelsi et al. (2017). [Deep Complex Networks](https://arxiv.org/abs/1705.09792). 

<a id="3">[3]</a> Titouan Parcollet, Mohamed Morchid, G. Linarès (2019). [A survey of quaternion neural networks](https://link.springer.com/article/10.1007/s10462-019-09752-1). Artificial Intelligence Review.
