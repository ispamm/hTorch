# Quaternion Neural Networks

This repository contains code inspired by many papers on quaternion neural networks. Here you will
find the code for quaternion batch normalization among the classical layers.<br>
Differently from other repos, every weight parameter is its own, thus every single real or imaginary part is *not* considered a parameter. <br>
The code also contains a function to turn real-valued NN's into quaternion ones.

![image](https://github.com/giorgiozannini/QNN/blob/main/image.png?raw=true)

#TODO:
- fix bn
- qselu (?)
- speed up code

## References
<a id="1">[1]</a> 
Chase Gaudet, Anthony Maida (2018).
Deep Quaternion Networks. 
2018 International Joint Conference on Neural Networks (IJCNN)
[link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489651&tag=1)

<a id="2">[2]</a> 
Chiheb Trabelsi et al. (2017).
Deep Complex Networks. 
[link](https://arxiv.org/abs/1705.09792)

<a id="3">[3]</a> 
Titouan Parcollet, Mohamed Morchid, G. Linarès (2019).
A survey of quaternion neural networks.
Artificial Intelligence Review.
[link](https://link.springer.com/article/10.1007/s10462-019-09752-1)
