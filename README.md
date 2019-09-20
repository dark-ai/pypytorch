![PyPyTorch-Logo](./assets/imgs/pypytorch-logo.png)

---

**PyPyTorch is a simple deep learning framework implemented by Python3**

# Contents

* [Introduction](#introduction)
* [Installation](#installation)
  * [PyPI](#pypi)
  * [From Source](#from-source)
* [TODO](#todo)


## Introduction

**PyPyTorch** is very similar to PyTorch, most of APIs are the same as PyTorch, at this stage, this framework enables us to understand how dynamic neural network works. If you master PyTorch, you can master PyPyTorch in a short time. There are some important and core modules in PyPyTorch. By the way, you can call its nickname **PPT** for short

| Module    | Description                                                  |
| --------- | ------------------------------------------------------------ |
| pypytorch | The entry of PyPyTorch framework, once you import pypytorch, you can work everything with PyPyTorch. |
| data      | Privides dataset, dataloader and transforms.                 |
| functions | A operator library for Tensor object in PyPyTorch.           |
| nn        | A high level neural network library in PyPyTorch which built on functions module, you can build a neural network model very fast with the help of nn. |
| optim     | There are some optimizers, including SGD, Adam and so on.    |

![pypytorch-run-mnist](./assets/gifs/pypytorch-run-mnist.gif)

Download the mnist.zip dataset at link [BaiduNetDisk](https://pan.baidu.com/s/1bfUBrjVqkHuunh4IsYW9EQ), then unzip it to examples/data/, if examples/data/ doesn't exist, you need create directory example/data/ by yourself. The full path of mnist dataset is `examples/data/mnist/` Besides, you can change configuration at examples/scripts/mnist/config.py

## Installation

### PyPI

1. `pip3 install pypytorch` I will add pypytorch to PyPI in the coming future

### From Source

1. `git clone https://github.com/dark-ai/pypytorch.git pypython`
2. `cd pypytorch`
3. `pip3 install -r requirements.txt`
4. `make install && make clean`

### Docker




## TODO

+ [x] SGD Optimizer
+ [x] Adam Optimizer
+ [x] Sequential
+ [x] ReLU
+ [x] BatchNorm 
+ [ ] DeConv2d(Waiting to test)
+ [ ] Upsample

