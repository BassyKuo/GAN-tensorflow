# 01-VanillaGAN

## Python Version

* 3.6.0


## Python Packages Requirements

* tensorflow >= 1.1.0
* numpy >= 1.12.1
* scikit-learn >= 0.18.1

Pip insatll:
```sh
$ pip install -r requirements.txt
```

## Usage

Example:

* MNIST
```sh
$ python3 01-VanillaGAN_mnist.py train --max_epoch 1000 --out_dir mnist_output/
```

* CIFAR10
```sh
$ python3 01-VanillaGAN_cifar10.py train --max_epoch 1000 --out_dir cifar10_output/
```

Using `-h` or `--help` to see more information:
```sh
$ python3 01-VanillaGAN_mnist.py -h
$ python3 01-VanillaGAN_cifar10.py -h
```

