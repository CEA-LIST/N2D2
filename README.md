<div align="center">
  <img src="docs/_static/N2D2_Logo.png" alt="N2D2" height="200">
</div>

----------------------------------------------------------------

[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

N2D2 (for *Neural Network Design & Deployment*) is [CEA LIST](http://www-list.cea.fr/)'s CAD framework for designing and simulating
Deep Neural Network (DNN), and building full DNN-based applications on embedded platforms.
N2D2 is developped along with industrial and academic partners and is open source.


| **Docs** | **`Linux CPU`**<br/><sub>&ge; GCC 4.4.7</sub> | **`Linux GPU`**<br/><sub>CUDA 11.3 + CuDNN 8</sub> | **`Windows CPU`**<br/><sub>&ge; Visual Studio 2015.2</sub> | **`Windows GPU`**<br/><sub>&ge; CUDA 8.0 + CuDNN 5.1</sub> |
| ---------- | --------------- | ------------------ | ------------------ | ------------------ |
| [![Documentation Status](https://readthedocs.org/projects/n2d2/badge/?version=latest)](https://cea-list.github.io/N2D2-docs/) | [![linux-cpu](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-cpu.yml/badge.svg)](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-cpu.yml) | [![linux-gpu](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-gpu.yml/badge.svg)](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-gpu.yml) | | |


## Getting the sources

Use the following command:

> git clone --recursive git@github.com:CEA-LIST/N2D2.git

It is important to specify the recursive option as it will download the PyBind submodule.

## Usage

To compile and use N2D2, please refer to the
[online documentation](https://cea-list.github.io/N2D2-docs/), which
contains the following resources:
- [General presentation of the framework](https://cea-list.github.io/N2D2-docs/intro/intro.html)
- [How to compile N2D2 and perform simulations](https://cea-list.github.io/N2D2-docs/intro/simus.html)
- How to write neural network models
  - [With the INI interface](https://cea-list.github.io/N2D2-docs/ini/intro.html)
  - [With the Python API](https://cea-list.github.io/N2D2-docs/python_api/intro.html)
- How to quantize neural network models
  - [With Post-Training Quantization](https://cea-list.github.io/N2D2-docs/quant/post.html)
  - [With Quantization-Aware Training](https://cea-list.github.io/N2D2-docs/quant/qat.html)
- [How to export neural network models](https://cea-list.github.io/N2D2-docs/export/CPP.html)
- Tutorials

The N2D2 executables and **application examples** are located in the [exec](exec) directory.


## Install Dependencies

The only mandatory dependencies for N2D2 are OpenCV and Gnuplot.

Moreover, the NVIDIA CUDA and CuDNN libraries are required to enable GPU-acceleration.
We highly recommend to use a **CUDA version higher than 10** with a **CuDNN version higher than 7**


## Docker Image

You can also pull a **pre-built docker image** from Docker Hub and run it with docker
```
docker pull cealist/n2d2
docker run --gpus all cealist/n2d2:latest
```

Another possibility is to build N2D2 from the `Dockerfile`. 
It is supplied to build images with CUDA 10.2 support and CuDNN 8.


## Contributing

If you would like to contribute to the N2D2 project, we’re happy to have your help! 
Everyone is welcome to contribute code via pull requests, to file issues on GitHub, 
to help people asking for help, fix bugs that people have filed, 
to add to our documentation, or to help out in any other way.

We grant commit access (which includes full rights to the issue database, such as being able to edit labels) 
to people who have gained our trust and demonstrated a commitment to N2D2. <br>
For more details see our [contribution guidelines](CONTRIBUTING.md).


## License

N2D2 is released under the [CeCILL-C](LICENSE) license, 
a free software license adapted to both international and French legal matters 
that is fully compatible with the FSF's GNU/LGPL license.

