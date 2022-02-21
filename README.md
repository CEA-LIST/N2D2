<div align="center">
  <img src="docs/_static/N2D2_Logo.png" alt="N2D2" height="200">
</div>

| **Docs** | **`Linux CPU`**<br/><sub>&ge; GCC 4.4.7</sub> | **`Linux GPU`**<br/><sub>&ge; CUDA 6.5 + CuDNN 1.0</sub> | **`Windows CPU`**<br/><sub>&ge; Visual Studio 2015.2</sub> | **`Windows GPU`**<br/><sub>&ge; CUDA 8.0 + CuDNN 5.1</sub>  | [![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)  |
| ---------- | --------------- | ------------------ | ------------------ | ------------------ | ------ |
| [![Documentation Status](https://readthedocs.org/projects/n2d2/badge/?version=latest)](https://cea-list.github.io/N2D2-docs/) | [![linux-cpu](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-cpu.yml/badge.svg)](https://github.com/CEA-LIST/N2D2/actions/workflows/build_linux-cpu.yml) | | | |  |

N2D2 (for 'Neural Network Design & Deployment') is a open source CAD framework for
Deep Neural Network (DNN) simulation and full DNN-based applications building.
It is developped by the [CEA LIST](http://www-list.cea.fr/) along with
industrial and academic partners and is open to community contributors.
The only mandatory dependencies for N2D2 are OpenCV (&ge; 2.0.0) and Gnuplot.
The NVIDIA CUDA and CuDNN libraries are required to enable GPU-acceleration.


## Usage

To compile and use N2D2, please refer to the
[online documentation](https://cea-list.github.io/N2D2-docs/), which
contains the following resources:
- General presentation of the framework;
- How to compile N2D2 and perform simulations;
- How to write neural network models;
- Tutorials.

The N2D2 executables and application examples are located in the [exec](exec) directory.


## Docker Image

You can also pull a pre-built docker image from Docker Hub and run it with docker
```
docker pull cealist/n2d2
docker run --gpus all cealist/n2d2:latest
```

You can also build N2D2 from the `Dockerfile`. 
It is supplied to build images with CUDA 10.2 support and cuDNN v8.


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

