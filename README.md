# <img src="manual/figs/N2D2_Logo.png" alt="N2D2" height="100"/>

| **`Linux CPU`**<br/><sub>&ge; GCC 4.4.7</sub> | **`Linux GPU`**<br/><sub>&ge; CUDA 6.5 + CuDNN 1.0</sub> | **`Windows CPU`**<br/><sub>&ge; Visual Studio 2013.3</sub> | **`Windows GPU`**<br/><sub>&ge; CUDA 8.0 + CuDNN 5.1</sub>  | [![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)  |
| --------------- | ------------------ | ------------------ | ------------------ | ------ |
| [![Build Status](https://travis-ci.org/CEA-LIST/N2D2.svg?branch=master)](https://travis-ci.org/CEA-LIST/N2D2) | [![Build Status](https://travis-ci.org/CEA-LIST/N2D2.svg?branch=master)](https://travis-ci.org/CEA-LIST/N2D2) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/CEA-LIST/N2D2?branch=master&svg=true)](https://ci.appveyor.com/project/olivierbichler-cea/n2d2) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/CEA-LIST/N2D2?branch=master&svg=true)](https://ci.appveyor.com/project/olivierbichler-cea/n2d2) | [![Coverage Status](https://coveralls.io/repos/github/CEA-LIST/N2D2/badge.svg?branch=master)](https://coveralls.io/github/CEA-LIST/N2D2?branch=master) |

N2D2 (for 'Neural Network Design & Deployment') is a open source CAD framework for
Deep Neural Network (DNN) simulation and full DNN-based applications building.
It is developped by the [CEA LIST](http://www-list.cea.fr/) along with
industrial and academic partners and is open to community contributors.
The only mandatory dependencies for N2D2 are OpenCV (&ge; 2.0.0) and Gnuplot.
The NVIDIA CUDA and CuDNN libraries are required to enable GPU-acceleration.

If you did like to contribute to N2D2, please make sure to review the
[contribution guidelines](CONTRIBUTING.md).

Usage
-----

To compile and use N2D2, please refer to the
[manual](https://github.com/CEA-LIST/N2D2/raw/master/manual/manual.pdf), which
contains the following resources:
- General presentation of the framework;
- How to compile N2D2 and perform simulations;
- How to write neural network models;
- Tutorials.

The N2D2 executables and application examples are located in the [exec](exec)
directory.

Obtain N2D2 with Docker: `docker pull cealist/n2d2`

License
-------

N2D2 is released under the [CeCILL-C](LICENSE) license, a free software license
 adapted to both international and French legal matters that is fully compatible
 with the FSF's GNU/LGPL license.

