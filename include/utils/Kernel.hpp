/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

/**
 * @file      Kernel.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define kernel object and common-specific ones.
 *
 * @details   These classes build kernel as a specific Matrix.
*/

#ifndef N2D2_KERNEL_H
#define N2D2_KERNEL_H

#include "containers/Matrix.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
/**
 * @class   Kernel
 * @brief   Interface shared by all specific Kernels.
 *
*/
template <class T> class Kernel : public Matrix<T> {
public:
    /**
     * Build a new Kernel from a string.
     *
     * @param kernel        string which contains kernel values
     * @param sizeX         vertical size of the kernel
     * @param sizeY         horizontal soze of the kernel
    */
    Kernel(const std::string& kernel,
           unsigned int sizeX = 0,
           unsigned int sizeY = 0);

    /**
     * Perform a normalization on the kernel data.
    */
    void zeroSummingNorm();

    template <class U> friend Kernel<U> operator-(const Kernel<U>& kernel);

protected:
    Kernel() {};
};

/**
 * @class   GaborKernel
 * @brief   Build a 2D normalized Gabor kernel.
 *
 * @param sizeX             Kernel width
 * @param sizeY             Kernel height
 * @param theta             Orientation of the normal to the parallel stripes of
 *a Gabor function
 * @param sigma             Sigma of the Gaussian envelope
 * @param lambda            Wavelength of the sinusoidal factor
 * @param psi               Phase offset
 * @param gamma             Spatial aspect ratio, specifies the ellipticity of
 *the support of the Gabor function
 * @param zeroSumming       Zero-summing normalization can be applied to the
 *kernel
*/
template <class T> class GaborKernel : public Kernel<T> {
public:
    GaborKernel(unsigned int sizeX,
                unsigned int sizeY,
                double theta,
                double sigma = std::sqrt(2.0),
                double lambda = 10.0,
                double psi = M_PI / 2.0,
                double gamma = 0.5,
                bool zeroSumming = true);
};

/**
 * @class   GaussianKernel
 * @brief   Build a 2D normalized Gaussian kernel.
 *
 * \f$w(x,y) =
 *\frac{1}{2\pi\sigma^2}exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)\f$
*/
template <class T> class GaussianKernel : public Kernel<T> {
public:
    GaussianKernel(unsigned int sizeX, unsigned int sizeY, double sigma = 1.4);
};

/**
 * @class   LaplacianOfGaussianKernel
 * @brief   Build a 2D Laplacian-of-Gaussian (LoG) kernel.
 *
 * \f$w(x,y) =
 *-\frac{1}{\pi\sigma^4}\left(1-\frac{x^2+y^2}{2\sigma^2}\right)exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)\f$
 * @n
 * Zero-summing normalization can be applied to the kernel.
*/
template <class T> class LaplacianOfGaussianKernel : public Kernel<T> {
public:
    LaplacianOfGaussianKernel(unsigned int sizeX,
                              unsigned int sizeY,
                              double sigma = 1.4,
                              bool zeroSumming = true);
};

/**
 * @class   DifferenceOfGaussianKernel
 * @brief   Build a 2D Difference-of-Gaussian (DoG) kernel.
 *
 * \f$w(x,y) =
 *\frac{1}{\sqrt{2\pi}}\left(\frac{1}{\sigma_1}exp\left(-\frac{x^2+y^2}{2{\sigma_{1}}^2}\right)
 * -
 *\frac{1}{\sigma_2}exp\left(-\frac{x^2+y^2}{2{\sigma_{2}}^2}\right)\right)\f$
 * @n
 * Zero-summing normalization can be applied to the kernel.
*/
template <class T> class DifferenceOfGaussianKernel : public Kernel<T> {
public:
    DifferenceOfGaussianKernel(unsigned int sizeX,
                               unsigned int sizeY,
                               double sigma1 = 2.0,
                               double sigma2 = 1.0,
                               bool zeroSumming = true);
};
}

template <class T>
N2D2::Kernel<T>::Kernel(const std::string& kernel,
                        unsigned int sizeX,
                        unsigned int sizeY)
{
    // ctor
    std::vector<T> kernelVec;
    std::stringstream kernelStr(kernel);
    double value;

    while (kernelStr >> value)
        kernelVec.push_back(value);

    if (sizeX == 0 && sizeY == 0) {
        // No size specified, the kernel is supposed to be a square
        sizeX = sizeY = std::sqrt((double)kernelVec.size());
    } else if (sizeX != 0 && sizeY == 0)
        sizeY = kernelVec.size() / sizeX;
    else if (sizeX == 0 && sizeY != 0)
        sizeX = kernelVec.size() / sizeY;

    if (kernelVec.size() != sizeX * sizeY)
        throw std::runtime_error("Kernel<T>::Kernel: Kernel size mismatch with "
                                 "the size of its specified values.");

    Matrix<T>::resize(sizeY, sizeX);
    std::copy(kernelVec.begin(), kernelVec.end(), Matrix<T>::mData.begin());
}

template <class T> void N2D2::Kernel<T>::zeroSummingNorm()
{
    double pos = 0.0, neg = 0.0;

    for (typename Matrix<T>::const_iterator it = Matrix<T>::mData.begin(),
                                            itEnd = Matrix<T>::mData.end();
         it != itEnd;
         ++it) {
        if ((*it) > 0.0)
            pos += (*it);
        else
            neg -= (*it);
    }

    pos = (pos > 0.0) ? pos : 1.0;
    neg = (neg > 0.0) ? neg : 1.0;

    for (typename Matrix<T>::iterator it = Matrix<T>::mData.begin(),
                                      itEnd = Matrix<T>::mData.end();
         it != itEnd;
         ++it)
        (*it) /= ((*it) > 0.0) ? pos : neg;
}

namespace N2D2 {
template <class T> Kernel<T> operator-(const Kernel<T>& kernel)
{
    Kernel<T> negKernel(kernel);
    std::transform(negKernel.mData.begin(),
                   negKernel.mData.end(),
                   negKernel.mData.begin(),
                   std::negate<T>());
    return negKernel;
}
}

template <class T>
N2D2::GaborKernel<T>::GaborKernel(unsigned int sizeX,
                                  unsigned int sizeY,
                                  double theta,
                                  double sigma,
                                  double lambda,
                                  double psi,
                                  double gamma,
                                  bool zeroSumming)
{
    const double x0 = (sizeX - 1.0) / 2.0;
    const double y0 = (sizeY - 1.0) / 2.0;
    const double sigmaX = sigma;
    const double sigmaY = sigma / gamma;

    Matrix<T>::resize(sizeY, sizeX);

    for (unsigned int x = 0; x < sizeX; ++x) {
        for (unsigned int y = 0; y < sizeY; ++y) {
            const double xTheta = (x - x0) * std::cos(theta)
                                  + (y - y0) * std::sin(theta);
            const double yTheta =
                -(x - x0) * std::sin(theta) + (y - y0) * std::cos(theta);

            (*this)(y, x)
                = 1.0 / (2.0 * M_PI * sigmaX * sigmaY)
                  * std::exp(-0.5 * (xTheta * xTheta / (sigmaX * sigmaX)
                                     + yTheta * yTheta / (sigmaY * sigmaY)))
                  * std::cos((2.0 * M_PI / lambda) * xTheta + psi);
        }
    }

    if (zeroSumming)
        Kernel<T>::zeroSummingNorm();
}

template <class T>
N2D2::GaussianKernel
    <T>::GaussianKernel(unsigned int sizeX, unsigned int sizeY, double sigma)
{
    const double x0 = (sizeX - 1.0) / 2.0;
    const double y0 = (sizeY - 1.0) / 2.0;
    const double sigma_2 = sigma * sigma;
    const double vnorm = 1.0 / (2.0 * M_PI * sigma_2);

    Matrix<T>::resize(sizeY, sizeX);

    for (unsigned int x = 0; x < sizeX; ++x) {
        for (unsigned int y = 0; y < sizeY; ++y) {
            const double r_2s = ((x - x0) * (x - x0) + (y - y0) * (y - y0))
                                / (2.0 * sigma_2);
            (*this)(y, x) = vnorm * std::exp(-r_2s);
        }
    }
}

template <class T>
N2D2::LaplacianOfGaussianKernel<T>::LaplacianOfGaussianKernel(
    unsigned int sizeX, unsigned int sizeY, double sigma, bool zeroSumming)
{
    const double x0 = (sizeX - 1.0) / 2.0;
    const double y0 = (sizeY - 1.0) / 2.0;
    const double sigma_2 = sigma * sigma;
    const double vnorm = -1.0 / (M_PI * sigma_2 * sigma_2);

    Matrix<T>::resize(sizeY, sizeX);

    for (unsigned int x = 0; x < sizeX; ++x) {
        for (unsigned int y = 0; y < sizeY; ++y) {
            const double r_2s = ((x - x0) * (x - x0) + (y - y0) * (y - y0))
                                / (2.0 * sigma_2);
            (*this)(y, x) = vnorm * (1.0 - r_2s) * std::exp(-r_2s);
        }
    }

    if (zeroSumming)
        Kernel<T>::zeroSummingNorm();
}

template <class T>
N2D2::DifferenceOfGaussianKernel
    <T>::DifferenceOfGaussianKernel(unsigned int sizeX,
                                    unsigned int sizeY,
                                    double sigma1,
                                    double sigma2,
                                    bool zeroSumming)
{
    const double x0 = (sizeX - 1.0) / 2.0;
    const double y0 = (sizeY - 1.0) / 2.0;
    const double vnorm = 1.0 / std::sqrt(2.0 * M_PI);

    Matrix<T>::resize(sizeY, sizeX);

    for (unsigned int x = 0; x < sizeX; ++x) {
        for (unsigned int y = 0; y < sizeY; ++y) {
            const double r_2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            (*this)(y, x)
                = vnorm
                  * ((1.0 / sigma1) * std::exp(-r_2 / (2.0 * sigma1 * sigma1))
                     - (1.0 / sigma2)
                       * std::exp(-r_2 / (2.0 * sigma2 * sigma2)));
        }
    }

    if (zeroSumming)
        Kernel<T>::zeroSummingNorm();
}

#endif // N2D2_KERNEL_H
