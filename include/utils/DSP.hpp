/*
    (C) Copyright 2011 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_DSP_H
#define N2D2_DSP_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/WindowFunction.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace N2D2 {
namespace DSP {
    namespace internal {
        inline int bitReverse(int n, int bits);
        template <typename T, bool INV>
        void fft_(std::vector<std::complex<T> >& x);
    }

    template <typename T>
    std::vector<std::complex<T> > toComplex(const std::vector<T>& x);
    template <typename T>
    std::vector<T> real(const std::vector<std::complex<T> >& x);
    template <typename T>
    std::vector<T> imag(const std::vector<std::complex<T> >& x);

    /**
     * Implementation of the radix-2 DIT case form of the Cooley-Tukey FFT
     * algorithm.
     * If necessary, x is zero-padded so that its size is a power of two.
    */
    template <typename T> void fft(std::vector<std::complex<T> >& x)
    {
        internal::fft_<T, false>(x);
    }
    template <typename T> void ifft(std::vector<std::complex<T> >& x)
    {
        internal::fft_<T, true>(x);
    }
    template <typename T> void hilbert(std::vector<std::complex<T> >& x);

    /**
     * Short-time Fourier transform (STFT).
     *
     * @param   x           Input signal.
     * @param   nFft        FFT size used for each frame. If 0, nFft = min(256,
     *size(x)).
     *                      If not a power of 2, the next power of 2 is used.
     * @param   wFunction   Window function to use.
     * @param   wSize       Size of the window function applied to each frame.
     *wSize cannot exceed nFft.
     *                      If 0, wSize = nFft and nOverlap = wSize/2.
     * @param   nOverlap    If >= 0, nOverlap is the number of samples the
     *frames overlap.
     *                      If < 0, nOverlap is the "hop size", i.e., the number
     *of samples to advance successive windows.
     *                      The overlap is the window length minus the hop size.
     *nOverlap must be less than wSize.
    */
    template <typename T>
    std::vector<std::vector<std::complex<T> > > stft(const std::vector<T>& x,
                                                     unsigned int nFft = 0,
                                                     const WindowFunction
                                                     <T>& wFunction = Hann<T>(),
                                                     unsigned int wSize = 0,
                                                     int nOverlap = 0);
    template <typename T>
    std::vector<std::vector<T> > spectrogram(const std::vector<T>& x,
                                             unsigned int nFft = 0,
                                             const WindowFunction<T>& wFunction
                                             = Hann<T>(),
                                             unsigned int wSize = 0,
                                             int nOverlap = 0,
                                             bool logScale = true);
}
}

int N2D2::DSP::internal::bitReverse(int n, int bits)
{
    int reversedN = n;
    int count = bits - 1;

    n >>= 1;
    while (n > 0) {
        reversedN = (reversedN << 1) | (n & 1);
        count--;
        n >>= 1;
    }

    return ((reversedN << count) & ((1 << bits) - 1));
}

template <typename T, bool INV>
void N2D2::DSP::internal::fft_(std::vector<std::complex<T> >& x)
{
    unsigned int size = x.size();

    if ((size & (size - 1))
        != 0) { // Standard bit hack to check if size is a power of 2
        // If not, perform zero-padding.
        size = 1 << ((int)std::ceil(std::log((double)size) / std::log(2.0)));
        x.resize(size, 0.0);
    }

    const int bits = (int)(std::log((double)size) / std::log(2.0));
    for (unsigned int j = 1; j < size / 2; j++) {
        const int swapPos = internal::bitReverse(j, bits);
        std::swap(x[j], x[swapPos]);
    }

    for (unsigned int N = 2; N <= size; N <<= 1) {
        for (unsigned int i = 0; i < size; i += N) {
            for (unsigned int k = 0; k < N / 2; k++) {
                const int evenIndex = i + k;
                const int oddIndex = i + k + (N / 2);
                const std::complex<T> even = x[evenIndex];
                const std::complex<T> odd = x[oddIndex];

                const double term = (-2.0 * M_PI * k) / (double)N;
                const std::complex<T> twiddle
                    = std::polar(1.0, (INV) ? -term : term) * odd;

                x[evenIndex] = even + twiddle;
                x[oddIndex] = even - twiddle;
            }
        }
    }

    if (INV) {
        std::transform(x.begin(),
                       x.end(),
                       x.begin(),
                       std::bind(std::divides<std::complex<T> >(),
                                 std::placeholders::_1,
                                 (double)size));
    }
}

template <typename T>
std::vector<std::complex<T> > N2D2::DSP::toComplex(const std::vector<T>& x)
{
    std::vector<std::complex<T> > y;
    y.reserve(x.size());
    std::copy(x.begin(), x.end(), std::back_inserter(y));
    return y;
}

template <typename T>
std::vector<T> N2D2::DSP::real(const std::vector<std::complex<T> >& x)
{
    std::vector<T> real;
    real.reserve(x.size());
    std::transform(x.begin(),
                   x.end(),
                   std::back_inserter(real),
                   std::bind(static_cast<T (std::complex<T>::*)() const>(
                                 &std::complex<T>::real),
                             std::placeholders::_1));
    return real;
}

template <typename T>
std::vector<T> N2D2::DSP::imag(const std::vector<std::complex<T> >& x)
{
    std::vector<T> imag;
    imag.reserve(x.size());
    std::transform(x.begin(),
                   x.end(),
                   std::back_inserter(imag),
                   std::bind(static_cast<T (std::complex<T>::*)() const>(
                                 &std::complex<T>::imag),
                             std::placeholders::_1));
    return imag;
}

template <typename T> void N2D2::DSP::hilbert(std::vector<std::complex<T> >& x)
{
    fft(x);

    for (unsigned int i = 1; i < x.size() / 2; ++i)
        x[i] *= 2.0;

    for (unsigned int i = x.size() / 2 + 1; i < x.size(); ++i)
        x[i] = 0.0;

    ifft(x);
}

template <typename T>
std::vector<std::vector<std::complex<T> > >
N2D2::DSP::stft(const std::vector<T>& x,
                unsigned int nFft,
                const WindowFunction<T>& wFunction,
                unsigned int wSize,
                int nOverlap)
{
    // Default values
    if (nFft == 0)
        nFft = std::min((unsigned int)256, (unsigned int)x.size());

    // Ensure that nFft is a power of 2, because the size of y has to match the
    // vector size returned by fft()
    if ((nFft & (nFft - 1)) != 0)
        nFft = 1 << ((int)std::ceil(std::log((double)nFft) / std::log(2.0)));

    if (wSize == 0) {
        wSize = nFft;
        nOverlap = wSize / 2;
    }

    // Range check
    if (wSize > nFft)
        throw std::runtime_error(
            "The size of the window function (wSize) cannot exceed nFft.");

    if (nOverlap >= 0 && ((unsigned int)nOverlap) >= wSize)
        throw std::runtime_error("The overlap (nOverlap) must be less than the "
                                 "size of the window function (wSize).");

    unsigned int nHop;

    if (nOverlap < 0) {
        nHop = -nOverlap;
        nOverlap = wSize - nHop;
    } else
        nHop = wSize - nOverlap;

    // Window
    const std::vector<T> w = wFunction(wSize);
    const unsigned int n = x.size();
    const int nFrames = 1 + (int)std::floor((n - nOverlap) / (double)nHop);

    // Output pre-allocation
    std::vector<std::vector<std::complex<T> > > y(
        nFft, std::vector<std::complex<T> >(nFrames, 0.0));

    std::vector<T> xt;
    std::vector<std::complex<T> > yt;
    xt.reserve(nFft);
    yt.reserve(nFft);

#pragma omp parallel for private(xt, yt) if (nFrames > 4)
    for (int t = 0; t < nFrames; ++t) {
        const int offset = -((int)wSize / 2) + t * nHop;

        // Extract frame of input data
        if (offset < 0) {
            xt.assign(x.begin(), x.begin() + offset + wSize);
            xt.resize(wSize, 0.0);
        } else if (offset + wSize > n) {
            xt.assign(x.begin() + offset, x.end());
            xt.resize(wSize, 0.0);
        } else
            xt.assign(x.begin() + offset, x.begin() + offset + wSize);

        // Apply window to current frame
        std::transform(
            xt.begin(), xt.end(), w.begin(), xt.begin(), std::multiplies<T>());

        // Zero padding
        std::rotate(xt.begin(), xt.begin() + wSize / 2, xt.end());
        xt.insert(xt.begin() + wSize / 2, nFft - wSize, 0.0);

        yt = toComplex(xt);
        fft(yt);

#pragma omp critical
        {
            for (unsigned int f = 0; f < nFft; ++f)
                y[f][t] = yt[f];
        }
    }

    return y;
}

template <typename T>
std::vector<std::vector<T> > N2D2::DSP::spectrogram(const std::vector<T>& x,
                                                    unsigned int nFft,
                                                    const WindowFunction
                                                    <T>& wFunction,
                                                    unsigned int wSize,
                                                    int nOverlap,
                                                    bool logScale)
{
    const std::vector<std::vector<std::complex<T> > > y
        = stft(x, nFft, wFunction, wSize, nOverlap);
    const unsigned int size = y.size() / 2;

    std::vector<std::vector<T> > yMag(size, std::vector<T>());

    for (unsigned int i = 0; i < size; ++i) {
        const std::vector<std::complex<T> >& yi = y[i];
        std::vector<T>& yiMag = yMag[i];

        std::transform(
            yi.begin(), yi.end(), std::back_inserter(yiMag), std::abs<T>);

        if (logScale) {
            // yMag = 20*log10(|y|)
            std::transform(
                yiMag.begin(),
                yiMag.end(),
                yiMag.begin(),
                std::bind(std::multiplies<T>(),
                          20.0,
                          std::bind(static_cast<T (*)(T)>(&std::log10),
                                    std::placeholders::_1)));
        } else
            // yMag = |y|^2
            std::transform(yiMag.begin(),
                           yiMag.end(),
                           yiMag.begin(),
                           yiMag.begin(),
                           std::multiplies<T>());
    }

    return yMag;
}

#endif // N2D2_DSP_H
