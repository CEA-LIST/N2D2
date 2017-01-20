/*
    (C) Copyright 2012 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_WINDOWFUNCTION_H
#define N2D2_WINDOWFUNCTION_H

#include <cmath>
#include <vector>

namespace N2D2 {
/**
 * Abstract base class to implement a window function, used for various signal
 * processing functions in Sound.
*/
template <class T> class WindowFunction {
public:
    virtual std::vector<T> operator()(unsigned int size) const = 0;
    virtual ~WindowFunction() {};
};

/**
 * Rectangular window, used for various signal processing functions in Sound.
 * \f$w(n) = 1\f$
*/
template <class T> class Rectangular : public WindowFunction<T> {
public:
    std::vector<T> operator()(unsigned int size) const;
};

/**
 * Hann window, used for various signal processing functions in Sound.
 * \f$w(n) = 0.5\; \left(1 - \cos \left ( \frac{2 \pi n}{N-1} \right) \right)\f$
*/
template <class T> class Hann : public WindowFunction<T> {
public:
    std::vector<T> operator()(unsigned int size) const;
};

/**
 * Hamming window, used for various signal processing functions in Sound.
 * \f$w(n) = \alpha - \beta\; \cos\left( \frac{2 \pi n}{N - 1} \right)\f$
 * with \f$\alpha = 0.54,\; \beta = 1 - \alpha = 0.46\f$
*/
template <class T> class Hamming : public WindowFunction<T> {
public:
    std::vector<T> operator()(unsigned int size) const;
};

/**
 * Cosine window, used for various signal processing functions in Sound.
 * \f$w(n) = \cos\left(\frac{\pi n}{N-1} - \frac{\pi}{2}\right) =
 * \sin\left(\frac{\pi n}{N-1}\right)\f$
*/
template <class T> class Cosine : public WindowFunction<T> {
public:
    std::vector<T> operator()(unsigned int size) const;
};

/**
 * Gaussian window, used for various signal processing functions in Sound.
 * \f$w(n)=e^{-\frac{1}{2} \left ( \frac{n-(N-1)/2}{\sigma (N-1)/2}
 * \right)^{2}}\f$
*/
template <class T> class Gaussian : public WindowFunction<T> {
public:
    Gaussian(T sigma = 0.4) : mSigma(sigma) {};
    std::vector<T> operator()(unsigned int size) const;

private:
    /// \f$\sigma\f$
    const T mSigma;
};

/**
 * Blackman window, used for various signal processing functions in Sound.
 * \f$w(n)=a_0 -  a_1 \cos \left ( \frac{2 \pi n}{N-1} \right) + a_2 \cos \left
 * ( \frac{4 \pi n}{N-1} \right)\f$ @n
 * \f$a_0=\frac{1-\alpha}{2};\quad a_1=\frac{1}{2};\quad a_2=\frac{\alpha}{2}\f$
*/
template <class T> class Blackman : public WindowFunction<T> {
public:
    Blackman(T alpha = 0.16) : mAlpha(alpha) {};
    std::vector<T> operator()(unsigned int size) const;

private:
    /// \f$\alpha\f$
    const T mAlpha;
};

/**
 * Kaiser window, used for various signal processing functions in Sound.
 * \f$w(n)=\frac{I_0\Bigg (\beta \sqrt{1 - ( \frac{2 n}{N-1} -1)^2}\Bigg )}
 * {I_0(\beta)}\f$ @n
 * where \f$I_0\f$ is the zero-th order modified Bessel function of the first
 * kind.
*/
template <class T> class Kaiser : public WindowFunction<T> {
public:
    Kaiser(T beta = 5.0) : mBeta(beta) {};
    std::vector<T> operator()(unsigned int size) const;

private:
    /**
     * Zeroth order modified Bessel Function of the First Kind.
    */
    T besselI0(T x) const;

    /// \f$\beta\f$
    const T mBeta;
};
}

template <class T>
std::vector<T> N2D2::Rectangular<T>::operator()(unsigned int size) const
{
    return std::vector<T>(size, 1.0);
}

template <class T>
std::vector<T> N2D2::Hann<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    for (unsigned int n = 0; n < size; ++n)
        w.push_back(0.5 * (1.0 - std::cos(2.0 * M_PI * n / (size - 1))));

    return w;
}

template <class T>
std::vector<T> N2D2::Hamming<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    for (unsigned int n = 0; n < size; ++n)
        w.push_back(0.54 - 0.46 * std::cos(2.0 * M_PI * n / (size - 1)));

    return w;
}

template <class T>
std::vector<T> N2D2::Cosine<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    for (unsigned int n = 0; n < size; ++n)
        w.push_back(std::sin(M_PI * n / (size - 1)));

    return w;
}

template <class T>
std::vector<T> N2D2::Gaussian<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    const T n_2 = (size - 1) / 2.0;

    for (unsigned int n = 0; n < size; ++n) {
        const T v = (n - n_2) / (mSigma * n_2);
        w.push_back(std::exp(-1.0 / 2.0 * v * v));
    }

    return w;
}

template <class T>
std::vector<T> N2D2::Blackman<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    for (unsigned int n = 0; n < size; ++n)
        w.push_back((1.0 - mAlpha) / 2.0
                    - 1.0 / 2.0 * std::cos(2.0 * M_PI * n / (size - 1))
                    + mAlpha / 2.0 * std::cos(4.0 * M_PI * n / (size - 1)));

    return w;
}

template <class T>
std::vector<T> N2D2::Kaiser<T>::operator()(unsigned int size) const
{
    std::vector<T> w;
    w.reserve(size);

    const T denom = besselI0(mBeta);

    for (unsigned int n = 0; n < size; ++n) {
        const T v = (2.0 * n / (size - 1) - 1.0);
        w.push_back(besselI0(mBeta * std::sqrt(1.0 - v * v)) / denom);
    }

    return w;
}

template <class T> T N2D2::Kaiser<T>::besselI0(T x) const
{
    const T numFactor = x * x / 4.0;

    T num = 1.0;
    T denom = 1.0;
    T sum = 0.0;

    for (unsigned int k = 0; k < 100; ++k) {
        const T v = num / (denom * denom);
        sum += v;

        if (v / sum < 1.0e-12)
            break;

        num *= numFactor;
        denom *= (k + 1);
    }

    return sum;
}

#endif // N2D2_WINDOWFUNCTION_H
