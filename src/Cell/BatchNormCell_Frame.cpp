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

#include "Cell/BatchNormCell_Frame.hpp"

template <>
N2D2::Registrar<N2D2::BatchNormCell>
N2D2::BatchNormCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::BatchNormCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::BatchNormCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::BatchNormCell>
N2D2::BatchNormCell_Frame<float>::mRegistrar("Frame",
    N2D2::BatchNormCell_Frame<float>::create,
    N2D2::Registrar<N2D2::BatchNormCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::BatchNormCell>
N2D2::BatchNormCell_Frame<double>::mRegistrar("Frame",
    N2D2::BatchNormCell_Frame<double>::create,
    N2D2::Registrar<N2D2::BatchNormCell>::Type<double>());

template <class T>
N2D2::BatchNormCell_Frame<T>::BatchNormCell_Frame(
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Activation>& activation)
    : Cell(name, nbOutputs),
      BatchNormCell(name, nbOutputs),
      Cell_Frame<T>(name, nbOutputs, activation),
      mScale(std::make_shared<Tensor<T> >()),
      mBias(std::make_shared<Tensor<T> >()),
      mMean(std::make_shared<Tensor<T> >()),
      mVariance(std::make_shared<Tensor<T> >())
{
    // ctor
    mScaleSolver = std::make_shared<SGDSolver_Frame<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::initialize()
{
    mInputs.synchronizeDToH();

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("BatchNormCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    mNbPropagate = 0;

    if (mEpsilon == 0.0)
        mEpsilon = 1.0e-5; // Same as CUDNN_BN_MIN_EPSILON

    std::vector<size_t> requiredDims(mInputs[0].nbDims(), 1);
    requiredDims[mInputs[0].nbDims() - 2] = mInputs.dimZ();

    if (mScale->empty())
        mScale->resize(requiredDims, T(1.0));
    else {
        if (mScale->dims() != requiredDims) {
            std::stringstream msgStr;
            msgStr << "BatchNormCell_Frame<T>::initialize():"
                " in cell " + mName + ", wrong size for shared scale, expected"
                " size is " << requiredDims << " whereas actual size is "
                << mScale->dims() << std::endl;

            throw std::runtime_error(msgStr.str());
        }
    }

    if (mBias->empty())
        mBias->resize(requiredDims, T(0.0));
    else {
        if (mBias->dims() != requiredDims) {
            std::stringstream msgStr;
            msgStr << "BatchNormCell_Frame<T>::initialize():"
                " in cell " + mName + ", wrong size for shared bias, expected"
                " size is " << requiredDims << " whereas actual size is "
                << mBias->dims() << std::endl;

            throw std::runtime_error(msgStr.str());
        }
    }

    if (mMean->empty())
        mMean->resize(requiredDims, T(0.0));
    else {
        if (mMean->dims() != requiredDims) {
            std::stringstream msgStr;
            msgStr << "BatchNormCell_Frame<T>::initialize():"
                " in cell " + mName + ", wrong size for shared mean, expected"
                " size is " << requiredDims << " whereas actual size is "
                << mMean->dims() << std::endl;

            throw std::runtime_error(msgStr.str());
        }
    }

    if (mVariance->empty())
        mVariance->resize(requiredDims, T(0.0));
    else {
        if (mVariance->dims() != requiredDims) {
            std::stringstream msgStr;
            msgStr << "BatchNormCell_Frame<T>::initialize():"
                " in cell " + mName + ", wrong size for shared variance, expected"
                " size is " << requiredDims << " whereas actual size is "
                << mVariance->dims() << std::endl;

            throw std::runtime_error(msgStr.str());
        }
    }

    mSavedMean.resize(requiredDims);
    mSavedVariance.resize(requiredDims);

    mDiffScale.resize(requiredDims);
    mDiffBias.resize(requiredDims);
    mDiffSavedMean.resize(requiredDims);
    mDiffSavedVariance.resize(requiredDims);
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::propagate(bool inference)
{
    unsigned int outputOffset = 0;

    for (unsigned int k = 0, kSize = mInputs.size(); k < kSize; ++k) {
        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);
        const unsigned int size = mInputs.dimB() * input.dimZ();

        if (inference) {
#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < input.dimZ();
                    ++channel)
                {
                    const unsigned int output = outputOffset + channel;
                    const T var(std::sqrt((*mVariance)(output)
                                                  + mEpsilon));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T normalized
                                = (input(ox, oy, channel, batchPos)
                                   - (*mMean)(output)) / var;
                            mOutputs(ox, oy, output, batchPos)
                                = (*mScale)(output) * normalized
                                    + (*mBias)(output);
                        }
                    }
                }
            }
        } else {
            const unsigned int size = input.dimX() * input.dimY()
                                      * mInputs.dimB();
            // Cumulative Moving Average (CMA)
            const double expAverageFactor = 1.0 / (1.0 + mNbPropagate);

#pragma omp parallel for if (input.dimZ() > 16)
            for (int channel = 0; channel < (int)input.dimZ(); ++channel) {
                const unsigned int output = outputOffset + channel;
                T sum(0.0);

                for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox)
                            sum += input(ox, oy, channel, batchPos);
                    }
                }

                mSavedMean(output) = sum / (T)size;

                sum = 0.0;

                for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T zeroed = input(ox, oy, channel, batchPos)
                                                   - mSavedMean(output);
                            sum += zeroed * zeroed;
                        }
                    }
                }

                mSavedVariance(output) = sum / (T)size;

                (*mMean)(output) = mSavedMean(output) * expAverageFactor
                                + (*mMean)(output) * (1.0 - expAverageFactor);
                (*mVariance)(output) = mSavedVariance(output) * expAverageFactor
                                    + (*mVariance)(output) * (1.0 - expAverageFactor);
            }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
                    const unsigned int output = outputOffset + channel;
                    const T var(std::sqrt(mSavedVariance(output) + mEpsilon));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T normalized
                                = (input(ox, oy, channel, batchPos)
                                   - mSavedMean(output)) / var;
                            mOutputs(ox, oy, output, batchPos)
                                = (*mScale)(output) * normalized + (*mBias)(output);
                        }
                    }
                }
            }

            ++mNbPropagate;
        }

        outputOffset += input.dimZ();
    }

    Cell_Frame<T>::propagate();
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::backPropagate()
{
    Cell_Frame<T>::backPropagate();

    const T betaScale = (mScaleSolver->isNewIteration()) ? T(0.0) : T(1.0);
    const T betaBias = (mBiasSolver->isNewIteration()) ? T(0.0) : T(1.0);
    unsigned int outputOffset = 0;

    for (unsigned int k = 0, kSize = mInputs.size(); k < kSize; ++k) {
        const Tensor<T>& input = tensor_cast_nocopy<T>(mInputs[k]);
        const unsigned int size = input.dimX() * input.dimY() * mInputs.dimB();

#pragma omp parallel for if (input.dimZ() > 16)
        for (int channel = 0; channel < (int)input.dimZ(); ++channel) {
            const unsigned int output = outputOffset + channel;
            const T var(std::sqrt(mSavedVariance(output) + mEpsilon));

            T sumScale(0.0);
            T sumBias(0.0);
            T sumMean1(0.0);
            T sumMean2(0.0);
            T sumVariance(0.0);

            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                        const T normalized((input(ox, oy, channel, batchPos)
                               - mSavedMean(output)) / var);
                        const T diffNormalized(
                            mDiffInputs(ox, oy, output, batchPos)
                              * (*mScale)(output));

                        sumScale += mDiffInputs(ox, oy, output, batchPos)
                                    * normalized;
                        sumBias += mDiffInputs(ox, oy, output, batchPos);

                        sumMean1 += diffNormalized;
                        sumMean2 += -2.0 * (input(ox, oy, channel, batchPos)
                                            - mSavedMean(output));
                        sumVariance += diffNormalized
                                       * (input(ox, oy, channel, batchPos)
                                          - mSavedMean(output));
                    }
                }
            }

            mDiffSavedVariance(output)
                = sumVariance * (-1.0 / 2.0)
                  * std::pow(mSavedVariance(output) + mEpsilon, -3.0 / 2.0);
            mDiffSavedMean(output) = sumMean1 * (-1.0 / var)
                                     + mDiffSavedVariance(output) * sumMean2
                                       / (T)size;

            mDiffScale(output) = sumScale + betaScale * mDiffScale(output);
            mDiffBias(output) = sumBias + betaBias * mDiffBias(output);
        }

        if (!mDiffOutputs.empty()) {
            const unsigned int size = mInputs.dimB() * getNbOutputs();
            const bool isValid = mDiffOutputs[k].isValid();
            Tensor<T> diffOutput = (isValid)
                ? tensor_cast<T>(mDiffOutputs[k])
                : tensor_cast_nocopy<T>(mDiffOutputs[k]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < input.dimZ();
                    ++channel)
                {
                    const unsigned int output = outputOffset + channel;
                    const T var(std::sqrt(mSavedVariance(output) + mEpsilon));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T diffNormalized(
                                mDiffInputs(ox, oy, output, batchPos)
                                  * (*mScale)(output));
                            const T gradient(diffNormalized / var
                                  + mDiffSavedVariance(output) * 2.0
                                    * (input(ox, oy, channel, batchPos)
                                       - mSavedMean(output)) / (T)size
                                  + mDiffSavedMean(output) / (T)size);

                            diffOutput(ox, oy, channel, batchPos)
                                = gradient
                                  + isValid
                                    * diffOutput(ox, oy, channel, batchPos);
                        }
                    }
                }
            }

            mDiffOutputs[k] = diffOutput;
        }

        outputOffset += input.dimZ();
    }

    if (!mDiffOutputs.empty()) {
        mDiffOutputs.setValid();
        mDiffOutputs.synchronizeHToD();
    }
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::update()
{
    assert(mScale->size() == mDiffScale.size());
    assert(mBias->size() == mDiffBias.size());
    assert(mScale->size() == mBias->size());

    mScaleSolver->update(*mScale, mDiffScale, mInputs.dimB());
    mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&BatchNormCell_Frame<T>::propagate, this, false),
                  std::bind(&BatchNormCell_Frame<T>::backPropagate, this));
    gc.check(mName + "_mDiffSavedMean", mSavedMean, mDiffSavedMean);
    gc.check(mName + "_mDiffSavedVariance", mSavedVariance, mDiffSavedVariance);
    gc.check(mName + "_mDiffScale", (*mScale), mDiffScale);
    gc.check(mName + "_mDiffBias", (*mBias), mDiffBias);

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::saveFreeParameters(const std::string
                                                   & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create parameter file (.SYN): "
                                 + fileName);

    for (typename std::vector<T>::const_iterator it = mScale->begin();
         it != mScale->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::const_iterator it = mBias->begin();
         it != mBias->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::const_iterator it = mMean->begin();
         it != mMean->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::const_iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    if (!syn.good())
        throw std::runtime_error("Error writing parameter file: " + fileName);
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::loadFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open parameter file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open parameter file (.SYN): "
                                     + fileName);
    }

    for (typename std::vector<T>::iterator it = mScale->begin(); it != mScale->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::iterator it = mBias->begin(); it != mBias->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::iterator it = mMean->begin(); it != mMean->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (typename std::vector<T>::iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in parameter file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading parameter file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

template <class T>
N2D2::BatchNormCell_Frame<T>::~BatchNormCell_Frame()
{
}

namespace N2D2 {
    template class BatchNormCell_Frame<half_float::half>;
    template class BatchNormCell_Frame<float>;
    template class BatchNormCell_Frame<double>;
}
