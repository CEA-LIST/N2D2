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
#include "DeepNet.hpp"
#include "GradientCheck.hpp"
#include "Solver/SGDSolver_Frame.hpp"
#include "third_party/half.hpp"

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
    const DeepNet& deepNet,
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      BatchNormCell(deepNet, name, nbOutputs),
      Cell_Frame<T>(deepNet, name, nbOutputs, activation),
      mScale(std::make_shared<Tensor<ParamT> >()),
      mBias(std::make_shared<Tensor<ParamT> >()),
      mMean(std::make_shared<Tensor<ParamT> >()),
      mVariance(std::make_shared<Tensor<ParamT> >())
{
    // ctor
    mScaleSolver = std::make_shared<SGDSolver_Frame<ParamT> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<ParamT> >();
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("BatchNormCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    mNbPropagate = 0;

    std::vector<size_t> requiredDims(mInputs[0].nbDims(), 1);
    requiredDims[mInputs[0].nbDims() - 2] = mInputs.dimZ();

    if (mScale->empty())
        mScale->resize(requiredDims, ParamT(1.0));
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
        mBias->resize(requiredDims, ParamT(0.0));
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
        mMean->resize(requiredDims, ParamT(0.0));
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
        mVariance->resize(requiredDims, ParamT(0.0));
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
    if(mMovingAverageMomentum < 0.0 || mMovingAverageMomentum >= 1.0)
    {
        std::stringstream msgStr;
        msgStr << "BatchNormCell_Frame<T>::initialize():"
            " in cell " + mName + ", wrong value for MovingAverageMomentum. "
            "Expected value range [0.0, 1.0[ whereas actual value is "
            << mMovingAverageMomentum << std::endl;

        throw std::runtime_error(msgStr.str());

    }

}




template <class T>
void N2D2::BatchNormCell_Frame<T>::initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs)
{
    // BEGIN: addition to initialize()
    if (nbInputs != 1) {
          throw std::runtime_error("nbInputs != 1 for cell " + mName);
    }
    // TODO: This is only required because getNbChannels() uses the input tensor dimensions to infer the number of input channels. 
    // However, this requires a reinitialization of the input dims which is unsafe
    setInputsDims({nbInputChannels});
    // END: addition to initialize()

    //std::vector<size_t> requiredDims(mInputs[0].nbDims(), 1);
    //requiredDims[mInputs[0].nbDims() - 2] = mInputs.dimZ();

    // NOTE/TODO: In contrast to normal initialize, this works only for 4D Tensors at the moment!
    std::vector<size_t> requiredDims(4, 1);
    requiredDims[2] = nbInputChannels;

    if (mScale->empty())
        mScale->resize(requiredDims, ParamT(1.0));
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
        mBias->resize(requiredDims, ParamT(0.0));
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
        mMean->resize(requiredDims, ParamT(0.0));
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
        mVariance->resize(requiredDims, ParamT(0.0));
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
    if(mMovingAverageMomentum < 0.0 || mMovingAverageMomentum >= 1.0)
    {
        std::stringstream msgStr;
        msgStr << "BatchNormCell_Frame<T>::initialize():"
            " in cell " + mName + ", wrong value for MovingAverageMomentum. "
            "Expected value range [0.0, 1.0[ whereas actual value is "
            << mMovingAverageMomentum << std::endl;

        throw std::runtime_error(msgStr.str());

    }
}


template <class T>
void N2D2::BatchNormCell_Frame<T>::check_input()
{
    if (mInputs.size() == 0) {
          throw std::runtime_error("mInputs.size() = 0 for cell " + mName);
    }

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("BatchNormCell_Frame<T>::initializeDataDependent():"
                            " the number of output channels must be equal "
                            "to the sum of inputs channels.");
    }

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("BatchNormCell_Frame<T>::initializeDataDependent():"
                            " the number of output channels must be equal "
                            "to the sum of inputs channels.");
    }
}


template <class T>
void N2D2::BatchNormCell_Frame<T>::initializeDataDependent(){
    Cell_Frame<T>::initializeDataDependent();

    check_input();

    mNbPropagate = 0;
}



template <class T>
void N2D2::BatchNormCell_Frame<T>::propagate(bool inference)
{
    check_input();
    mInputs.synchronizeDBasedToH();
    unsigned int outputOffset = 0;

    for (unsigned int k = 0, kSize = mInputs.size(); k < kSize; ++k) {
        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);
        const unsigned int size = mInputs.dimB() * input.dimZ();

        if (inference || mMovingAverageMomentum == 0.0) {
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
                    const T var(std::sqrt( T((*mVariance)(output))
                                                  + T(mEpsilon)));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T normalized
                                = (input(ox, oy, channel, batchPos)
                                   - T((*mMean)(output)) )  / var;
                            mOutputs(ox, oy, output, batchPos)
                                = T((*mScale)(output)) * normalized
                                    + T((*mBias)(output));
                        }
                    }
                }
            }
        } else {
            const unsigned int size = input.dimX() * input.dimY()
                                      * mInputs.dimB();

#pragma omp parallel for if (input.dimZ() > 16)
            for (int channel = 0; channel < (int)input.dimZ(); ++channel) {
                const unsigned int output = outputOffset + channel;
                ParamT sum(0.0);

                for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox)
                            sum += input(ox, oy, channel, batchPos);
                    }
                }

                mSavedMean(output) = sum / (ParamT)size;

                sum = 0.0;

                for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T zeroed = input(ox, oy, channel, batchPos)
                                                   - ((T)mSavedMean(output));
                            sum += (ParamT) (zeroed * zeroed);
                        }
                    }
                }

                mSavedVariance(output) = sum / (ParamT)size;

                (*mMean)(output) = mSavedMean(output) * mMovingAverageMomentum
                                + (*mMean)(output) * (1.0 - mMovingAverageMomentum);
                (*mVariance)(output) = mSavedVariance(output) * ((ParamT)size / ((ParamT)size-1)) * mMovingAverageMomentum
                                    + (*mVariance)(output) * (1.0 - mMovingAverageMomentum);
            }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
                    const unsigned int output = outputOffset + channel;
                    const T var(std::sqrt( (T)mSavedVariance(output) + mEpsilon));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T normalized
                                = (input(ox, oy, channel, batchPos)
                                   - ( (T) mSavedMean(output)) ) / var;
                            mOutputs(ox, oy, output, batchPos)
                                = (T) (*mScale)(output) * normalized + (T) (*mBias)(output);
                        }
                    }
                }
            }

            ++mNbPropagate;
        }

        outputOffset += input.dimZ();
    }

    Cell_Frame<T>::propagate(inference);
    mDiffInputs.clearValid();
    mDiffScale.clearValid();
    mDiffBias.clearValid();
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

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
            const T var(std::sqrt((T) mSavedVariance(output) + mEpsilon));

            ParamT sumScale(0.0);
            ParamT sumBias(0.0);
            ParamT sumMean1(0.0);
            ParamT sumMean2(0.0);
            ParamT sumVariance(0.0);

            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                        const T normalized((input(ox, oy, channel, batchPos)
                               - (T)mSavedMean(output)) / var);
                        const T diffNormalized(
                            mDiffInputs(ox, oy, output, batchPos)
                              * (T)(*mScale)(output));

                        sumScale += (ParamT) (mDiffInputs(ox, oy, output, batchPos)
                                    * normalized);
                        sumBias += (ParamT) (mDiffInputs(ox, oy, output, batchPos));

                        sumMean1 += (ParamT)diffNormalized;
                        sumMean2 += (ParamT) (-2.0 * (input(ox, oy, channel, batchPos)
                                            - mSavedMean(output)));
                        sumVariance += (ParamT) (diffNormalized
                                       * (input(ox, oy, channel, batchPos)
                                          - mSavedMean(output)) );
                    }
                }
            }

            mDiffSavedVariance(output)
                = sumVariance * (-1.0 / 2.0)
                  * std::pow(mSavedVariance(output) + mEpsilon, -3.0 / 2.0);
            mDiffSavedMean(output) = sumMean1 * (-1.0 / (ParamT)var)
                                     + mDiffSavedVariance(output) * sumMean2
                                       / (ParamT)size;

            mDiffScale(output) = sumScale + betaScale * mDiffScale(output);
            mDiffBias(output) = sumBias + betaBias * mDiffBias(output);
        }

        if (!mDiffOutputs[k].empty() && mBackPropagate) {
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
                    const T var(std::sqrt( (T)mSavedVariance(output) + mEpsilon));

                    for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                        for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                            const T diffNormalized(
                                mDiffInputs(ox, oy, output, batchPos)
                                  * (T)(*mScale)(output));
                            const T gradient( diffNormalized / var
                                  + (T)mDiffSavedVariance(output) * 2.0
                                    * (input(ox, oy, channel, batchPos)
                                       -(T) mSavedMean(output)) / (T)size
                                  + (T)mDiffSavedMean(output) / (T)size);

                            diffOutput(ox, oy, channel, batchPos)
                                = (gradient
                                + isValid * diffOutput(ox, oy, channel, batchPos));
                        }
                    }
                }
            }

            mDiffOutputs[k] = diffOutput;
            mDiffOutputs[k].setValid();
            mDiffOutputs[k].synchronizeHToD();
        }

        outputOffset += input.dimZ();
    }

    mDiffScale.setValid();
    mDiffBias.setValid();
}

template <class T>
void N2D2::BatchNormCell_Frame<T>::update()
{
    assert(mScale->size() == mDiffScale.size());
    assert(mBias->size() == mDiffBias.size());
    assert(mScale->size() == mBias->size());

    if (mDiffScale.isValid())
        mScaleSolver->update(*mScale, mDiffScale, mInputs.dimB());

    if (mDiffBias.isValid())
        mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
        
    Cell_Frame<T>::update();
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

    for (unsigned int k = 0; k < mInputs.size(); ++k) {
        if (mDiffOutputs[k].empty()) {
            std::cout << Utils::cwarning << "Empty diff. outputs #" << k
                    << " for cell " << mName
                    << ", could not check the gradient!" << Utils::cdef
                    << std::endl;
            continue;
        }

        std::stringstream name;
        name << mName + "_mDiffOutputs[" << k << "]";

        gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
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

    mScale->save(syn);
    mBias->save(syn);
    mMean->save(syn);
    mVariance->save(syn);

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

    mScale->load(syn);
    mBias->load(syn);
    mMean->load(syn);
    mVariance->load(syn);

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
