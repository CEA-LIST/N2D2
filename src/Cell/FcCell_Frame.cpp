/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "GradientCheck.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Filler/NormalFiller.hpp"
#include "Solver/SGDSolver_Frame.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::FcCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame<float>::mRegistrar("Frame",
    N2D2::FcCell_Frame<float>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame<double>::mRegistrar("Frame",
    N2D2::FcCell_Frame<double>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<double>());

template <class T>
N2D2::FcCell_Frame<T>::FcCell_Frame(const DeepNet& deepNet, const std::string& name,
                                 unsigned int nbOutputs,
                                 const std::shared_ptr
                                 <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      FcCell(deepNet, name, nbOutputs),
      Cell_Frame<T>(deepNet, name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().,
      mDropConnect(this, "DropConnect", 1.0),
      mLockRandom(false)
{
    // ctor
    mWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mTopDownWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mRecWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mWeightsSolver = std::make_shared<SGDSolver_Frame<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
void N2D2::FcCell_Frame<T>::initialize()
{
    if (!mNoBias && mBias.empty()) {
        mBias.resize({mOutputs.dimZ(), 1, 1, 1});
        mDiffBias.resize({mOutputs.dimZ(), 1, 1, 1});
        mBiasFiller->apply(mBias);
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FcCell " + mName);

        if (k < mWeightsSolvers.size())
            continue;  // already initialized, skip!

        mWeightsSolvers.push_back(mWeightsSolver->clone());
        mSynapses.push_back(new Tensor<T>(
            {1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()}));
        mDiffSynapses.push_back(new Tensor<T>(
            {1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()}));
        mDropConnectMask.push_back(new Tensor<bool>(
            {1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()}, true));
        mWeightsFiller->apply(mSynapses.back());
    }
}

template <class T>
void N2D2::FcCell_Frame<T>::save(const std::string& dirName) const
{
    Cell_Frame<T>::save(dirName);

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->save(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->save(dirName + "/BiasSolver");
}

template <class T>
void N2D2::FcCell_Frame<T>::load(const std::string& dirName)
{
    Cell_Frame<T>::load(dirName);

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->load(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->load(dirName + "/BiasSolver");
}

template <class T>
void N2D2::FcCell_Frame<T>::propagate(bool inference)
{
    if (mNormalize) {
        for (unsigned int n = 0, nbOutputs = mOutputs.dimZ(); n < nbOutputs;
            ++n)
        {
            T sumSq(0.0);

            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                const unsigned int inputSize = mInputs[k].dimX()
                    * mInputs[k].dimY() * mInputs[k].dimZ();
                const Tensor<T>& synapses = mSynapses[k][n];

                for (unsigned int i = 0; i < inputSize; ++i)
                    sumSq += synapses(i) * synapses(i);
            }

            const T scale(1.0 / (std::sqrt(sumSq + 1.0e-6)));

            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                const unsigned int inputSize = mInputs[k].dimX()
                    * mInputs[k].dimY() * mInputs[k].dimZ();
                Tensor<T>& synapses = mSynapses[k];

                for (unsigned int i = 0; i < inputSize; ++i)
                    synapses(i, n) *= scale;
            }
        }
    }

    mInputs.synchronizeDBasedToH();

    const unsigned int outputSize = mOutputs.dimX() * mOutputs.dimY()
                                    * mOutputs.dimZ();
    const unsigned int count = mInputs.dimB() * outputSize;

    T beta(0.0);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        if (mDropConnect < 1.0 && !inference && !mLockRandom) {
            // Random::randBernoulli() is not thread-safe!
            for (unsigned int index = 0; index < mDropConnectMask[k].size();
                 ++index)
                mDropConnectMask[k](index)
                    = Random::randBernoulli(mDropConnect);
        }

        const Tensor<T>& synapses = mSynapses[k];
        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);
        const unsigned int inputSize = input.dimX() * input.dimY()
                                        * input.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && count > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int output = 0; output < outputSize; ++output) {
                // Compute the weighted sum
                T weightedSum((!mNoBias) ? mBias(output) : 0.0);

                if (mDropConnect < 1.0 && !inference) {
                    for (unsigned int channel = 0; channel < inputSize;
                        ++channel)
                    {
                        if (mDropConnectMask[k](channel, output))
                            weightedSum += input(channel, batchPos)
                                           * synapses(channel, output);
                    }
                } else {
                    // init with weightedSum and not 0.0 to match for loop
                    // (otherwise it can lead to different result because of
                    // limited machine precision)
                    weightedSum = std::inner_product(
                                    input.begin() + batchPos * inputSize,
                                    input.begin() + (batchPos + 1) * inputSize,
                                    synapses[output].begin(),
                                    weightedSum);
                }

                mOutputs(output, batchPos)
                    = weightedSum + beta * mOutputs(output, batchPos);
            }
        }
    }

    Cell_Frame<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::FcCell_Frame<T>::backPropagate()
{
    Cell_Frame<T>::backPropagate();

    const unsigned int outputSize = mOutputs.dimX() * mOutputs.dimY()
                                    * mOutputs.dimZ();

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Tensor<T>& input = tensor_cast_nocopy<T>(mInputs[k]);
        const unsigned int nbChannels = input.size() / input.dimB();

        if (!mDiffOutputs.empty() && mBackPropagate) {
            const T beta((mDiffOutputs[k].isValid()) ? 1.0 : 0.0);

            Tensor<T> diffOutput = (mDiffOutputs[k].isValid())
                ? tensor_cast<T>(mDiffOutputs[k])
                : tensor_cast_nocopy<T>(mDiffOutputs[k]);

            const Tensor<T>& synapses = mSynapses[k];
            const unsigned int count = mInputs.dimB() * nbChannels;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && count > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < nbChannels; ++channel)
                {
                    T gradient(0.0);

                    if (mDropConnect < 1.0) {
                        for (unsigned int output = 0; output < outputSize;
                             ++output)
                        {
                            if (mDropConnectMask[k](channel, output))
                                gradient += synapses(channel, output)
                                            * mDiffInputs(output, batchPos);
                        }
                    }
                    else {
                        for (unsigned int output = 0; output < outputSize;
                             ++output)
                        {
                            gradient += synapses(channel, output)
                                        * mDiffInputs(output, batchPos);
                        }
                    }

                    diffOutput(channel, batchPos) = gradient
                        + beta * diffOutput(channel, batchPos);
                }
            }

            mDiffOutputs[k] = diffOutput;
            mDiffOutputs[k].setValid();
        }

        Tensor<T>& diffSynapses = mDiffSynapses[k];
        const unsigned int count2 = nbChannels * getNbOutputs();

        const float beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count2 > 16)
#else
#pragma omp parallel for if (getNbOutputs() > 4 && count2 > 16)
#endif
        for (int output = 0; output < (int)getNbOutputs(); ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                if (!(mDropConnect < 1.0)
                    || mDropConnectMask[k](channel, output)) {
                    T sum(0.0);

                    for (unsigned int batchPos = 0; batchPos < input.dimB();
                         ++batchPos)
                        sum += input(channel, batchPos)
                               * mDiffInputs(output, batchPos);

                    diffSynapses(channel, output) = sum
                        + beta * diffSynapses(channel, output);
                }
                else {
                    diffSynapses(channel, output) = beta
                        * diffSynapses(channel, output);
                }
            }
        }
    }

    if (!mNoBias) {
        const float beta = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

#pragma omp parallel for if (getNbOutputs() > 16)
        for (int output = 0; output < (int)getNbOutputs(); ++output) {
            T sum(0.0);

            for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                 ++batchPos)
                sum += mDiffInputs(output, batchPos);

            mDiffBias(output) = sum + beta * mDiffBias(output);
        }
    }

    mDiffOutputs.synchronizeHToD();
}

template <class T>
void N2D2::FcCell_Frame<T>::update()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]
            ->update(mSynapses[k], mDiffSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(mBias, mDiffBias, mInputs.dimB());
}

template <class T>
void N2D2::FcCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&FcCell_Frame<T>::propagate, this, false),
                  std::bind(&FcCell_Frame<T>::backPropagate, this));

    mLockRandom = true;

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream name;
        name << mName + "_mDiffSynapses[" << k << "]";

        gc.check(name.str(), mSynapses[k], mDiffSynapses[k]);
    }

    if (!mNoBias)
        gc.check(mName + "_mDiffBias", mBias, mDiffBias);

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

    mLockRandom = false;
}

template <class T>
void N2D2::FcCell_Frame<T>::saveFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (unsigned int k = 0; k < mSynapses.size(); ++k)
        mSynapses[k].save(syn);

    if (!mNoBias)
        mBias.save(syn);

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

template <class T>
void N2D2::FcCell_Frame<T>::loadFreeParameters(const std::string& fileName,
                                            bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file (.SYN): "
                                     + fileName);
    }

    for (unsigned int k = 0; k < mSynapses.size(); ++k)
        mSynapses[k].load(syn);

    if (!mNoBias)
        mBias.load(syn);

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

template <class T>
N2D2::FcCell_Frame<T>::~FcCell_Frame()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        delete &mSynapses[k];
}

namespace N2D2 {
    template class FcCell_Frame<half_float::half>;
    template class FcCell_Frame<float>;
    template class FcCell_Frame<double>;
}
