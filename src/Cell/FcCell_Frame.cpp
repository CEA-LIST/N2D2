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

#include "Cell/FcCell_Frame.hpp"

N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame::mRegistrar("Frame", N2D2::FcCell_Frame::create);

N2D2::FcCell_Frame::FcCell_Frame(const std::string& name,
                                 unsigned int nbOutputs,
                                 const std::shared_ptr
                                 <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      FcCell(name, nbOutputs),
      Cell_Frame(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().,
      mDropConnect(this, "DropConnect", 1.0),
      mLockRandom(false)
{
    // ctor
    mWeightsSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
}

void N2D2::FcCell_Frame::initialize()
{
    if (!mNoBias) {
        mBias.resize(mOutputs.dimZ());
        mDiffBias.resize(mOutputs.dimZ());
        mBiasFiller->apply(mBias);
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FcCell " + mName);

        mWeightsSolvers.push_back(mWeightsSolver->clone());
        mSynapses.push_back(new Tensor4d<Float_T>(
            1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()));
        mDiffSynapses.push_back(new Tensor4d<Float_T>(
            1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()));
        mDropConnectMask.push_back(new Tensor4d<bool>(
            1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ(), true));
        mWeightsFiller->apply(mSynapses.back());
    }
}

void N2D2::FcCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();

    const unsigned int outputSize = mOutputs.dimX() * mOutputs.dimY()
                                    * mOutputs.dimZ();
    const unsigned int count = mInputs.dimB() * outputSize;

    Float_T beta = 0.0;

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

        const Tensor4d<Float_T>& synapses = mSynapses[k];

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && count > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int output = 0; output < outputSize; ++output) {
                const Tensor3d<Float_T> inputs = mInputs[k][batchPos];
                const int inputSize = inputs.size();

                // Compute the weighted sum
                Float_T weightedSum = (!mNoBias) ? mBias(output) : 0.0;

                if (mDropConnect < 1.0 && !inference) {
                    for (int channel = 0; channel < inputSize; ++channel) {
                        if (mDropConnectMask[k](channel, output))
                            weightedSum += inputs(channel)
                                           * synapses(channel, output);
                    }
                } else {
                    // init with weightedSum and not 0.0 to match for loop
                    // (otherwise it can lead to different result because of
                    // limited machine precision)
                    weightedSum = std::inner_product(inputs.begin(),
                                                     inputs.end(),
                                                     synapses[output].begin(),
                                                     weightedSum);
                }

                mOutputs(output, batchPos)
                    = weightedSum + beta * mOutputs(output, batchPos);
            }
        }
    }

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::FcCell_Frame::backPropagate()
{
    Cell_Frame::backPropagate();

    const unsigned int outputSize = mOutputs.dimX() * mOutputs.dimY()
                                    * mOutputs.dimZ();

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Tensor4d<Float_T>& input = mInputs[k];
        const unsigned int nbChannels = input.size() / input.dimB();

        if (!mDiffOutputs.empty() && mBackPropagate) {
            Tensor4d<Float_T>& diffOutputs = mDiffOutputs[k];
            const Float_T beta = (diffOutputs.isValid()) ? 1.0 : 0.0;

            const Tensor4d<Float_T>& synapses = mSynapses[k];
            const unsigned int count = mInputs.dimB() * nbChannels;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && count > 16)
#endif
            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int channel = 0; channel < nbChannels; ++channel)
                {
                    const Tensor3d<Float_T> diffInputs = mDiffInputs[batchPos];

                    Float_T gradient = 0.0;

                    if (mDropConnect < 1.0) {
                        for (unsigned int output = 0; output < outputSize;
                             ++output)
                        {
                            if (mDropConnectMask[k](channel, output))
                                gradient += synapses(channel, output)
                                            * diffInputs(output);
                        }
                    }
                    else {
                        for (unsigned int output = 0; output < outputSize;
                             ++output)
                        {
                            gradient += synapses(channel, output)
                                        * diffInputs(output);
                        }
                    }

                    diffOutputs(channel, batchPos) = gradient
                        + beta * diffOutputs(channel, batchPos);
                }
            }

            diffOutputs.setValid();
        }

        Tensor4d<Float_T>& diffSynapses = mDiffSynapses[k];
        const unsigned int count2 = nbChannels * mNbOutputs;

        const float beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (count2 > 16)
#else
#pragma omp parallel for if (mNbOutputs > 4 && count2 > 16)
#endif
        for (int output = 0; output < (int)mNbOutputs; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                if (!(mDropConnect < 1.0)
                    || mDropConnectMask[k](channel, output)) {
                    Float_T sum = 0.0;

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

#pragma omp parallel for if (mNbOutputs > 16)
        for (int output = 0; output < (int)mNbOutputs; ++output) {
            Float_T sum = 0.0;

            for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                 ++batchPos)
                sum += mDiffInputs(output, batchPos);

            mDiffBias(output) = sum + beta * mDiffBias(output);
        }
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::FcCell_Frame::update()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]
            ->update(&mSynapses[k], &mDiffSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(&mBias, &mDiffBias, mInputs.dimB());
}

void N2D2::FcCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&FcCell_Frame::propagate, this, false),
                  std::bind(&FcCell_Frame::backPropagate, this));

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

void N2D2::FcCell_Frame::saveFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (unsigned int k = 0; k < mSynapses.size(); ++k) {
        for (std::vector<Float_T>::const_iterator it = mSynapses[k].begin();
             it != mSynapses[k].end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    for (std::vector<Float_T>::const_iterator it = mBias.begin();
         it != mBias.end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

void N2D2::FcCell_Frame::loadFreeParameters(const std::string& fileName,
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

    for (unsigned int k = 0; k < mSynapses.size(); ++k) {
        for (std::vector<Float_T>::iterator it = mSynapses[k].begin();
             it != mSynapses[k].end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    for (std::vector<Float_T>::iterator it = mBias.begin(); it != mBias.end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

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

N2D2::FcCell_Frame::~FcCell_Frame()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        delete &mSynapses[k];
}
