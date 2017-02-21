/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Cell/DeconvCell_Frame.hpp"

N2D2::Registrar<N2D2::DeconvCell>
N2D2::DeconvCell_Frame::mRegistrar("Frame", N2D2::DeconvCell_Frame::create);

N2D2::DeconvCell_Frame::DeconvCell_Frame(const std::string& name,
                                         unsigned int kernelWidth,
                                         unsigned int kernelHeight,
                                         unsigned int nbOutputs,
                                         unsigned int strideX,
                                         unsigned int strideY,
                                         int paddingX,
                                         int paddingY,
                                         const std::shared_ptr
                                         <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      DeconvCell(name,
                 kernelWidth,
                 kernelHeight,
                 nbOutputs,
                 strideX,
                 strideY,
                 paddingX,
                 paddingY),
      Cell_Frame(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(1, 1, mNbOutputs, 1),
      mDiffBias(1, 1, mNbOutputs, 1),
      mConvDesc(1, 1, strideX, strideY, paddingX, paddingY)
{
    // ctor
    mWeightsSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
}

void N2D2::DeconvCell_Frame::initialize()
{
    if (!mNoBias)
        mBiasFiller->apply(mBias);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DeconvCell "
                                     + mName);
        }

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        // Weight filler expect dimZ as input and dimB as output
        Tensor4d<Float_T>* sharedSynapses = new Tensor4d<Float_T>(
            mKernelWidth, mKernelHeight, mInputs[k].dimZ(), mNbOutputs);
        mWeightsFiller->apply(*sharedSynapses);
        // Inverse dimZ and dimB for Deconv
        sharedSynapses->resize(
            mKernelWidth, mKernelHeight, mNbOutputs, mInputs[k].dimZ());

        mSharedSynapses.push_back(sharedSynapses);
        mDiffSharedSynapses.push_back(new Tensor4d<Float_T>(
            mKernelWidth, mKernelHeight, mNbOutputs, mInputs[k].dimZ()));
    }
}

void N2D2::DeconvCell_Frame::propagate(bool /*inference*/)
{
    const Float_T alpha = 1.0;
    Float_T beta = 0.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        ConvCell_Frame_Kernels::backwardData(
            &alpha, mSharedSynapses[k], mInputs[k], mConvDesc, &beta, mOutputs);
    }

    if (!mNoBias)
        ConvCell_Frame_Kernels::forwardBias(mBias, mOutputs);

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::DeconvCell_Frame::backPropagate()
{
    Cell_Frame::backPropagate();

    const Float_T alpha = 1.0;
    const Float_T beta = 0.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k)
        ConvCell_Frame_Kernels::backwardFilter(&alpha,
                                               mDiffInputs,
                                               mInputs[k],
                                               mConvDesc,
                                               &beta,
                                               mDiffSharedSynapses[k]);

    if (!mNoBias)
        ConvCell_Frame_Kernels::backwardBias(mDiffInputs, mDiffBias);

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

            ConvCell_Frame_Kernels::forward(&alpha,
                                            mDiffInputs,
                                            mSharedSynapses[k],
                                            mConvDesc,
                                            &beta,
                                            mDiffOutputs[k]);
            mDiffOutputs[k].setValid();
        }
    }
}

void N2D2::DeconvCell_Frame::update()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]->update(
            &mSharedSynapses[k], &mDiffSharedSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(&mBias, &mDiffBias, mInputs.dimB());
}

void N2D2::DeconvCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&DeconvCell_Frame::propagate, this, false),
                  std::bind(&DeconvCell_Frame::backPropagate, this));

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        std::stringstream name;
        name << mName + "_mDiffSharedSynapses[" << k << "]";

        gc.check(name.str(), mSharedSynapses[k], mDiffSharedSynapses[k]);
    }

    if (!mNoBias)
        gc.check(mName + "_mDiffBias", mBias, mDiffBias);

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0; k < mInputs.size(); ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

void N2D2::DeconvCell_Frame::saveFreeParameters(const std::string
                                                & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k) {
        for (std::vector<Float_T>::const_iterator it
             = mSharedSynapses[k].begin();
             it != mSharedSynapses[k].end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    if (!mNoBias) {
        for (std::vector<Float_T>::const_iterator it = mBias.begin();
             it != mBias.end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

void N2D2::DeconvCell_Frame::loadFreeParameters(const std::string& fileName,
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

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k) {
        for (std::vector<Float_T>::iterator it = mSharedSynapses[k].begin();
             it != mSharedSynapses[k].end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    if (!mNoBias) {
        for (std::vector<Float_T>::iterator it = mBias.begin();
             it != mBias.end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

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

N2D2::DeconvCell_Frame::~DeconvCell_Frame()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        delete &mSharedSynapses[k];
}
