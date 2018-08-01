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
                                 const std::vector<unsigned int>& kernelDims,
                                 unsigned int nbOutputs,
                                 const std::vector<unsigned int>& strideDims,
                                 const std::vector<int>& paddingDims,
                                 const std::shared_ptr
                                         <Activation>& activation)
    : Cell(name, nbOutputs),
      DeconvCell(name,
                 kernelDims,
                 nbOutputs,
                 strideDims,
                 paddingDims),
      Cell_Frame(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(std::make_shared<Tensor<Float_T> >()),
      mDiffBias({1, 1, getNbOutputs(), 1}),
      mConvDesc(std::vector<unsigned int>({1, 1}), strideDims, paddingDims)
{
    // ctor
    if (kernelDims.size() != 2) {
        throw std::domain_error("DeconvCell_Frame: only 2D convolution is"
                                " supported");
    }

    if (strideDims.size() != kernelDims.size()) {
        throw std::domain_error("DeconvCell_Frame: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the kernel.");
    }

    if (paddingDims.size() != kernelDims.size()) {
        throw std::domain_error("DeconvCell_Frame: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the kernel.");
    }

    mWeightsSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
}

void N2D2::DeconvCell_Frame::initialize()
{
    if (!mNoBias) {
        if (mBias->empty()) {
            mBias->resize({1, 1, getNbOutputs(), 1});
            mBiasFiller->apply((*mBias));
        }
        else {
            if (mBias->dimX() != 1 || mBias->dimY() != 1
                || mBias->dimZ() != getNbOutputs() || mBias->dimB() != 1)
            {
                throw std::runtime_error("DeconvCell_Frame::initialize(): in "
                    "cell " + mName + ", wrong size for shared bias");
            }
        }
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DeconvCell "
                                     + mName);
        }

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        std::map<unsigned int,
            std::pair<Interface<Float_T>*, unsigned int> >::const_iterator
                it = mExtSharedSynapses.find(k);

        std::vector<size_t> kernelDims(mKernelDims.begin(), mKernelDims.end());
        kernelDims.push_back(getNbOutputs());
        kernelDims.push_back(mInputs[k].dimZ());

        if (it != mExtSharedSynapses.end()) {
            Tensor<Float_T>* extWeights
                = &(*((*it).second.first))[(*it).second.second];

            if (!std::equal(kernelDims.begin(), kernelDims.end(),
                            extWeights->dims().begin()))
            {
                std::stringstream errorStr;
                errorStr << "DeconvCell_Frame::initialize(): in cell "
                    << mName << ", mismatch between external weights dim. ("
                    << extWeights->dims() << ") and expected dim. ("
                    << kernelDims << ")";

                throw std::runtime_error(errorStr.str());
            }

            mSharedSynapses.push_back(extWeights);
        }
        else {
            // Weight filler expect dimZ as input and dimB as output
            std::vector<size_t> fillerKernelDims(kernelDims);
            std::swap(fillerKernelDims.back(),
                      fillerKernelDims[kernelDims.size() - 2]);

            Tensor<Float_T>* sharedSynapses
                = new Tensor<Float_T>(fillerKernelDims);
            mWeightsFiller->apply(*sharedSynapses);
            // Inverse dimZ and dimB for Deconv
            sharedSynapses->reshape(kernelDims);

            mSharedSynapses.push_back(sharedSynapses);
        }

        mDiffSharedSynapses.push_back(new Tensor<Float_T>(kernelDims));
    }
}

void N2D2::DeconvCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();

    const Float_T alpha = 1.0;
    Float_T beta = 0.0;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[k]);

        ConvCell_Frame_Kernels::backwardData<Float_T>(&alpha,
                                             mSharedSynapses[k],
                                             input,
                                             mConvDesc,
                                             &beta,
                                             mOutputs,
                                             mMaps.rows(offset,
                                                        mInputs[k].dimZ()));

        offset += mInputs[k].dimZ();
    }

    if (!mNoBias)
        ConvCell_Frame_Kernels::forwardBias<Float_T>(&alpha, (*mBias), &alpha, mOutputs);

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::DeconvCell_Frame::backPropagate()
{
    Cell_Frame::backPropagate();

    const Float_T alpha = 1.0;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Float_T beta = (mWeightsSolvers[k]->isNewIteration())
            ? 0.0f : 1.0f;

        const Tensor<Float_T>& input = tensor_cast_nocopy<Float_T>(mInputs[k]);

        ConvCell_Frame_Kernels::backwardFilter<Float_T>(&alpha,
                                               mDiffInputs,
                                               input,
                                               mConvDesc,
                                               &beta,
                                               mDiffSharedSynapses[k],
                                               mMaps.rows(offset,
                                                          mInputs[k].dimZ()));

        offset += mInputs[k].dimZ();
    }

    if (!mNoBias) {
        const Float_T beta = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

        ConvCell_Frame_Kernels::backwardBias<Float_T>(&alpha, mDiffInputs,
                                             &beta, mDiffBias);
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        offset = 0;

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

            Tensor<Float_T> diffOutput = (mDiffOutputs[k].isValid())
                ? tensor_cast<Float_T>(mDiffOutputs[k])
                : tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

            ConvCell_Frame_Kernels::forward<Float_T>(&alpha,
                                            mDiffInputs,
                                            mSharedSynapses[k],
                                            mConvDesc,
                                            &beta,
                                            diffOutput,
                                            mMaps.rows(offset,
                                                       mInputs[k].dimZ()));

            offset += mInputs[k].dimZ();

            mDiffOutputs[k] = diffOutput;
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeHToD();
    }
}

void N2D2::DeconvCell_Frame::update()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]->update(
            mSharedSynapses[k], mDiffSharedSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
}

void N2D2::DeconvCell_Frame::setWeights(unsigned int k,
                                        Interface<Float_T>* weights,
                                        unsigned int offset)
{
    mExtSharedSynapses[k] = std::make_pair(weights, offset);
}

void N2D2::DeconvCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
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
        gc.check(mName + "_mDiffBias", (*mBias), mDiffBias);

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
        for (std::vector<Float_T>::const_iterator it = mBias->begin();
             it != mBias->end();
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
        for (std::vector<Float_T>::iterator it = mBias->begin();
             it != mBias->end();
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
