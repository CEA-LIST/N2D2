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

#include "Cell/ConvCell_Frame.hpp"

template <>
N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::ConvCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::ConvCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Frame<float>::mRegistrar("Frame",
    N2D2::ConvCell_Frame<float>::create,
    N2D2::Registrar<N2D2::ConvCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Frame<double>::mRegistrar("Frame",
    N2D2::ConvCell_Frame<double>::create,
    N2D2::Registrar<N2D2::ConvCell>::Type<double>());

template <class T>
N2D2::ConvCell_Frame<T>::ConvCell_Frame(const std::string& name,
                                 const std::vector<unsigned int>& kernelDims,
                                 unsigned int nbOutputs,
                                 const std::vector<unsigned int>& subSampleDims,
                                 const std::vector<unsigned int>& strideDims,
                                 const std::vector<int>& paddingDims,
                                 const std::shared_ptr
                                 <Activation>& activation)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelDims,
               nbOutputs,
               subSampleDims,
               strideDims,
               paddingDims),
      Cell_Frame<T>(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(std::make_shared<Tensor<T> >()),
      mDiffBias({1, 1, getNbOutputs(), 1}),
      mConvDesc(subSampleDims, strideDims, paddingDims)
{
    // ctor
    if (kernelDims.size() != 2) {
        throw std::domain_error("ConvCell_Frame: only 2D convolution is"
                                " supported");
    }

    if (subSampleDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Frame: the number of dimensions"
                                " of subSample must match the number of"
                                " dimensions of the kernel.");
    }

    if (strideDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Frame: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the kernel.");
    }

    if (paddingDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Frame: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the kernel.");
    }

    mWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mWeightsSolver = std::make_shared<SGDSolver_Frame<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
void N2D2::ConvCell_Frame<T>::initialize()
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
                throw std::runtime_error("ConvCell_Frame<T>::initialize(): in "
                    "cell " + mName + ", wrong size for shared bias");
            }
        }
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for ConvCell " + mName);

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        typename std::map<unsigned int,
            std::pair<Interface<T>*, unsigned int> >::iterator
                it = mExtSharedSynapses.find(k);

        std::vector<size_t> kernelDims(mKernelDims.begin(), mKernelDims.end());
        kernelDims.push_back(mInputs[k].dimZ());
        kernelDims.push_back(getNbOutputs());

        if (it != mExtSharedSynapses.end()) {
            Tensor<T>* extWeights
                = &((*(*it).second.first)[(*it).second.second]);

            if (!std::equal(kernelDims.begin(), kernelDims.end(),
                            extWeights->dims().begin()))
            {
                std::stringstream errorStr;
                errorStr << "ConvCell_Frame<T>::initialize(): in cell "
                    << mName << ", mismatch between external weights dim. ("
                    << extWeights->dims() << ") and expected dim. ("
                    << kernelDims << ")";

                throw std::runtime_error(errorStr.str());
            }

            mSharedSynapses.push_back(extWeights);
        }
        else {
            mSharedSynapses.push_back(new Tensor<T>(kernelDims), 0);
            mWeightsFiller->apply(mSharedSynapses.back());
        }

        mDiffSharedSynapses.push_back(new Tensor<T>(kernelDims), 0);
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::save(const std::string& dirName) const
{
    Cell_Frame<T>::save(dirName);

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->save(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->save(dirName + "/BiasSolver");
}

template <class T>
void N2D2::ConvCell_Frame<T>::load(const std::string& dirName)
{
    Cell_Frame<T>::load(dirName);

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->load(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->load(dirName + "/BiasSolver");
}

template <class T>
void N2D2::ConvCell_Frame<T>::propagate(bool inference)
{
    mInputs.synchronizeDToH();

    const T alpha = T(1.0);
    T beta = T(0.0);

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);

        ConvCell_Frame_Kernels::forward<T>(&alpha,
                                        input,
                                        mSharedSynapses[k],
                                        mConvDesc,
                                        &beta,
                                        mOutputs,
                                        mMapping.rows(offset, mInputs[k].dimZ()));

        offset += mInputs[k].dimZ();
    }

    if (!mNoBias)
        ConvCell_Frame_Kernels::forwardBias<T>(&alpha, (*mBias), &alpha, mOutputs);

    Cell_Frame<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::ConvCell_Frame<T>::backPropagate()
{
    Cell_Frame<T>::backPropagate();

    const T alpha = T(1.0);

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const T beta = (mWeightsSolvers[k]->isNewIteration())
            ? T(0.0) : T(1.0);

        const Tensor<T>& input = tensor_cast_nocopy<T>(mInputs[k]);

        ConvCell_Frame_Kernels::backwardFilter<T>(&alpha,
                                               input,
                                               mDiffInputs,
                                               mConvDesc,
                                               &beta,
                                               mDiffSharedSynapses[k],
                                               mMapping.rows(offset,
                                                          mInputs[k].dimZ()));

        offset += mInputs[k].dimZ();
    }

    if (!mNoBias) {
        const T beta = (mBiasSolver->isNewIteration()) ? T(0.0) : T(1.0);

        ConvCell_Frame_Kernels::backwardBias<T>(&alpha, mDiffInputs,
                                             &beta, mDiffBias);
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        offset = 0;

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const T beta = (mDiffOutputs[k].isValid()) ? T(1.0) : T(0.0);

            Tensor<T> diffOutput = (mDiffOutputs[k].isValid())
                ? tensor_cast<T>(mDiffOutputs[k])
                : tensor_cast_nocopy<T>(mDiffOutputs[k]);

            ConvCell_Frame_Kernels::backwardData<T>(&alpha,
                                                 mSharedSynapses[k],
                                                 mDiffInputs,
                                                 mConvDesc,
                                                 &beta,
                                                 diffOutput,
                                                 mMapping.rows(offset,
                                                            mInputs[k].dimZ()));

            offset += mInputs[k].dimZ();

            mDiffOutputs[k] = diffOutput;
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeHToD();
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::update()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]->update(
            mSharedSynapses[k], mDiffSharedSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
}

template <class T>
void N2D2::ConvCell_Frame<T>::setWeights(unsigned int k,
                                      BaseInterface* weights,
                                      unsigned int offset)
{
    Interface<T>* weightsInterface = dynamic_cast<Interface<T>*>(weights);

    if (!weightsInterface) {
        throw std::runtime_error("ConvCell_Frame<T>::setWeights(): "
                                 "incompatible types.");
    }

    mExtSharedSynapses[k] = std::make_pair(weightsInterface, offset);
}

template <class T>
void N2D2::ConvCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ConvCell_Frame<T>::propagate, this, false),
                  std::bind(&ConvCell_Frame<T>::backPropagate, this));

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

template <class T>
void N2D2::ConvCell_Frame<T>::saveFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k)
        mSharedSynapses[k].save(syn);

    if (!mNoBias)
        mBias->save(syn);

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

template <class T>
void N2D2::ConvCell_Frame<T>::loadFreeParameters(const std::string& fileName,
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

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k)
        mSharedSynapses[k].load(syn);

    if (!mNoBias)
        mBias->load(syn);

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
N2D2::ConvCell_Frame<T>::~ConvCell_Frame()
{
    //dtor
}

namespace N2D2 {
    template class ConvCell_Frame<half_float::half>;
    template class ConvCell_Frame<float>;
    template class ConvCell_Frame<double>;
}
