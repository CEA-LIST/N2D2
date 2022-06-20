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

#include "GradientCheck.hpp"
#include "Cell/ConvCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Filler/NormalFiller.hpp"
#include "Solver/SGDSolver_Frame.hpp"
#include "third_party/half.hpp"

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
N2D2::ConvCell_Frame<T>::ConvCell_Frame(const DeepNet& deepNet, const std::string& name,
                                 const std::vector<unsigned int>& kernelDims,
                                 unsigned int nbOutputs,
                                 const std::vector<unsigned int>& subSampleDims,
                                 const std::vector<unsigned int>& strideDims,
                                 const std::vector<int>& paddingDims,
                                 const std::vector<unsigned int>& dilationDims,
                                 const std::shared_ptr
                                 <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      ConvCell(deepNet, name,
               kernelDims,
               nbOutputs,
               subSampleDims,
               strideDims,
               paddingDims,
               dilationDims),
      Cell_Frame<T>(deepNet, name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(std::make_shared<Tensor<T> >()),
      mDiffBias({1, 1, getNbOutputs(), 1}),
      mConvDesc(mSubSampleDims, mStrideDims, mPaddingDims, mDilationDims)
{
    // ctor
    if (mKernelDims.size() != 2) {
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

    if (dilationDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Frame: the number of dimensions"
                                " of dilation must match the number of"
                                " dimensions of the kernel.");
    }

    if (std::count(dilationDims.begin(), dilationDims.end(), 1U)
        != (int)dilationDims.size())
    {
        throw std::domain_error("ConvCell_Frame: dilation != 1 is currently not"
                                " supported.");
    }

    mWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mWeightsSolver = std::make_shared<SGDSolver_Frame<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
void N2D2::ConvCell_Frame<T>::setExtendedPadding(
    const std::vector<int>& paddingDims)
{
    ConvCell::setExtendedPadding(paddingDims);

    for (std::size_t dim = 0; dim < paddingDims.size(); ++dim) {
        mConvDesc.padding[dim] = mPaddingDims[dim % mPaddingDims.size()]
                                    + paddingDims[dim];
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::resetWeights()
{
    for (unsigned int i = 0, size = mSharedSynapses.size(); i < size; i++){
        mWeightsFiller->apply(mSharedSynapses[i]);
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::resetBias()
{
    mBiasFiller->apply(*mBias);
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

    unsigned int nbChannels = 0;

    mNbGroups.clear();
    mDiffSharedSynapses.clear();

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for ConvCell " + mName);

        if (k < mWeightsSolvers.size())
            continue;  // already initialized, skip!

        mNbGroups.push_back(getNbGroups(mMapping.rows(nbChannels,
                                                mInputs[k].dimZ())));
        mWeightsSolvers.push_back(mWeightsSolver->clone());

        typename std::map<unsigned int,
            std::pair<Interface<T>*, unsigned int> >::iterator
                it = mExtSharedSynapses.find(k);

        // Computing a kernel dimension adapted to the mapping in case of a depthwise convolution
        std::vector<size_t> kernelDims(mKernelDims.begin(), mKernelDims.end());
        if (mNbGroups[k] > 1)
            kernelDims.push_back(mInputs[k].dimZ() / mNbGroups[k]);
        else
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

            if (mNbGroups[k] == 0) {
                // Set the non-connected kernels coefficients to 0
                for (unsigned int output = 0; output < getNbOutputs(); ++output)
                {
                    for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                        ++channel) {
                        if (!isConnection(nbChannels + channel, output)) {
                            mSharedSynapses.back()[output][channel].fill(T(0.0));
                        }
                    }
                }
            }
        }

        nbChannels += mInputs[k].dimZ();
        // initialize with tensor filled with value 0
        mDiffSharedSynapses.push_back(new Tensor<T>(kernelDims), 0);
    }
    if (mQuantizer) {
        for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
            mQuantizer->addWeights(mSharedSynapses[k], mDiffSharedSynapses[k]);
        }
        if (!mNoBias) {
            mQuantizer->addBiases(*mBias, mDiffBias);
        }
        mQuantizer->initialize();
    }
}



template <class T>
void N2D2::ConvCell_Frame<T>::initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs)
{
     // BEGIN: addition to initialize()
    // TODO: This is only required because getNbChannels() uses the input tensor dimensions to infer the number of input channels. 
    // However, this requires a reinitialization of the input dims which is unsafe
    setInputsDims({nbInputChannels});
    // END: addition to initialize

    if (mMapping.empty()) {
        mMapping.append(Tensor<bool>({getNbOutputs(), nbInputs*nbInputChannels}, true));
    }
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

    unsigned int nbChannels = 0;

    for (unsigned int k = 0, size = nbInputs; k < size; ++k) {
        if (k < mWeightsSolvers.size())
            continue;  // already initialized, skip!
        
        if (k < mNbGroups.size()) {
            nbChannels += nbInputChannels;
            continue; // already initialized, skip!
        }

        mNbGroups.push_back(getNbGroups(mMapping.rows(nbChannels,
                                                    nbInputChannels)));

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        typename std::map<unsigned int,
            std::pair<Interface<T>*, unsigned int> >::iterator
                it = mExtSharedSynapses.find(k);

        std::vector<size_t> kernelDims(mKernelDims.begin(), mKernelDims.end());
        if (mNbGroups[k] > 1)
            kernelDims.push_back(nbInputChannels / mNbGroups[k]);
        else
            kernelDims.push_back(nbInputChannels);
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

            if (mNbGroups[k] == 0) {
                // Set the non-connected kernels coefficients to 0
                for (unsigned int output = 0; output < getNbOutputs(); ++output)
                {
                    for (unsigned int channel = 0; channel < nbInputChannels;
                         ++channel) {
                        if (!isConnection(nbChannels + channel, output)) {
                            mSharedSynapses.back()[output][channel]
                                                                .fill(T(0.0));
                        }
                    }
                }
            }
        }
        nbChannels += mInputs[k].dimZ();

        mDiffSharedSynapses.push_back(new Tensor<T>(kernelDims), 0);
    }

    initializeWeightQuantizer();
}



template <class T>
void N2D2::ConvCell_Frame<T>::initializeWeightQuantizer()
{
    if (mQuantizer) {
        for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
            mQuantizer->addWeights(mSharedSynapses[k], mDiffSharedSynapses[k]);
        }
        if (!mNoBias) {
            mQuantizer->addBiases(*mBias, mDiffBias);
        }
        mQuantizer->initialize();
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::check_input()
{
    if (mInputs.size() != mSharedSynapses.size()) {
          throw std::runtime_error("mInputs.size() != mSharedSynapses.size() for cell " + mName + 
          ". Please verify that the number of input tensors given to the cell is"
          " equal to the number of inputs defined for the cell.");
    }
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if ((mNbGroups[k] > 0 && mInputs[k].dimZ() != mSharedSynapses[k].dimZ()*mNbGroups[k])
        || (mNbGroups[k] == 0 && mInputs[k].dimZ() != mSharedSynapses[k].dimZ())){
            std::cout << "mInputs.dimZ(): " << mInputs[k].dimZ() << std::endl;
            std::cout << "mSharedSynapses.dimZ(): " << mSharedSynapses[k].dimZ() << std::endl;
            std::cout << "mNbGroups: " << mNbGroups[k] << std::endl;
            std::stringstream ss;
            ss << "Unmatching dimension Z"
            " between input and weight " << k << " for cell " + mName;
            throw std::runtime_error(ss.str());
        }
    }
}


template <class T>
void N2D2::ConvCell_Frame<T>::initializeDataDependent() 
{
    // NOTE: this is addition to initialize()
    Cell_Frame<T>::initializeDataDependent();
    
    check_input();

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

/**
 * @brief Wrapper for the forward function
 * 
 * @tparam T feature type
 * @param inference 
 */
template <class T>
void N2D2::ConvCell_Frame<T>::propagate(bool inference)
{
    check_input(); // right number and sizes

    // if (mInputs.size() < mSharedSynapses.size()) {
    //     throw std::runtime_error("ConvCell_Frame<T>::propagate(): multiple "
    //         "synapse tensors per input is not supported for ConvCell "
    //         + mName);
    // }

    const T alpha = T(1.0); // propagation coefficient applied to the weighted sum
    T beta = T(0.0); // accumultion coefficient to sum the output from several input tensors

    unsigned int offset = 0;

    if (mQuantizer) {
        mQuantizer->propagate(); // quantify weights
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);
        
        // Quantize weights if quantizer is on
        const Tensor<T>& sharedSynapses 
            = mQuantizer ? (tensor_cast<T>(mQuantizer->getQuantizedWeights(k))) 
                        : tensor_cast<T>(mSharedSynapses[k]);

        ConvCell_Frame_Kernels::forward<T>(&alpha,
                                        input,
                                        sharedSynapses,
                                        mConvDesc,
                                        &beta,
                                        mOutputs,
                                        mMapping.rows(offset, mInputs[k].dimZ()));

        offset += mInputs[k].dimZ();
    }

    if (!mNoBias) {
        const Tensor<T>& biases 
            = mQuantizer ? tensor_cast<T>(mQuantizer->getQuantizedBiases())
                        : tensor_cast<T>(*mBias);
        ConvCell_Frame_Kernels::forwardBias<T>(&alpha, biases, &alpha, mOutputs);
    }

    // propagation through the activation function
    Cell_Frame<T>::propagate(inference);
    // The input gradient needs to be computed to allow backpropagation after the propagation
    mDiffInputs.clearValid();
    mDiffSharedSynapses.clearValid();
    mDiffBias.clearValid();
}

template <class T>
void N2D2::ConvCell_Frame<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return; // the gradient from these tensors has already been passed or is not valid yet

    // activation backpropagation
    Cell_Frame<T>::backPropagate();

    const T alpha = T(1.0);
    // Set the non-connected kernels diff to 0
    unsigned int offset = 0;

    const unsigned int kernelSize = (!mKernelDims.empty())
        ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                          1U, std::multiplies<unsigned int>())
        : 0U;

    unsigned int nbChannels = 0;

    if(mInputs.size()==0)
        std::cout << mName << " : no Input" << std::endl;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const T beta = (mWeightsSolvers[k]->isNewIteration())
            ? T(0.0) : T(1.0);

        const Tensor<T>& input = tensor_cast_nocopy<T>(mInputs[k]);

        // 1st: gradient is computed for the quantification layer if any
        Tensor<T> diffSharedSynapses 
            = mQuantizer ? tensor_cast<T>(mQuantizer->getDiffQuantizedWeights(k))
                        : tensor_cast<T>(mDiffSharedSynapses[k]);
        // 2nd: gradient is computed for 
        ConvCell_Frame_Kernels::backwardFilter<T>(&alpha,
                                               input,
                                               mDiffInputs,
                                               mConvDesc,
                                               &beta,
                                               diffSharedSynapses,
                                               mMapping.rows(offset,
                                                          mInputs[k].dimZ()));

        if (mNbGroups[k] == 0) {

            for (unsigned int output = 0; output < getNbOutputs(); ++output) {
                for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                     ++channel) {
                    if (!isConnection(nbChannels + channel, output)) {
                        diffSharedSynapses[output][channel].fill(T(0.0));
                    }

                    offset += kernelSize;
                }
            }
        }
        mDiffSharedSynapses[k].setValid(); // allow synapses to be updated
        offset += mInputs[k].dimZ();
    }

    if (!mNoBias) {
        const T beta = (mBiasSolver->isNewIteration()) ? T(0.0) : T(1.0);
        Tensor<T> diffBiases 
            = mQuantizer ? tensor_cast<T>(mQuantizer->getDiffQuantizedBiases())
                        : tensor_cast<T>(mDiffBias);

        ConvCell_Frame_Kernels::backwardBias<T>(&alpha, mDiffInputs,
                                             &beta, diffBiases);
        
        mDiffBias.setValid();
    }
    // set to true in the ConvCell constructor
    if (mBackPropagate) {
        offset = 0;

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            if (mDiffOutputs[k].empty()) {
                offset += mInputs[k].dimZ();
                continue;
            }

            const T beta = (mDiffOutputs[k].isValid()) ? T(1.0) : T(0.0);

            const Tensor<T>& sharedSynapses 
            = mQuantizer ? tensor_cast<T>(mQuantizer->getQuantizedWeights(k))
                        : tensor_cast<T>(mSharedSynapses[k]);

            Tensor<T> diffOutput = (mDiffOutputs[k].isValid())
                    ? tensor_cast<T>(mDiffOutputs[k])
                    : tensor_cast_nocopy<T>(mDiffOutputs[k]);

            ConvCell_Frame_Kernels::backwardData<T>(&alpha,
                                                 sharedSynapses,
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
    }

    // Calculate full precision weights and activation gradients
    if (mQuantizer && mBackPropagate) {
        mQuantizer->back_propagate();
    }
}

template <class T>
void N2D2::ConvCell_Frame<T>::update()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        if (mDiffSharedSynapses[k].isValid() && !mQuantizer) {
            mWeightsSolvers[k]->update(
                mSharedSynapses[k], mDiffSharedSynapses[k], mInputs.dimB());
        }
        else if (mDiffSharedSynapses[k].isValid() && mQuantizer) {
            mWeightsSolvers[k]->update(
                mSharedSynapses[k], mQuantizer->getDiffFullPrecisionWeights(k), mInputs.dimB());
        }
    }

    if (!mNoBias && mDiffBias.isValid()) {
        if(!mQuantizer) {
            mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
        } else {
            mBiasSolver->update(*mBias, mQuantizer->getDiffFullPrecisionBiases(), mInputs.dimB());
        }
    }

    if(mQuantizer){
        mQuantizer->update((unsigned int)mInputs.dimB());
    }
    Cell_Frame<T>::update();
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
