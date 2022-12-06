/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell_Frame.hpp"
#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_Kernels.hpp"

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<float>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<double>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<double>());


namespace N2D2 {

template<class T>
PruneQuantizerCell_Frame<T>::PruneQuantizerCell_Frame()
    : PruneQuantizerCell(),
      QuantizerCell_Frame<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<T> >();
}


template<class T>
void PruneQuantizerCell_Frame<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new Tensor<T>(weights.dims()));
    mMasksWeights.push_back(new Tensor<unsigned int>(weights.dims(), 1U));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new Tensor<T>(diffWeights.dims()));
}

template<class T>
void PruneQuantizerCell_Frame<T>::addBiases(BaseTensor& /*biases*/, BaseTensor& /*diffBiases*/)
{
    if(mInitialized)
        return;
}


template<typename T>
void PruneQuantizerCell_Frame<T>::initialize()
{
    mNbZeroMaxWeights = std::ceil(mThreshold * mMasksWeights.dataSize());

    // initialize masks
    switch (mPruningMode) {

    case Identity:
        // Nothing to do
        break;
    case Static:
    {
        unsigned int nbZeroMax = std::floor(mNbZeroMaxWeights / mMasksWeights.size());

        for (unsigned int k = 0, size = mMasksWeights.size(); k < size; ++k) {
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);
            PruneQuantizer_Frame_Kernels::update_masks_random(mask, mNbZeroWeights, nbZeroMax);
        }
        break;
    }
    case Gradual:
    {
        break;
    }
    default:
        // Should never be here
        break;
    }
    
    mInitialized = true;
}


template<typename T>
void PruneQuantizerCell_Frame<T>::propagate()
{
    switch (mPruningMode) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
            Tensor<T> fpW = tensor_cast<T>(mFullPrecisionWeights[k]);
            Tensor<T> pW = tensor_cast<T>(mQuantizedWeights[k]);
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

            PruneQuantizer_Frame_Kernels::apply_pruning_with_masks(fpW, pW, mask);
        }
        break;
    }
    default:
        // Should never be here
        break;
    }
}


template<typename T>
void PruneQuantizerCell_Frame<T>::back_propagate()
{
    switch (mPruningMode) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
            Tensor<T> diffW = tensor_cast<T>(mDiffFullPrecisionWeights[k]);
            Tensor<T> diffPrunedW = tensor_cast<T>(mDiffQuantizedWeights[k]);
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

            PruneQuantizer_Frame_Kernels::apply_pruning_with_masks(diffPrunedW, diffW, mask);
        }
        break;
    }
    default:
        // Should never be here
        break;
    }
}


template<typename T>
void PruneQuantizerCell_Frame<T>::update(unsigned int /*batchSize*/)
{
    // Nothing to update
}


template <class T>
PruneQuantizerCell_Frame<T>::~PruneQuantizerCell_Frame()
{
    // dtor
}


template <class T>
void PruneQuantizerCell_Frame<T>::exportFreeParameters(const std::string& /*fileName*/) const 
{

}

template <class T>
void PruneQuantizerCell_Frame<T>::importFreeParameters(const std::string& /*fileName*/, 
                                                       bool /*ignoreNotExists*/)
{

}

}
