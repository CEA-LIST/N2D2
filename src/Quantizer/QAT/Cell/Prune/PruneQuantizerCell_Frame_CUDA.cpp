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


#ifdef CUDA

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_Kernels.hpp"


template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::PruneQuantizerCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::PruneQuantizerCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::PruneQuantizerCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<double>());


namespace N2D2 {

template<class T>
PruneQuantizerCell_Frame_CUDA<T>::PruneQuantizerCell_Frame_CUDA()
    : PruneQuantizerCell(),
      QuantizerCell_Frame_CUDA<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
}


template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized){
        //reset all refs, as they were changed by calling initialize() again
        mFullPrecisionWeights.clear();
        mDiffQuantizedWeights.clear();

        mFullPrecisionWeights.push_back(&weights);
        mDiffQuantizedWeights.push_back(&diffWeights);
        return;
    }

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new CudaTensor<T>(weights.dims()), 0);
    mMasksWeights.push_back(new CudaTensor<unsigned int>(weights.dims()), 0);
    mMasksWeights.back().fill(1U);

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new CudaTensor<T>(diffWeights.dims()), 0);
}

template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::addBiases(BaseTensor& /*biases*/, BaseTensor& /*diffBiases*/)
{
    if(mInitialized)
        return;
}

template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::initialize()
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
            CudaTensor<unsigned int> mask = cuda_tensor_cast<unsigned int>(mMasksWeights[k]);
            PruneQuantizer_Frame_Kernels::update_masks_random(mask, mNbZeroWeights, nbZeroMax);
            mask.synchronizeHToD();
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

template<>
void PruneQuantizerCell_Frame_CUDA<half_float::half>::propagate()
{

}

template<>
void PruneQuantizerCell_Frame_CUDA<float>::propagate()
{
    switch (mPruningMode) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
            std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
                = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

            PruneQuantizer_Frame_CUDA_Kernels::apply_pruning_with_masks_F(
                fullPrecWeights->getDevicePtr(),
                mQuantizedWeights[k].getDevicePtr(), 
                mMasksWeights[k].getDevicePtr(), 
                mFullPrecisionWeights[k].size()
            );
        }
        break;
    }
    default:
        // Should never be here
        break;
    }
}

template<>
void PruneQuantizerCell_Frame_CUDA<double>::propagate()
{

}

template<>
void PruneQuantizerCell_Frame_CUDA<half_float::half>::back_propagate()
{
    
}

template<>
void PruneQuantizerCell_Frame_CUDA<float>::back_propagate()
{
    switch (mPruningMode) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
            std::shared_ptr<CudaDeviceTensor<float> > diffPrunedW
                = cuda_device_tensor_cast<float>(mDiffQuantizedWeights[k]);

            PruneQuantizer_Frame_CUDA_Kernels::apply_pruning_with_masks_F(
                diffPrunedW->getDevicePtr(),
                mDiffFullPrecisionWeights[k].getDevicePtr(), 
                mMasksWeights[k].getDevicePtr(), 
                mDiffFullPrecisionWeights[k].size()
            );
        }
        break;
    }
    default:
        // Should never be here
        break;
    }
}

template<>
void PruneQuantizerCell_Frame_CUDA<double>::back_propagate()
{
    
}


template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::update(unsigned int /*batchSize*/)
{
    // Nothing to update
}


template <class T>
PruneQuantizerCell_Frame_CUDA<T>::~PruneQuantizerCell_Frame_CUDA()
{
    // dtor
}


template <class T>
void PruneQuantizerCell_Frame_CUDA<T>::exportFreeParameters(const std::string& /*fileName*/) const 
{
    
}

template <class T>
void PruneQuantizerCell_Frame_CUDA<T>::importFreeParameters(const std::string& /*fileName*/, 
                                                          bool /*ignoreNotExists*/)
{
    
}


}

namespace N2D2 {
    template class PruneQuantizerCell_Frame_CUDA<half_float::half>;
    template class PruneQuantizerCell_Frame_CUDA<float>;
    template class PruneQuantizerCell_Frame_CUDA<double>;
}

#endif