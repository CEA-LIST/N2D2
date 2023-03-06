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
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"


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
void PruneQuantizerCell_Frame_CUDA<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<CudaBaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<CudaBaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}

template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::initialize()
{
    // Initialize masks if PruningFiller not None
    if (mPruningFiller != None) {
        for (unsigned int k = 0, size = mMasksWeights.size(); k < size; ++k) {
            CudaTensor<T> weights = cuda_tensor_cast<T>(mFullPrecisionWeights[k]);
            CudaTensor<unsigned int> mask = cuda_tensor_cast<unsigned int>(mMasksWeights[k]);

            mCurrentThreshold = 0.0f;

            switch (getPruningMode()) {
            case Identity:
                // Nothing to do
                break;
            case Static:
            {
                mCurrentThreshold = mThreshold;
                break;
            }
            case Gradual:
            {
                mCurrentThreshold = mStartThreshold;
                break;
            }
            default:
                // Should never be here
                break;
            }

            if (mPruningFiller == Random) {
                PruneQuantizer_Frame_Kernels::update_masks_random(mask, mCurrentThreshold);
            } else if (mPruningFiller == IterNonStruct) {
                PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct(weights, mask, mCurrentThreshold, mDelta);
            }
            mask.synchronizeHToD();
        }
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
    switch (getPruningMode()) {
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

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecBiases
            = cuda_device_tensor_cast<float>(
                cuda_tensor_cast<float>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(
            fullPrecBiases->getDevicePtr(), mQuantizedBiases.getDevicePtr(),
            mFullPrecisionBiases->size());
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
    switch (getPruningMode()) {
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
    if (mDiffQuantizedBiases) {

        std::shared_ptr<CudaDeviceTensor<float> > diffQuantBiases
            = cuda_device_tensor_cast<float>(
                cuda_tensor_cast<float>(*mDiffQuantizedBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(
            diffQuantBiases->getDevicePtr(),
            mDiffFullPrecisionBiases.getDevicePtr(),
            mDiffQuantizedBiases->size());
    }
}

template<>
void PruneQuantizerCell_Frame_CUDA<double>::back_propagate()
{
    
}


template<class T>
void PruneQuantizerCell_Frame_CUDA<T>::update(unsigned int batchSize)
{
    if (getPruningMode() == PruningMode::Gradual) {
        if (!mScheduler) {
            unsigned int stepSize = (mStepSizeThreshold > 0) ? mStepSizeThreshold : SGDSolver::mLogSteps;
            mScheduler = std::make_shared<Scheduler>(stepSize, batchSize);
        }
        if(mScheduler->step()) {
            if (mCurrentThreshold < mThreshold) {
                std::cout << "\nUpdate masks" << std::endl;

                // Update threshold
                mCurrentThreshold += mGammaThreshold;

                float maxthreshold = mThreshold;
                mCurrentThreshold = std::min(mCurrentThreshold, maxthreshold);

                for (unsigned int k = 0, size = mMasksWeights.size(); k < size; ++k) {
                    CudaTensor<T> weights = cuda_tensor_cast<T>(mFullPrecisionWeights[k]);
                    CudaTensor<unsigned int> mask = cuda_tensor_cast<unsigned int>(mMasksWeights[k]);

                    if (mPruningFiller == Random) {
                        PruneQuantizer_Frame_Kernels::update_masks_random(mask, mCurrentThreshold);
                    } else if (mPruningFiller == IterNonStruct) {
                        PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct(weights, mask, mCurrentThreshold, mDelta);
                    }

                    mask.synchronizeHToD();
                }
            }
        }
    }
}


template <class T>
PruneQuantizerCell_Frame_CUDA<T>::~PruneQuantizerCell_Frame_CUDA()
{
    // dtor
}


template <class T>
void PruneQuantizerCell_Frame_CUDA<T>::exportFreeParameters(const std::string& fileName) const 
{
    
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string masksWFile = fileBase + "_masks" + fileExt;
    std::ofstream masksW(masksWFile.c_str());

    if (!masksW.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + masksWFile);

    for (unsigned int k = 0; k < mMasksWeights.size(); ++k) {
        CudaTensor<unsigned int> mask = cuda_tensor_cast<unsigned int>(mMasksWeights[k]);
        mask.synchronizeDToH();
        for (unsigned int output = 0; output < mask.dimB(); ++output) {

            for (unsigned int i = 0; i < mask[output].size(); ++i) {
                masksW << mask[output](i) << " ";
            }
            masksW << "\n";
        }
    }
}

template <class T>
void PruneQuantizerCell_Frame_CUDA<T>::importFreeParameters(const std::string& fileName, 
                                                            bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string masksWFile = fileBase + "_masks" + fileExt;
    std::ifstream masksW(masksWFile.c_str());

    if (!masksW.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << masksWFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + masksWFile);
    }

    std::vector<unsigned int> out;
    std::copy(std::istream_iterator<unsigned int>(masksW), 
              std::istream_iterator<unsigned int>(), 
              std::back_inserter(out));

    CudaTensor<unsigned int> mask = cuda_tensor_cast<unsigned int>(mMasksWeights[0]);
    for (unsigned int i = 0; i < mask.size(); ++i) {
        mask(i) = out[i];
    }
    mask.synchronizeHToD();

    mCurrentThreshold = (float)((mask.eq(0)).sum()) / mask.size();
}


}

namespace N2D2 {
    template class PruneQuantizerCell_Frame_CUDA<half_float::half>;
    template class PruneQuantizerCell_Frame_CUDA<float>;
    template class PruneQuantizerCell_Frame_CUDA<double>;
}

#endif