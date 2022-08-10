/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#ifdef CUDA

#include "Quantizer/QAT/Cell/SAT/SATQuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_CUDA_Kernels.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"

// To avoid long function names
namespace sat_cuda = N2D2::SATQuantizer_Frame_CUDA_Kernels;

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<double>());


namespace N2D2 {

template<class T>
SATQuantizerCell_Frame_CUDA<T>::SATQuantizerCell_Frame_CUDA()
    : SATQuantizerCell(),
    QuantizerCell_Frame_CUDA<T>()
{
    // ctor
}

template<class T>
void SATQuantizerCell_Frame_CUDA<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new CudaTensor<T>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new CudaTensor<T>(diffWeights.dims()));

    mSAT_tanh_max.push_back(new T(0.0f));
    mSAT_scaling.push_back(new T(0.0f));

    mOutputsSize += weights.dimB() * weights.dimY() * weights.dimX();
}

template<class T>
void SATQuantizerCell_Frame_CUDA<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<CudaBaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<CudaBaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());   
}

template<class T>
void SATQuantizerCell_Frame_CUDA<T>::initialize()
{
    std::cout << "      " << std::setprecision(8)
              << "Quantizer::SAT || "  
              << " Quantization[" << mApplyQuantization << "] || "
              << " Clamping[" << !mApplyQuantization << "] || " 
              << " AdjustedVariance[" << mApplyScaling << "]" 
              << std::endl;
    
    initializeQWeights();
    mInitialized = true;
}

template<>
void SATQuantizerCell_Frame_CUDA<half_float::half>::initializeQWeights()
{
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<half_float::half> > fullPrecWeights
            = cuda_device_tensor_cast<half_float::half>(mFullPrecisionWeights[k]);

        if (mApplyQuantization) {
            sat_cuda::cudaH_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              mQuantizedWeights[k].getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
        } else {
            sat_cuda::cudaH_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     mQuantizedWeights[k].getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
        }

        if (mApplyScaling) {

            // To calculate the scaling with partial sum (required to the half version)
            if(mDeviceWorkspace.empty()) {
                mDeviceWorkspace.resize({32, 1, 1, 1});
                mDeviceWorkspace.fill(half_float::half(0.0));
                mDeviceWorkspace.synchronizeHToD();
            }

            sat_cuda::cudaH_apply_scaling(mQuantizedWeights[k].getDevicePtr(),
                                          mSAT_scaling[k],
                                          mDeviceWorkspace.getDevicePtr(),
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());
        }
    }    
}

template<>
void SATQuantizerCell_Frame_CUDA<double>::initializeQWeights()
{
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<double> > fullPrecWeights
            = cuda_device_tensor_cast<double>(mFullPrecisionWeights[k]);

        if (mApplyQuantization) {
            sat_cuda::cudaD_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              mQuantizedWeights[k].getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
        } else {
            sat_cuda::cudaD_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     mQuantizedWeights[k].getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
        }

        if (mApplyScaling)
            sat_cuda::cudaD_apply_scaling(mQuantizedWeights[k].getDevicePtr(),
                                          mSAT_scaling[k],
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());

    }
}

template<>
void SATQuantizerCell_Frame_CUDA<float>::initializeQWeights()
{
    unsigned int mode = 0;
    if (mQuantMode == QuantizerCell::FullRange) {
        mode = mApplyQuantization ? 2 : 0;
    }
    else if (mQuantMode == QuantizerCell::Symmetric) {
        mode = mApplyQuantization ? 3 : 0;
    }
    else if (mQuantMode == QuantizerCell::Asymmetric) {
        mode = mApplyQuantization ? 4 : 5;
    }
    else { // mQuantMode == QuantizerCell::Default
        mode = mApplyQuantization ? 1 : 0;
    }

    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

        switch (mode) {
        case 0:
            sat_cuda::cudaF_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     mQuantizedWeights[k].getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
            break;

        case 1:
            sat_cuda::cudaF_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              mQuantizedWeights[k].getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
            break;

        case 2:
            sat_cuda::cudaF_quantize_weight_fullrange_propagate(fullPrecWeights->getDevicePtr(),
                                                                mQuantizedWeights[k].getDevicePtr(),
                                                                mRange,
                                                                mSAT_tanh_max[k],
                                                                mFullPrecisionWeights[k].size());
            break;

        case 3:
            sat_cuda::cudaF_quantize_weight_symrange_propagate(fullPrecWeights->getDevicePtr(),
                                                               mQuantizedWeights[k].getDevicePtr(),
                                                               mRange,
                                                               mSAT_tanh_max[k],
                                                               mFullPrecisionWeights[k].size());
            break;

        case 4:
            sat_cuda::cudaF_quantize_weight_asymrange_propagate(fullPrecWeights->getDevicePtr(),
                                                                mQuantizedWeights[k].getDevicePtr(),
                                                                mRange,
                                                                mSAT_tanh_max[k],
                                                                mFullPrecisionWeights[k].size());
            break;

        case 5:
            sat_cuda::cudaF_weight_asymrange_propagate(fullPrecWeights->getDevicePtr(),
                                                       mQuantizedWeights[k].getDevicePtr(),
                                                       mRange,
                                                       mSAT_tanh_max[k],
                                                       mFullPrecisionWeights[k].size());
            break;

        default:
            // Should never be here
            break;
        }

        if (mApplyScaling)
            sat_cuda::cudaF_apply_scaling(mQuantizedWeights[k].getDevicePtr(),
                                          mSAT_scaling[k],
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());
    }

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecBiases
            = cuda_device_tensor_cast<float>(cuda_tensor_cast<float>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(fullPrecBiases->getDevicePtr(),
                                                     mQuantizedBiases.getDevicePtr(),
                                                     mFullPrecisionBiases->size());
    }

    // //set SAT scaling (preparation for export)
    // if(mApplyScaling){
    //     setSATScaling((double)(*mSAT_scaling[k]));
    // }
    // else{
    //     setSATScaling(1.0);
    // }
}

template<>
void SATQuantizerCell_Frame_CUDA<half_float::half>::propagate()
{    
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {

        std::shared_ptr<CudaDeviceTensor<half_float::half> > fullPrecWeights
            = cuda_device_tensor_cast<half_float::half>(mFullPrecisionWeights[k]);
        std::shared_ptr<CudaDeviceTensor<half_float::half> > quantizedWeights
            = cuda_device_tensor_cast<half_float::half>(mQuantizedWeights[k]);

        if (mApplyQuantization) {
            sat_cuda::cudaH_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              quantizedWeights->getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
        } else {
            sat_cuda::cudaH_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     quantizedWeights->getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
        }

        if (mApplyScaling) {

            // To calculate the scaling with partial sum (required to the half version)
            if(mDeviceWorkspace.empty()) {
                mDeviceWorkspace.resize({32, 1, 1, 1});
                mDeviceWorkspace.fill(half_float::half(0.0));
                mDeviceWorkspace.synchronizeHToD();
            }

            sat_cuda::cudaH_apply_scaling(quantizedWeights->getDevicePtr(),
                                          mSAT_scaling[k],
                                          mDeviceWorkspace.getDevicePtr(),
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());
        }
    }

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<half_float::half> > fullPrecBiases
            = cuda_device_tensor_cast<half_float::half>(cuda_tensor_cast<half_float::half>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaH_copyData(fullPrecBiases->getDevicePtr(),
                                                     mQuantizedBiases.getDevicePtr(),
                                                     mFullPrecisionBiases->size());
    }
    
}

template<>
void SATQuantizerCell_Frame_CUDA<float>::propagate()
{
    unsigned int mode = 0;
    if (mQuantMode == QuantizerCell::FullRange) {
        mode = mApplyQuantization ? 2 : 0;
    }
    else if (mQuantMode == QuantizerCell::Symmetric) {
        mode = mApplyQuantization ? 3 : 0;
    }
    else if (mQuantMode == QuantizerCell::Asymmetric) {
        mode = mApplyQuantization ? 4 : 5;
    }
    else { // mQuantMode == QuantizerCell::Default
        mode = mApplyQuantization ? 1 : 0;
    }

    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);
        std::shared_ptr<CudaDeviceTensor<float> > quantizedWeights
            = cuda_device_tensor_cast<float>(mQuantizedWeights[k]);

        switch (mode) {
        case 0:
            sat_cuda::cudaF_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     quantizedWeights->getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
            break;

        case 1:
            sat_cuda::cudaF_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              quantizedWeights->getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
            break;

        case 2:
            sat_cuda::cudaF_quantize_weight_fullrange_propagate(fullPrecWeights->getDevicePtr(),
                                                                quantizedWeights->getDevicePtr(),
                                                                mRange,
                                                                mSAT_tanh_max[k],
                                                                mFullPrecisionWeights[k].size());
            break;

        case 3:
            sat_cuda::cudaF_quantize_weight_symrange_propagate(fullPrecWeights->getDevicePtr(),
                                                               quantizedWeights->getDevicePtr(),
                                                               mRange,
                                                               mSAT_tanh_max[k],
                                                               mFullPrecisionWeights[k].size());
            break;

        case 4:
            sat_cuda::cudaF_quantize_weight_asymrange_propagate(fullPrecWeights->getDevicePtr(),
                                                                quantizedWeights->getDevicePtr(),
                                                                mRange,
                                                                mSAT_tanh_max[k],
                                                                mFullPrecisionWeights[k].size());
            break;

        case 5:
            sat_cuda::cudaF_weight_asymrange_propagate(fullPrecWeights->getDevicePtr(),
                                                       quantizedWeights->getDevicePtr(),
                                                       mRange,
                                                       mSAT_tanh_max[k],
                                                       mFullPrecisionWeights[k].size());
            break;

        default:
            // Should never be here
            break;
        }

        if (mApplyScaling)
            sat_cuda::cudaF_apply_scaling(quantizedWeights->getDevicePtr(),
                                          mSAT_scaling[k],
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());
    }

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecBiases
            = cuda_device_tensor_cast<float>(cuda_tensor_cast<float>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(fullPrecBiases->getDevicePtr(),
                                                     mQuantizedBiases.getDevicePtr(),
                                                     mFullPrecisionBiases->size());
    }
}


template<>
void SATQuantizerCell_Frame_CUDA<double>::propagate()
{
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {

        std::shared_ptr<CudaDeviceTensor<double> > fullPrecWeights
            = cuda_device_tensor_cast<double>(mFullPrecisionWeights[k]);
        std::shared_ptr<CudaDeviceTensor<double> > quantizedWeights
            = cuda_device_tensor_cast<double>(mQuantizedWeights[k]);

        if (mApplyQuantization) {
            sat_cuda::cudaD_quantize_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                              quantizedWeights->getDevicePtr(),
                                                              mRange,
                                                              mSAT_tanh_max[k],
                                                              mFullPrecisionWeights[k].size());
        } else {
            sat_cuda::cudaD_weight_default_propagate(fullPrecWeights->getDevicePtr(),
                                                     quantizedWeights->getDevicePtr(),
                                                     mRange,
                                                     mSAT_tanh_max[k],
                                                     mFullPrecisionWeights[k].size());
        }

        if (mApplyScaling)
            sat_cuda::cudaD_apply_scaling(quantizedWeights->getDevicePtr(),
                                          mSAT_scaling[k],
                                          mOutputsSize,
                                          mQuantizedWeights[k].size());
    }

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<double> > fullPrecBiases
            = cuda_device_tensor_cast<double>(cuda_tensor_cast<double>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaD_copyData(fullPrecBiases->getDevicePtr(),
                                                     mQuantizedBiases.getDevicePtr(),
                                                     mFullPrecisionBiases->size());
    }
}



template<>
void SATQuantizerCell_Frame_CUDA<half_float::half>::back_propagate()
{
    
    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<half_float::half> > diffQuantizedWeights
            = cuda_device_tensor_cast<half_float::half>(mDiffQuantizedWeights[k]);
        std::shared_ptr<CudaDeviceTensor<half_float::half> > fullPrecWeights
            = cuda_device_tensor_cast<half_float::half>(mFullPrecisionWeights[k]);

        half_float::half scale = mApplyScaling ? *(mSAT_scaling[k]) : (half_float::half)1.0f;
        half_float::half factor = *(mSAT_tanh_max[k]) * scale;

        sat_cuda::cudaH_quantize_weight_default_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                               mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                               fullPrecWeights->getDevicePtr(),
                                                               factor,
                                                               mDiffQuantizedWeights[k].size());
    }

    if (mDiffQuantizedBiases) {

        std::shared_ptr<CudaDeviceTensor<half_float::half> > diffQuantBiases
            = cuda_device_tensor_cast<half_float::half>(cuda_tensor_cast<half_float::half>(*mDiffQuantizedBiases));

        Quantizer_Frame_CUDA_Kernels::cudaH_copyData(diffQuantBiases->getDevicePtr(),
                                                     mDiffFullPrecisionBiases.getDevicePtr(),
                                                     mDiffQuantizedBiases->size());
    }
    
}

template<>
void SATQuantizerCell_Frame_CUDA<float>::back_propagate()
{
    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {

        std::shared_ptr<CudaDeviceTensor<float> > diffQuantizedWeights
            = cuda_device_tensor_cast<float>(mDiffQuantizedWeights[k]);
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

        float scale = mApplyScaling ? *(mSAT_scaling[k]) : 1.0f;
        float factor = *(mSAT_tanh_max[k]) * scale;

        if (mQuantMode == QuantizerCell::Asymmetric) {
            sat_cuda::cudaF_quantize_weight_asymrange_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                                     mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                                     fullPrecWeights->getDevicePtr(),
                                                                     mRange,
                                                                     factor,
                                                                     mDiffQuantizedWeights[k].size());
        }
        else if (mQuantMode == QuantizerCell::FullRange) {
            sat_cuda::cudaF_quantize_weight_fullrange_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                                     mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                                     fullPrecWeights->getDevicePtr(),
                                                                     mRange,
                                                                     factor,
                                                                     mDiffQuantizedWeights[k].size());
        }
        else {
            sat_cuda::cudaF_quantize_weight_default_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                                   mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                                   fullPrecWeights->getDevicePtr(),
                                                                   factor,
                                                                   mDiffQuantizedWeights[k].size());
        }
    }

    if (mDiffQuantizedBiases) {
        std::shared_ptr<CudaDeviceTensor<float> > diffQuantBiases
            = cuda_device_tensor_cast<float>(cuda_tensor_cast<float>(*mDiffQuantizedBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(diffQuantBiases->getDevicePtr(),
                                                     mDiffFullPrecisionBiases.getDevicePtr(),
                                                     mDiffQuantizedBiases->size());
    }
}


template<>
void SATQuantizerCell_Frame_CUDA<double>::back_propagate()
{
    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<double> > diffQuantizedWeights
            = cuda_device_tensor_cast<double>(mDiffQuantizedWeights[k]);
        std::shared_ptr<CudaDeviceTensor<double> > fullPrecWeights
            = cuda_device_tensor_cast<double>(mFullPrecisionWeights[k]);

        double scale = mApplyScaling ? *(mSAT_scaling[k]) : 1.0;
        double factor = *(mSAT_tanh_max[k]) * scale;

        sat_cuda::cudaD_quantize_weight_default_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                               mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                               fullPrecWeights->getDevicePtr(),
                                                               factor,
                                                               mDiffQuantizedWeights[k].size());
    }

    if (mDiffQuantizedBiases) {

        std::shared_ptr<CudaDeviceTensor<double> > diffQuantBiases
            = cuda_device_tensor_cast<double>(cuda_tensor_cast<double>(*mDiffQuantizedBiases));

        Quantizer_Frame_CUDA_Kernels::cudaD_copyData(diffQuantBiases->getDevicePtr(),
                                                     mDiffFullPrecisionBiases.getDevicePtr(),
                                                     mDiffQuantizedBiases->size());
    }    
}


template<>
void SATQuantizerCell_Frame_CUDA<half_float::half>::update(unsigned int /*batchSize = 1*/)
{
    // Nothing to do
}

template<>
void SATQuantizerCell_Frame_CUDA<float>::update(unsigned int /*batchSize = 1*/)
{
    // Nothing to do
}
template<>
void SATQuantizerCell_Frame_CUDA<double>::update(unsigned int /*batchSize = 1*/)
{
    // Nothing to do
}


template <class T>
SATQuantizerCell_Frame_CUDA<T>::~SATQuantizerCell_Frame_CUDA()
{
}

template <class T>
void SATQuantizerCell_Frame_CUDA<T>::exportFreeParameters(const std::string& /*fileName*/) const 
{
    // Nothing to do
}

template <class T>
void SATQuantizerCell_Frame_CUDA<T>::importFreeParameters(const std::string
                                                     & /*fileName*/, bool /*ignoreNotExists*/)
{
    // Nothing to do
}

}


namespace N2D2 {
    template class SATQuantizerCell_Frame_CUDA<half_float::half>;
    template class SATQuantizerCell_Frame_CUDA<float>;
    template class SATQuantizerCell_Frame_CUDA<double>;
}

#endif
