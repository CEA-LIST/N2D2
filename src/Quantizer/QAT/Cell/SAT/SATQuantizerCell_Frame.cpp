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

#include "Quantizer/QAT/Cell/SAT/SATQuantizerCell_Frame.hpp"
#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_Kernels.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"

// To avoid long function names
namespace sat_cpu = N2D2::SATQuantizer_Frame_Kernels;

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerCell_Frame<float>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::SATQuantizerCell>
N2D2::SATQuantizerCell_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerCell_Frame<double>::create,
    N2D2::Registrar<N2D2::SATQuantizerCell>::Type<double>());

template<class T>
N2D2::SATQuantizerCell_Frame<T>::SATQuantizerCell_Frame()
    : SATQuantizerCell(),
    QuantizerCell_Frame<T>()
{
    // ctor
}

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new Tensor<T>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new Tensor<T>(diffWeights.dims()));

    mSAT_tanh_max.push_back(new T(0.0f));
    mSAT_scaling.push_back(new T(0.0f));

    mOutputsSize += weights.dimB()*weights.dimY()*weights.dimX();
}


template<class T>
void N2D2::SATQuantizerCell_Frame<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<BaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<BaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::initialize()
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

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::initializeQWeights()
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
        Tensor<T> fpW = tensor_cast<T>(mFullPrecisionWeights[k]);

        switch (mode) {
        case 0:
            sat_cpu::no_quantize_weight_default_propagate(fpW,
                                                          mQuantizedWeights[k],
                                                          mRange,
                                                          mSAT_tanh_max[k]);
            break;

        case 1:
            sat_cpu::quantize_weight_default_propagate(fpW,
                                                       mQuantizedWeights[k],
                                                       mRange,
                                                       mSAT_tanh_max[k]);
            break;

        case 2:
            sat_cpu::quantize_weight_fullrange_propagate(fpW,
                                                         mQuantizedWeights[k],
                                                         mRange,
                                                         mSAT_tanh_max[k]);
            break;

        case 3:
            sat_cpu::quantize_weight_symrange_propagate(fpW,
                                                        mQuantizedWeights[k],
                                                        mRange,
                                                        mSAT_tanh_max[k]);
            break;

        case 4:
            sat_cpu::quantize_weight_asymrange_propagate(fpW,
                                                         mQuantizedWeights[k],
                                                         mRange,
                                                         mSAT_tanh_max[k]);
            break;

        case 5:
            sat_cpu::no_quantize_weight_asymrange_propagate(fpW,
                                                            mQuantizedWeights[k],
                                                            mRange,
                                                            mSAT_tanh_max[k]);
            break;

        default:
            // Should never be here
            break;
        }

        if (mApplyScaling)
                sat_cpu::apply_scaling(mQuantizedWeights[k],
                                       mSAT_scaling[k],
                                       mOutputsSize);
    }

    if (mFullPrecisionBiases) {
        Tensor<T> fpBiases = tensor_cast<T>(*mFullPrecisionBiases);
        Tensor<T> quantBiases = tensor_cast<T>(mQuantizedBiases);

        sat_cpu::quantize_bias_propagate(fpBiases, quantBiases);
    }

    // //set SAT scaling (preparation for export)
    // if(mApplyScaling){
    //     setSATScaling((double)(*mSAT_scaling[k]));
    // }
    // else{
    //     setSATScaling(1.0);
    // }
}

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::propagate()
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
        Tensor<T> fpW = tensor_cast<T>(mFullPrecisionWeights[k]);
        Tensor<T> qW = tensor_cast<T>(mQuantizedWeights[k]);

        switch (mode) {
        case 0:
            sat_cpu::no_quantize_weight_default_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        case 1:
            sat_cpu::quantize_weight_default_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        case 2:
            sat_cpu::quantize_weight_fullrange_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        case 3:
            sat_cpu::quantize_weight_symrange_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        case 4:
            sat_cpu::quantize_weight_asymrange_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        case 5:
            sat_cpu::no_quantize_weight_asymrange_propagate(fpW, qW, mRange, mSAT_tanh_max[k]);
            break;

        default:
            // Should never be here
            break;
        }

        if (mApplyScaling)
                sat_cpu::apply_scaling(qW, mSAT_scaling[k], mOutputsSize);
    }

    if (mFullPrecisionBiases) {
        Tensor<T> fpBiases = tensor_cast<T>(*mFullPrecisionBiases);
        Tensor<T> quantBiases = tensor_cast<T>(mQuantizedBiases);

        sat_cpu::quantize_bias_propagate(fpBiases, quantBiases);
    }

    // //set SAT scaling (preparation for export)
    // if(mApplyScaling){
    //     setSATScaling((double)(*mSAT_scaling[k]));
    // }
    // else{
    //     setSATScaling(1.0);
    // }
}

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::back_propagate()
{
    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
        Tensor<T> diffQuantWeights = tensor_cast<T>(mDiffQuantizedWeights[k]);
        Tensor<T> fullPrecisionWeights = tensor_cast<T>(mFullPrecisionWeights[k]);
        Tensor<T> diffFullPrecisionWeights = tensor_cast<T>(mDiffFullPrecisionWeights[k]);

        T scale = mApplyScaling ? *(mSAT_scaling[k]) : T(1.0);
        T factor = *(mSAT_tanh_max[k]) * scale;

        if (mQuantMode == QuantizerCell::Asymmetric) {
            sat_cpu::quantize_weight_asymrange_back_propagate(diffQuantWeights,
                                                              diffFullPrecisionWeights,
                                                              fullPrecisionWeights,
                                                              mRange,
                                                              factor);
        }
        else if (mQuantMode == QuantizerCell::FullRange) {
            sat_cpu::quantize_weight_fullrange_back_propagate(diffQuantWeights,
                                                              diffFullPrecisionWeights,
                                                              fullPrecisionWeights,
                                                              mRange,
                                                              factor);
        }
        else {
            sat_cpu::quantize_weight_default_back_propagate(diffQuantWeights,
                                                            diffFullPrecisionWeights,
                                                            fullPrecisionWeights,
                                                            factor);
        }
    }

    if (mDiffQuantizedBiases) {
        Tensor<T> diffQuantBiases = tensor_cast<T>(*mDiffQuantizedBiases);
        Tensor<T> diffFullPrecisionBiases = tensor_cast<T>(mDiffFullPrecisionBiases);

        sat_cpu::quantize_biases_back_propagate(diffQuantBiases, diffFullPrecisionBiases);
    }

}

template<class T>
void N2D2::SATQuantizerCell_Frame<T>::update(unsigned int /*batchSize = 1*/)
{
    // Nothing to do for SAT method in Cell mode    
}


template <class T>
N2D2::SATQuantizerCell_Frame<T>::~SATQuantizerCell_Frame()
{
    // dtor
}

template <class T>
void N2D2::SATQuantizerCell_Frame<T>::exportFreeParameters(const std::string& /*fileName*/) const 
{
    // Nothing to do for SAT method in Cell mode    
}

template <class T>
void N2D2::SATQuantizerCell_Frame<T>::importFreeParameters(const std::string
                                                     & /*fileName*/, bool /*ignoreNotExists*/)
{
    // Nothing to do for SAT method in Cell mode    
}


namespace N2D2 {
    template class SATQuantizerCell_Frame<half_float::half>;
    template class SATQuantizerCell_Frame<float>;
    template class SATQuantizerCell_Frame<double>;
}
