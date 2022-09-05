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

#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_Kernels.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"


// ----------------------------------------------------------------------------
// ---------------------------- STANDARD KERNELS ------------------------------
// ----------------------------------------------------------------------------

namespace N2D2 {
namespace SATQuantizer_Frame_Kernels {

template <class T>
void apply_tanh_transform(Tensor<T>& input, Tensor<T>& output)
{
    std::transform(input.begin(),
                   input.end(),
                   output.begin(),
                   [](T value) {return std::tanh(value);}
                   );
}

template <class T>
std::pair<T, T> get_minmax_element(Tensor<T>& data)
{
    std::pair<typename Tensor<T>::const_iterator, typename Tensor<T>::const_iterator> minMaxPair
        = std::minmax_element(data.begin(), data.end());

    return std::make_pair(*(minMaxPair.first), *(minMaxPair.second));
}

}
}

// ----------------------------------------------------------------------------
// ----------------------------- FORWARD KERNELS ------------------------------
// ----------------------------------------------------------------------------

// ---------------------- WEIGHT DEFAULT FORWARD KERNEL -----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_default_propagate(Tensor<T>& weights,
                                                                         Tensor<T>& weightsQ,
                                                                         float range,
                                                                         T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

    // Dorefa Quantization
#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {

        T q = T(0.5) * ((weightsQ(i) / *tanh_max_value) + T(1.0));

        // Check if q is between 0 and 1
        assert(q >= T(0.0) && q <= T(1.0));

        q = T(1.0f / range) * T(rintf(q * range));

        // Check if q is between 0 and 1
        assert(q >= T(0.0) && q <= T(1.0));

        weightsQ(i) = q * T(2.0) - T(1.0);
    }
}

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::no_quantize_weight_default_propagate(Tensor<T>& weights,
                                                                            Tensor<T>& weightsQ,
                                                                            float /*range*/,
                                                                            T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {
        weightsQ(i) = weightsQ(i) / *tanh_max_value;
    }
}  

// ----------------------------------------------------------------------------

// --------------------- WEIGHT FULLRANGE FORWARD KERNEL ----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_fullrange_propagate(Tensor<T>& weights,
                                                                           Tensor<T>& weightsQ,
                                                                           float range,
                                                                           T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

    // Dorefa Quantization
#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {

        T q = T(0.5) * ((weightsQ(i) / *tanh_max_value) + T(1.0));

        // Check if q is between 0 and 1
        assert(q >= T(0.0) && q <= T(1.0));

        q = T(1.0f + 0.9998f / range) * q - T(0.4999f / range);
        q = T(1.0f / range) * T(rintf(q * range));

        // Check if q is between 0 and 1
        assert(q >= T(0.0) && q <= T(1.0));

        weightsQ(i) = q * T(2.0) - T(1.0);
    }
}                                                                           

// ----------------------------------------------------------------------------

// ---------------------- WEIGHT SYMRANGE FORWARD KERNEL ----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_symrange_propagate(Tensor<T>& weights,
                                                                          Tensor<T>& weightsQ,
                                                                          float range,
                                                                          T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

    range = floor(range / 2);

    // Dorefa Quantization
#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {

        T q = weightsQ(i) / *tanh_max_value;

        // Check if q is between -1 and 1
        assert(q >= T(-1.0) && q <= T(1.0));

        q = T(1.0f / range) * T(rintf(q * range));

        // Check if q is between -1 and 1
        assert(q >= T(-1.0) && q <= T(1.0));

        weightsQ(i) = q;
    }
}                                                                          

// ----------------------------------------------------------------------------

// --------------------- WEIGHT ASYMRANGE FORWARD KERNEL ----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_asymrange_propagate(Tensor<T>& weights,
                                                                           Tensor<T>& weightsQ,
                                                                           float range,
                                                                           T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

    range = floor(range / 2);

    // Dorefa Quantization
#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {

        T q = weightsQ(i) / *tanh_max_value;

        // Check if q is between -1 and 1
        assert(q >= T(-1.0) && q <= T(1.0));

        q = T(1.0f + 1.0f/(2.0f * range)) * q - T(1.0f / (2.0f * range));
        q = T(1.0f / range) * T(rintf(q * range));

        // Check if q is between -1 - 1/range and 1
        assert(q >= T(-1.0 - 1.0/range) && q <= T(1.0));

        weightsQ(i) = q;
    }
}

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::no_quantize_weight_asymrange_propagate(Tensor<T>& weights,
                                                                              Tensor<T>& weightsQ,
                                                                              float range,
                                                                              T* tanh_max_value)
{
    apply_tanh_transform(weights, weightsQ);

    std::pair<T, T> minmax = get_minmax_element(weightsQ);
    *tanh_max_value = T(std::max(std::abs(minmax.first), std::abs(minmax.second)));

    range = floor(range / 2);

    // Dorefa Quantization
#pragma omp parallel for if (weightsQ.size() > 1024)
    for (unsigned int i = 0; i < weightsQ.size(); ++i) {
        T q = weightsQ(i) / *tanh_max_value;
        q = T(1.0f + 1.0f/(2.0f * range)) * q - T(1.0f / (2.0f * range));
        weightsQ(i) = q;
    }
}                                                                              

// ----------------------------------------------------------------------------

// ------------------------- SCALING FORWARD KERNEL ---------------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::apply_scaling(Tensor<T>& data,
                                                     T* scaling_value,
                                                     unsigned int scaling_factor)
{
    // Apply Scale Adjust Training method
    T sum = T(std::accumulate(data.begin(), data.end(), T(0.0)));
    T mean = T(sum / data.size());
    T var = T(0.0);
    std::for_each (std::begin(data), std::end(data), [&](const T d) {
        var += (d - mean) * (d - mean);
    });
    var /= (data.size() - T(1.0));
    *scaling_value = T(std::sqrt(var * scaling_factor));

#pragma omp parallel for if (data.size() > 1024)
    for (unsigned int i = 0; i < data.size(); ++i) {
        data(i) /= (*scaling_value);
    }
}                                                     

// ----------------------------------------------------------------------------

// ----------------------- ACTIVATION FORWARD KERNELS -------------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_activation_propagate(Tensor<T>& activations,
                                                                     const float range,
                                                                     Tensor<T>& Alpha,
                                                                     Tensor<T>& fpActivations,
                                                                     bool inference)
{
    const T alpha_ = T(abs(Alpha(0)));

    if(inference) {

#pragma omp parallel for if (activations.size() > 1024)
        for (unsigned int i = 0; i < activations.size(); ++i) {
            
            const T x = activations(i);
            const T x_clip = (x < T(0.0)) ? T(0.0) : (x < alpha_) ? x : alpha_;
            T q = x_clip / alpha_;

            // Test if q is in [0;1] before rounding
            assert(q >= T(0.0) && q <= T(1.0));

            q = round(T(range) * q) / T(range);

            // Test if q is in [0;1] after rounding
            assert(q >= T(0.0) && q <= T(1.0));

            activations(i) = q * alpha_;
        }

    } else {

#pragma omp parallel for if (activations.size() > 1024)
        for (unsigned int i = 0; i < activations.size(); ++i) {
            
            // Save full precision data value before quantization
            fpActivations(i) = activations(i);
            
            const T x = activations(i);
            const T x_clip = (x < T(0.0)) ? T(0.0) : (x < alpha_) ? x : alpha_;
            T q = x_clip / alpha_;

            // Test if q is in [0;1] before rounding
            assert(q >= T(0.0) && q <= T(1.0));

            q = round(T(range) * q) / T(range);

            // Test if q is in [0;1] after rounding
            assert(q >= T(0.0) && q <= T(1.0));

            activations(i) = q * alpha_;
        }

    }

}

// ----------------------------------------------------------------------------

// ------------------------- BIASES FORWARD KERNELS ---------------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_bias_propagate(Tensor<T>& biases,
                                                               Tensor<T>& biasesQ)
{
    // Simple copy for biases, no quantization for the moment
#pragma omp parallel for if (biases.size() > 1024)
    for (unsigned int i = 0; i < biases.size(); ++i) {
        biasesQ(i) = biases(i);
    }
}

// ----------------------------------------------------------------------------



// ----------------------------------------------------------------------------
// ---------------------------- BACKWARD KERNELS ------------------------------
// ----------------------------------------------------------------------------

// ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_default_back_propagate(Tensor<T>& diffWeightsQ,
                                                                              Tensor<T>& diffWeights,
                                                                              Tensor<T>& weights,
                                                                              T factor)
{
#pragma omp parallel for if (diffWeightsQ.size() > 1024)
    for (unsigned int i = 0; i < diffWeightsQ.size(); ++i) {
        T inv_cosh = T(1/std::cosh(weights(i)));
        T grad = inv_cosh * inv_cosh * T(1/factor);
        diffWeights(i) = diffWeightsQ(i) * grad;
    }
}

// ----------------------------------------------------------------------------

// -------------------- WEIGHT FULLRANGE BACKWARD KERNEL ----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_fullrange_back_propagate(Tensor<T>& diffWeightsQ,
                                                                                Tensor<T>& diffWeights,
                                                                                Tensor<T>& weights,
                                                                                float range,
                                                                                T factor)
{
    T fullrange_factor = T(1.0f + 0.9998f/range);

#pragma omp parallel for if (diffWeightsQ.size() > 1024)
    for (unsigned int i = 0; i < diffWeightsQ.size(); ++i) {
        T inv_cosh = T(1/std::cosh(weights(i)));
        T grad = inv_cosh * inv_cosh * T(1/factor) * fullrange_factor;
        diffWeights(i) = diffWeightsQ(i) * grad;
    }
}

// ----------------------------------------------------------------------------

// -------------------- WEIGHT ASYMRANGE BACKWARD KERNEL ----------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_weight_asymrange_back_propagate(Tensor<T>& diffWeightsQ,
                                                                                Tensor<T>& diffWeights,
                                                                                Tensor<T>& weights,
                                                                                float range,
                                                                                T factor)
{
    range = floor(range / 2);
    T asymm_factor = T(1.0f + 1.0f/(2.0f * range));

#pragma omp parallel for if (diffWeightsQ.size() > 1024)
    for (unsigned int i = 0; i < diffWeightsQ.size(); ++i) {
        T inv_cosh = T(1/std::cosh(weights(i)));
        T grad = inv_cosh * inv_cosh * T(1/factor) * asymm_factor;
        diffWeights(i) = diffWeightsQ(i) * grad;
    }
}

// ----------------------------------------------------------------------------

// ----------------------- ACTIVATION BACKWARD KERNELS ------------------------

template <class T>  
void N2D2::SATQuantizer_Frame_Kernels::quantize_activation_back_propagate(const Tensor<T>& diffInput,
                                                                          Tensor<T>& diffOutput,
                                                                          Tensor<T>& diffAlpha,
                                                                          const Tensor<T>& fpActivations,
                                                                          const float range,
                                                                          const Tensor<T>& Alpha)
{
    const T alpha_ = T(abs(Alpha(0)));

#pragma omp parallel for if (diffInput.size() > 1024)
    for (unsigned int i = 0; i < diffInput.size(); ++i) {

        const T x = fpActivations(i);

        // --- Alpha gradient computation ---

        const T x_clip = (x < T(0.0)) ? T(0.0) : (x < alpha_) ? x : alpha_;
        const T q = x_clip / alpha_;

        // Test if q is in [0;1] before rounding
        assert(q >= T(0.0) && q <= T(1.0));

        T qData = round(T(range) * q) / T(range);

        // Test if qData is in [0;1] after rounding
        assert(qData >= T(0.0) && qData <= T(1.0));

        const T dQAlpha = (x >= alpha_) ? T(1.0) : (qData - q);
        diffAlpha(i) = dQAlpha * diffInput(i);


        // --- Activation gradient computation ---

        // STE
        const T dQAct = (x <= T(0.0)) ? T(0.0) : (x > alpha_) ? T(0.0) : T(1.0);
        diffOutput(i) = dQAct * diffInput(i);
    }
}

// ----------------------------------------------------------------------------


// ------------------------- BIASES BACKWARD KERNELS --------------------------

template <class T>
void N2D2::SATQuantizer_Frame_Kernels::quantize_biases_back_propagate(Tensor<T>& diffBiasesQ,
                                                                      Tensor<T>& diffBiases)
{
#pragma omp parallel for if (diffBiasesQ.size() > 1024)
    for (unsigned int i = 0; i < diffBiasesQ.size(); ++i) {
        diffBiases(i) = diffBiasesQ(i);
    } 
}

// ----------------------------------------------------------------------------


namespace N2D2 {

    // ----------------------------------------------------------------------------
    // ----------------------------- SPECIALIZATIONS ------------------------------
    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT DEFAULT FORWARD KERNEL -----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                  Tensor<half_float::half>& weightsQ,
                                                                                                  float range,
                                                                                                  half_float::half* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_propagate<float>(Tensor<float>& weights,
                                                                                       Tensor<float>& weightsQ,
                                                                                       float range,
                                                                                       float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_propagate<double>(Tensor<double>& weights,
                                                                                        Tensor<double>& weightsQ,
                                                                                        float range,
                                                                                        double* tanh_max_value);



    template void SATQuantizer_Frame_Kernels::no_quantize_weight_default_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                     Tensor<half_float::half>& weightsQ,
                                                                                                     float range,
                                                                                                     half_float::half* tanh_max_value); 

    template void SATQuantizer_Frame_Kernels::no_quantize_weight_default_propagate<float>(Tensor<float>& weights,
                                                                                          Tensor<float>& weightsQ,
                                                                                          float range,
                                                                                          float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::no_quantize_weight_default_propagate<double>(Tensor<double>& weights,
                                                                                           Tensor<double>& weightsQ,
                                                                                           float range,
                                                                                           double* tanh_max_value);                               

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT FULLRANGE FORWARD KERNEL ----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                    Tensor<half_float::half>& weightsQ,
                                                                                                    float range,
                                                                                                    half_float::half* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_propagate<float>(Tensor<float>& weights,
                                                                                         Tensor<float>& weightsQ,
                                                                                         float range,
                                                                                         float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_propagate<double>(Tensor<double>& weights,
                                                                                          Tensor<double>& weightsQ,
                                                                                          float range,
                                                                                          double* tanh_max_value);

    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT SYMRANGE FORWARD KERNEL ----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_symrange_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                   Tensor<half_float::half>& weightsQ,
                                                                                                   float range,
                                                                                                   half_float::half* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_symrange_propagate<float>(Tensor<float>& weights,
                                                                                        Tensor<float>& weightsQ,
                                                                                        float range,
                                                                                        float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_symrange_propagate<double>(Tensor<double>& weights,
                                                                                         Tensor<double>& weightsQ,
                                                                                         float range,
                                                                                         double* tanh_max_value);

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT ASYMRANGE FORWARD KERNEL ----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                    Tensor<half_float::half>& weightsQ,
                                                                                                    float range,
                                                                                                    half_float::half* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_propagate<float>(Tensor<float>& weights,
                                                                                         Tensor<float>& weightsQ,
                                                                                         float range,
                                                                                         float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_propagate<double>(Tensor<double>& weights,
                                                                                          Tensor<double>& weightsQ,
                                                                                          float range,
                                                                                          double* tanh_max_value);



    template void SATQuantizer_Frame_Kernels::no_quantize_weight_asymrange_propagate<half_float::half>(Tensor<half_float::half>& weights,
                                                                                                       Tensor<half_float::half>& weightsQ,
                                                                                                       float range,
                                                                                                       half_float::half* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::no_quantize_weight_asymrange_propagate<float>(Tensor<float>& weights,
                                                                                            Tensor<float>& weightsQ,
                                                                                            float range,
                                                                                            float* tanh_max_value);

    template void SATQuantizer_Frame_Kernels::no_quantize_weight_asymrange_propagate<double>(Tensor<double>& weights,
                                                                                             Tensor<double>& weightsQ,
                                                                                             float range,
                                                                                             double* tanh_max_value);

    // ----------------------------------------------------------------------------

    // ------------------------- SCALING FORWARD KERNEL ---------------------------

    template void SATQuantizer_Frame_Kernels::apply_scaling<half_float::half>(Tensor<half_float::half>& data,
                                                                              half_float::half* scaling_value,
                                                                              unsigned int scaling_factor);

    template void SATQuantizer_Frame_Kernels::apply_scaling<float>(Tensor<float>& data,
                                                                   float* scaling_value,
                                                                   unsigned int scaling_factor);

    template void SATQuantizer_Frame_Kernels::apply_scaling<double>(Tensor<double>& data,
                                                                    double* scaling_value,
                                                                    unsigned int scaling_factor);

    // ----------------------------------------------------------------------------

    // ----------------------- ACTIVATION FORWARD KERNELS -------------------------

    template void SATQuantizer_Frame_Kernels::quantize_activation_propagate<half_float::half>(Tensor<half_float::half>& activations,
                                                                                              const float range,
                                                                                              Tensor<half_float::half>& Alpha,
                                                                                              Tensor<half_float::half>& fpActivations,
                                                                                              bool inference);

    template void SATQuantizer_Frame_Kernels::quantize_activation_propagate<float>(Tensor<float>& activations,
                                                                                   const float range,
                                                                                   Tensor<float>& Alpha,
                                                                                   Tensor<float>& fpActivations,
                                                                                   bool inference);

    template void SATQuantizer_Frame_Kernels::quantize_activation_propagate<double>(Tensor<double>& activations,
                                                                                    const float range,
                                                                                    Tensor<double>& Alpha,
                                                                                    Tensor<double>& fpActivations,
                                                                                    bool inference);

    // ----------------------------------------------------------------------------

    // ------------------------- BIASES FORWARD KERNELS ---------------------------
    
    template void SATQuantizer_Frame_Kernels::quantize_bias_propagate<half_float::half>(Tensor<half_float::half>& biases, 
                                                                                        Tensor<half_float::half>& biasesQ);

    template void SATQuantizer_Frame_Kernels::quantize_bias_propagate<float>(Tensor<float>& biases, Tensor<float>& biasesQ);

    template void SATQuantizer_Frame_Kernels::quantize_bias_propagate<double>(Tensor<double>& biases, Tensor<double>& biasesQ);

    // ----------------------------------------------------------------------------


    // ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_back_propagate<half_float::half>(Tensor<half_float::half>& diffWeightsQ,
                                                                                                       Tensor<half_float::half>& diffWeights,
                                                                                                       Tensor<half_float::half>& weights,
                                                                                                       half_float::half factor);

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_back_propagate<float>(Tensor<float>& diffWeightsQ,
                                                                                            Tensor<float>& diffWeights,
                                                                                            Tensor<float>& weights,
                                                                                            float factor);

    template void SATQuantizer_Frame_Kernels::quantize_weight_default_back_propagate<double>(Tensor<double>& diffWeightsQ,
                                                                                             Tensor<double>& diffWeights,
                                                                                             Tensor<double>& weights,
                                                                                             double factor);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT FULLRANGE BACKWARD KERNEL ----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_back_propagate<half_float::half>(Tensor<half_float::half>& diffWeightsQ,
                                                                                                         Tensor<half_float::half>& diffWeights,
                                                                                                         Tensor<half_float::half>& weights,
                                                                                                         float range,
                                                                                                         half_float::half factor);

    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_back_propagate<float>(Tensor<float>& diffWeightsQ,
                                                                                              Tensor<float>& diffWeights,
                                                                                              Tensor<float>& weights,
                                                                                              float range,
                                                                                              float factor);
    
    template void SATQuantizer_Frame_Kernels::quantize_weight_fullrange_back_propagate<double>(Tensor<double>& diffWeightsQ,
                                                                                               Tensor<double>& diffWeights,
                                                                                               Tensor<double>& weights,
                                                                                               float range,
                                                                                               double factor);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT ASYMRANGE BACKWARD KERNEL ----------------------

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_back_propagate<half_float::half>(Tensor<half_float::half>& diffWeightsQ,
                                                                                                         Tensor<half_float::half>& diffWeights,
                                                                                                         Tensor<half_float::half>& weights,
                                                                                                         float range,
                                                                                                         half_float::half factor);

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_back_propagate<float>(Tensor<float>& diffWeightsQ,
                                                                                              Tensor<float>& diffWeights,
                                                                                              Tensor<float>& weights,
                                                                                              float range,
                                                                                              float factor);

    template void SATQuantizer_Frame_Kernels::quantize_weight_asymrange_back_propagate<double>(Tensor<double>& diffWeightsQ,
                                                                                               Tensor<double>& diffWeights,
                                                                                               Tensor<double>& weights,
                                                                                               float range,
                                                                                               double factor);

    // ----------------------------------------------------------------------------

    // ----------------------- ACTIVATION BACKWARD KERNELS ------------------------

    template void SATQuantizer_Frame_Kernels::quantize_activation_back_propagate<half_float::half>(const Tensor<half_float::half>& diffInput,
                                                                                                   Tensor<half_float::half>& diffOutput,
                                                                                                   Tensor<half_float::half>& diffAlpha,
                                                                                                   const Tensor<half_float::half>& fpActivations,
                                                                                                   const float range,
                                                                                                   const Tensor<half_float::half>& Alpha);

    template void SATQuantizer_Frame_Kernels::quantize_activation_back_propagate<float>(const Tensor<float>& diffInput,
                                                                                        Tensor<float>& diffOutput,
                                                                                        Tensor<float>& diffAlpha,
                                                                                        const Tensor<float>& fpActivations,
                                                                                        const float range,
                                                                                        const Tensor<float>& Alpha);

    template void SATQuantizer_Frame_Kernels::quantize_activation_back_propagate<double>(const Tensor<double>& diffInput,
                                                                                         Tensor<double>& diffOutput,
                                                                                         Tensor<double>& diffAlpha,
                                                                                         const Tensor<double>& fpActivations,
                                                                                         const float range,
                                                                                         const Tensor<double>& Alpha);

    // ----------------------------------------------------------------------------

    // ------------------------- BIASES BACKWARD KERNELS --------------------------

    template void SATQuantizer_Frame_Kernels::quantize_biases_back_propagate<half_float::half>(Tensor<half_float::half>& diffBiasesQ, 
                                                                                               Tensor<half_float::half>& diffBiases);

    template void SATQuantizer_Frame_Kernels::quantize_biases_back_propagate<float>(Tensor<float>& diffBiasesQ, Tensor<float>& diffBiases);

    template void SATQuantizer_Frame_Kernels::quantize_biases_back_propagate<double>(Tensor<double>& diffBiasesQ, Tensor<double>& diffBiases);

    // ----------------------------------------------------------------------------

}
