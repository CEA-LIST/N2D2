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
#include "containers/Tensor.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"


// ----------------------------------------------------------------------------
// ----------------------------- FORWARD KERNELS ------------------------------
// ----------------------------------------------------------------------------


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
// ---------------------------- BACKWARD KERNELS ------------------------------
// ----------------------------------------------------------------------------


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
    for (int i = 0; i < (int)diffInput.size(); ++i) {

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


namespace N2D2 {

    // ----------------------------------------------------------------------------
    // ----------------------------- SPECIALIZATIONS ------------------------------
    // ----------------------------------------------------------------------------

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

}