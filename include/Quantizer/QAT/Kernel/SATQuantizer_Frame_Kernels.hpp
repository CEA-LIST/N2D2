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

#ifndef N2D2_SATQUANTIZER_FRAME_KERNELS_H
#define N2D2_SATQUANTIZER_FRAME_KERNELS_H

#include <vector>
#include "containers/Tensor.hpp"

namespace N2D2 {

namespace SATQuantizer_Frame_Kernels {


    // ----------------------------------------------------------------------------
    // ----------------------------- FORWARD KERNELS ------------------------------
    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT DEFAULT FORWARD KERNEL -----------------------

    template <class T>
    void quantize_weight_default_propagate(Tensor<T>& weights,
                                           Tensor<T>& weightsQ,
                                           float range,
                                           T* tanh_max_value);

    template <class T>
    void no_quantize_weight_default_propagate(Tensor<T>& weights,
                                              Tensor<T>& weightsQ,
                                              float range,
                                              T* tanh_max_value);                                

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT FULLRANGE FORWARD KERNEL ----------------------

    template <class T>
    void quantize_weight_fullrange_propagate(Tensor<T>& weights,
                                             Tensor<T>& weightsQ,
                                             float range,
                                             T* tanh_max_value);

    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT SYMRANGE FORWARD KERNEL ----------------------

    template <class T>
    void quantize_weight_symrange_propagate(Tensor<T>& weights,
                                            Tensor<T>& weightsQ,
                                            float range,
                                            T* tanh_max_value);

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT ASYMRANGE FORWARD KERNEL ----------------------

    template <class T>
    void quantize_weight_asymrange_propagate(Tensor<T>& weights,
                                             Tensor<T>& weightsQ,
                                             float range,
                                             T* tanh_max_value);

    template <class T>
    void no_quantize_weight_asymrange_propagate(Tensor<T>& weights,
                                                Tensor<T>& weightsQ,
                                                float range,
                                                T* tanh_max_value);

    // ----------------------------------------------------------------------------

    // ------------------------- SCALING FORWARD KERNEL ---------------------------

    template <class T>
    void apply_scaling(Tensor<T>& data,
                       T* scaling_value,
                       unsigned int scaling_factor);

    // ----------------------------------------------------------------------------

    // ----------------------- ACTIVATION FORWARD KERNELS -------------------------

    template <class T>
    void quantize_activation_propagate(Tensor<T>& activations,
                                       const float range,
                                       Tensor<T>& Alpha,
                                       Tensor<T>& fpActivations,
                                       bool inference);

    // ----------------------------------------------------------------------------

    // ------------------------- BIASES FORWARD KERNELS ---------------------------

    template <class T>
    void quantize_bias_propagate(Tensor<T>& biases,
                                 Tensor<T>& biasesQ);

    // ----------------------------------------------------------------------------


    // ----------------------------------------------------------------------------
    // ---------------------------- BACKWARD KERNELS ------------------------------
    // ----------------------------------------------------------------------------

    // ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------

    template <class T>
    void quantize_weight_default_back_propagate(Tensor<T>& diffWeightsQ,
                                                Tensor<T>& diffWeights,
                                                Tensor<T>& weights,
                                                T factor);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT FULLRANGE BACKWARD KERNEL ----------------------

    template <class T>
    void quantize_weight_fullrange_back_propagate(Tensor<T>& diffWeightsQ,
                                                  Tensor<T>& diffWeights,
                                                  Tensor<T>& weights,
                                                  float range,
                                                  T factor);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT ASYMRANGE BACKWARD KERNEL ----------------------

    template <class T>
    void quantize_weight_asymrange_back_propagate(Tensor<T>& diffWeightsQ,
                                                  Tensor<T>& diffWeights,
                                                  Tensor<T>& weights,
                                                  float range,
                                                  T factor);

    // ----------------------------------------------------------------------------

    // ----------------------- ACTIVATION BACKWARD KERNELS ------------------------

    template <class T>
    void quantize_activation_back_propagate(const Tensor<T>& diffInput,
                                            Tensor<T>& diffOutput,
                                            Tensor<T>& diffAlpha,
                                            const Tensor<T>& fpActivations,
                                            const float range,
                                            const Tensor<T>& Alpha);

    // ----------------------------------------------------------------------------

    // ------------------------- BIASES BACKWARD KERNELS --------------------------

    template <class T>
    void quantize_biases_back_propagate(Tensor<T>& diffBiasesQ,
                                        Tensor<T>& diffBiases);

    // ----------------------------------------------------------------------------

}

}



#endif  // N2D2_SATQUANTIZER_FRAME_KERNELS_H