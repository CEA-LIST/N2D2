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

#ifndef N2D2_SATQUANTIZER_FRAME_CUDA_KERNELS_H
#define N2D2_SATQUANTIZER_FRAME_CUDA_KERNELS_H

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "CudaUtils.hpp"
#include "third_party/half.hpp"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


namespace N2D2 {

namespace SATQuantizer_Frame_CUDA_Kernels {

    // ----------------------------------------------------------------------------
    // ----------------------------- FORWARD KERNELS ------------------------------
    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT DEFAULT FORWARD KERNEL -----------------------

    void cudaH_quantize_weight_default_propagate(half_float::half* weights,
                                                 half_float::half* weightsQ,
                                                 float range,
                                                 half_float::half* tanh_max_value,
                                                 unsigned int size);

    void cudaF_quantize_weight_default_propagate(float* weights,
                                                 float* weightsQ,
                                                 float range,
                                                 float* tanh_max_value,
                                                 unsigned int size);

    void cudaD_quantize_weight_default_propagate(double* weights,
                                                 double* weightsQ,
                                                 float range,
                                                 double* tanh_max_value,
                                                 unsigned int size);


    void cudaH_weight_default_propagate(half_float::half* weights,
                                        half_float::half* weightsQ,
                                        float range,
                                        half_float::half* tanh_max_value,
                                        unsigned int size);

    void cudaF_weight_default_propagate(float* weights,
                                        float* weightsQ,
                                        float range,
                                        float* tanh_max_value,
                                        unsigned int size);

    void cudaD_weight_default_propagate(double* weights,
                                        double* weightsQ,
                                        float range,
                                        double* tanh_max_value,
                                        unsigned int size);

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT FULLRANGE FORWARD KERNEL ----------------------

    void cudaF_quantize_weight_fullrange_propagate(float* weights,
                                                   float* weightsQ,
                                                   float range,
                                                   float* tanh_max_value,
                                                   unsigned int size);

    // ----------------------------------------------------------------------------

    // ---------------------- WEIGHT SYMRANGE FORWARD KERNEL ----------------------

    void cudaF_quantize_weight_symrange_propagate(float* weights,
                                                  float* weightsQ,
                                                  float range,
                                                  float* tanh_max_value,
                                                  unsigned int size);

    // ----------------------------------------------------------------------------

    // --------------------- WEIGHT ASYMRANGE FORWARD KERNEL ----------------------

    void cudaF_quantize_weight_asymrange_propagate(float* weights,
                                                   float* weightsQ,
                                                   float range,
                                                   float* tanh_max_value,
                                                   unsigned int size);

    void cudaF_weight_asymrange_propagate(float* weights,
                                          float* weightsQ,
                                          float range,
                                          float* tanh_max_value,
                                          unsigned int size);

    // ----------------------------------------------------------------------------

    // ------------------------- SCALING FORWARD KERNELS --------------------------

    void cudaH_apply_scaling(half_float::half* data,
                             half_float::half* scaling_value,
                             half_float::half* partial_sum,
                             unsigned int scaling_factor,
                             unsigned int size);

    void cudaF_apply_scaling(float* data,
                             float* scaling_value,
                             unsigned int scaling_factor,
                             unsigned int size);

    void cudaD_apply_scaling(double* data,
                             double* scaling_value,
                             unsigned int scaling_factor,
                             unsigned int size);

    // ----------------------------------------------------------------------------


    // ----------------------- ACTIVATION FORWARD KERNELS -------------------------

    void cudaH_quantize_activation_propagate(half_float::half* activations,
                                             float range,
                                             const half_float::half* alpha,
                                             half_float::half* fpActivations,
                                             unsigned int size,
                                             bool inference);

    void cudaF_quantize_activation_propagate(float* activations,
                                             float range,
                                             const float* alpha,
                                             float* fpActivations,
                                             unsigned int size,
                                             bool inference);

    void cudaD_quantize_activation_propagate(double* activations,
                                             float range,
                                             const double* alpha,
                                             double* fpActivations,
                                             unsigned int size,
                                             bool inference);
    
    // ----------------------------------------------------------------------------



    // ----------------------------------------------------------------------------
    // ---------------------------- BACKWARD KERNELS ------------------------------
    // ----------------------------------------------------------------------------

    // ---------------- WEIGHT DEFAULT/SYMRANGE BACKWARD KERNEL -------------------

    void cudaH_quantize_weight_default_back_propagate(half_float::half* diffInputs,
                                                      half_float::half* diffOutputs,
                                                      half_float::half* fpWeights,
                                                      half_float::half factor,
                                                      unsigned int size);

    void cudaF_quantize_weight_default_back_propagate(float* diffInputs,
                                                      float* diffOutputs,
                                                      float* fpWeights,
                                                      float factor,
                                                      unsigned int size);

    void cudaD_quantize_weight_default_back_propagate(double* diffInputs,
                                                      double* diffOutputs,
                                                      double* fpWeights,
                                                      double factor,
                                                      unsigned int size);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT FULLRANGE BACKWARD KERNEL ----------------------

    void cudaF_quantize_weight_fullrange_back_propagate(float* diffInputs,
                                                        float* diffOutputs,
                                                        float* fpWeights,
                                                        float range,
                                                        float factor,
                                                        unsigned int size);

    // ----------------------------------------------------------------------------

    // -------------------- WEIGHT ASYMRANGE BACKWARD KERNEL ----------------------

    void cudaF_quantize_weight_asymrange_back_propagate(float* diffInputs,
                                                        float* diffOutputs,
                                                        float* fpWeights,
                                                        float range,
                                                        float factor,
                                                        unsigned int size);

    // ----------------------------------------------------------------------------

    // ----------------------- ACTIVATION BACKWARD KERNELS ------------------------

    void cudaH_quantize_activation_back_propagate(half_float::half* diffInput,
                                                  half_float::half* diffOutput,
                                                  half_float::half* diffAlpha,
                                                  const half_float::half* fpActivations,
                                                  float range,
                                                  const half_float::half* alpha,
                                                  unsigned int size);

    void cudaF_quantize_activation_back_propagate(float* diffInput,
                                                  float* diffOutput,
                                                  float* diffAlpha,
                                                  const float* fpActivations,
                                                  float range,
                                                  const float* alpha,
                                                  unsigned int size);

    void cudaD_quantize_activation_back_propagate(double* diffInput,
                                                  double* diffOutput,
                                                  double* diffAlpha,
                                                  const double* fpActivations,
                                                  float range,
                                                  const double* alpha,
                                                  unsigned int size);

    // ----------------------------------------------------------------------------

}

}

#endif  // N2D2_SATQUANTIZER_FRAME_CUDA_KERNELS_H