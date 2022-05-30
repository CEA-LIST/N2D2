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
    // ---------------------------- BACKWARD KERNELS ------------------------------
    // ----------------------------------------------------------------------------

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

}

}

#endif  // N2D2_SATQUANTIZER_FRAME_CUDA_KERNELS_H