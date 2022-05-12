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

#ifndef N2D2_LSQQUANTIZER_FRAME_CUDA_KERNELS_H
#define N2D2_LSQQUANTIZER_FRAME_CUDA_KERNELS_H

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

namespace LSQQuantizer_Frame_CUDA_Kernels {

    // Function which quantizes the input data with the LSQ method
    void cudaF_quantize_propagate(const float* fullPrecData,
                                  const float* stepSize,
                                  const int rangeMin,
                                  const int rangeMax,
                                  float* quantData,
                                  float* fullPrecDataCopy,
                                  bool saveCopy,
                                  const size_t inputSize);

    // Function which calculates the step size gradient (according to the LSQ paper)
    // Also applies the STE and multiplies the gradient with the step size gradient scale
    void cudaF_quantize_back_propagate(float* diffQuantData,
                                       const float* fullPrecData, 
                                       float* diffFullPrecData,
                                       float* diffStepSize,     
                                       const int rangeMin,
                                       const int rangeMax,
                                       const float* stepSize, 
                                       const float gradFactor,
                                       const float beta,       
                                       const size_t inputSize);

}

}


#endif  // N2D2_LSQQUANTIZER_FRAME_CUDA_KERNELS_H