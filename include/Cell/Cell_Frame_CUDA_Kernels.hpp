/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_CELL_FRAME_CUDA_KERNELS_H
#define N2D2_CELL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "CudaUtils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
void cudaPopulateNbTargetOutputs(int* targets,
                                 unsigned int* nbTargetOutputs,
                                 unsigned int nbOutputs,
                                 unsigned int outputsHeight,
                                 unsigned int outputsWidth,
                                 unsigned int batchSize);

//Half
double cudaHSetOutputTargets(int* targets,
                             unsigned int* nbTargetOutputs,
                             half_float::half* lossMem,
                             half_float::half* outputs,
                             half_float::half* diffInputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             unsigned int batchSize,
                             half_float::half targetVal,
                             half_float::half defaultVal);

//Float
double cudaSSetOutputTargets(int* targets,
                             unsigned int* nbTargetOutputs,
                             float* lossMem,
                             float* outputs,
                             float* diffInputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             unsigned int batchSize,
                             float targetVal,
                             float defaultVal);
//Double
double cudaDSetOutputTargets(int* targets,
                             unsigned int* nbTargetOutputs,
                             double* lossMem,
                             double* outputs,
                             double* diffInputs,
                             unsigned int nbOutputs,
                             unsigned int outputsHeight,
                             unsigned int outputsWidth,
                             unsigned int batchSize,
                             double targetVal,
                             double defaultVal);
}

#endif // N2D2_CELL_FRAME_CUDA_KERNELS_H
