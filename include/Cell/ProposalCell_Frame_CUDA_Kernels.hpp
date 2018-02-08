/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_PROPOSAL_FRAME_CUDA_KERNELS_H
#define N2D2_PROPOSAL_FRAME_CUDA_KERNELS_H

#include <stdexcept>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"

namespace N2D2 {
                                            

void cudaSToOutputROIs( const unsigned int nbProposals,
                        const unsigned int scoreIdx,
                        const unsigned int nbCls,
                        const unsigned int nbOutputs,
                        unsigned int maxParts,
                        unsigned int maxTemplates,
                        bool generateParts,
                        bool generateTemplates,
                        const int* numPartsPerClass,
                        const int* numTemplatesPerClass,
                        const int* maxCls,
                        const float* ROIEst,
                        const int* predictionIndex,
                        const float* partsPrediction,
                        const float* templatesPrediction,
                        float* outputs,
                        const dim3 threadsPerBlock,
                        const dim3 blocksPerGrid);

void cudaSNormalizeROIs( unsigned int inputSizeX,
                        unsigned int inputSizeY,
                        unsigned int nbProposals,
                        unsigned int batchSize,
                        unsigned int scoreIdx,
                        unsigned int nbCls,
                        unsigned int maxParts,
                        unsigned int maxTemplates,
                        bool keepMax,
                        bool generateParts,
                        bool generateTemplates,
                        const float normX,
                        const float normY,
                        const float* means,
                        const float* std,
                        const int* numPartsPerClass,
                        const int* numTemplatesPerClass,
                        const float* ROIRef,
                        const float* ROIEst,
                        const float* ValueEst,
                        const float* partsEst,
                        const float* partsVisibilityEst,
                        const float* templatesEst,
                        float* outputs,
                        int* maxCls,
                        float* partsPrediction,
                        float* templatesPrediction,
                        float scoreThreshold,
                        const dim3 threadsPerBlock,
                        const dim3 blocksPerGrid);

                            
}

#endif // N2D2_PROPOSAL_FRAME_CUDA_KERNELS_H

