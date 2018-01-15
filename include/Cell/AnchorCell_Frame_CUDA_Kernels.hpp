/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
                    
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

#ifndef N2D2_ANCHORCELL_FRAME_CUDA_KERNELS_H
#define N2D2_ANCHORCELL_FRAME_CUDA_KERNELS_H

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "AnchorCell_Frame_Kernels_struct.hpp"

namespace N2D2 {
void cudaSAnchorPropagate(
    unsigned int stimuliSizeX,
    unsigned int stimuliSizeY,
    bool flip,
    bool inference,
    float* inputsCls,
    float* inputsCoord,
    unsigned int scoresCls,
    AnchorCell_Frame_Kernels::Anchor* anchors,
    AnchorCell_Frame_Kernels::BBox_T** gts,
    unsigned int* nbLabels,
    float* outputs,
    int* argMaxIoU,
    float* maxIoU,
    unsigned int nbAnchors,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    unsigned int nbTotalCls,
    unsigned int nbInputs);
}

#endif // N2D2_ANCHORCELL_FRAME_CUDA_KERNELS_H
