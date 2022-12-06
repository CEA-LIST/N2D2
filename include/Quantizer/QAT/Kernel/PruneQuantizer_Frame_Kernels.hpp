/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#ifndef N2D2_PRUNEQUANTIZER_FRAME_KERNELS_H
#define N2D2_PRUNEQUANTIZER_FRAME_KERNELS_H

#include "containers/Tensor.hpp"

namespace N2D2 {

namespace PruneQuantizer_Frame_Kernels {

    template <class T>
    void apply_pruning_with_masks(Tensor<T>& data,
                                  Tensor<T>& dataPruned,
                                  Tensor<unsigned int>& masks);

    void update_masks_random(Tensor<unsigned int>& masks,
                             unsigned int& nbzero,
                             unsigned int nbzeromax);

}

}



#endif  // N2D2_PRUNEQUANTIZER_FRAME_KERNELS_H