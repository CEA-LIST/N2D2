/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_POOLCELL_FRAME_KERNELS_H
#define N2D2_POOLCELL_FRAME_KERNELS_H

#include "PoolCell_Frame_Kernels_struct.hpp"

namespace N2D2 {
namespace PoolCell_Frame_Kernels {
    // Forward
    template <class T>
    void forwardAverage(const T* alpha,
                        const Tensor<T>& inputs,
                        const Descriptor& desc,
                        const T* beta,
                        Tensor<T>& outputs,
                        bool isQuantized,
                        bool countIncludePadding = true,
                        const Tensor<bool>& maps = Tensor<bool>());

    template <class T>
    void forwardMax(const T* alpha,
                    const Tensor<T>& inputs,
                    const Descriptor& desc,
                    const T* beta,
                    Tensor<T>& outputs,
                    Tensor<ArgMax>& argMax,
                    bool useArgMax = false,
                    const Tensor<bool>& maps = Tensor<bool>());

    // Backward
    template <class T>
    void backwardAverage(const T* alpha,
                         const Tensor<T>& diffInputs,
                         const Descriptor& desc,
                         const T* beta,
                         Tensor<T>& diffOutputs,
                         bool countIncludePadding = true,
                         const Tensor<bool>& maps = Tensor<bool>());
    template <class T>
    void backwardMax(const T* alpha,
                     const Tensor<T>& diffInputs,
                     const Descriptor& desc,
                     const T* beta,
                     Tensor<T>& diffOutputs,
                     const Tensor<ArgMax>& argMax,
                     const Tensor<bool>& maps = Tensor<bool>());
}
}

#endif // N2D2_POOLCELL_FRAME_KERNELS_H
