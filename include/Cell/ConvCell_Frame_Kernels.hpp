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

#ifndef N2D2_CONVCELL_FRAME_KERNELS_H
#define N2D2_CONVCELL_FRAME_KERNELS_H

#include "Cell_Frame.hpp"

namespace N2D2 {
namespace ConvCell_Frame_Kernels {
    struct Descriptor {
        std::vector<unsigned int> subSample;
        std::vector<unsigned int> stride;
        std::vector<int> padding;

        Descriptor(const std::vector<unsigned int>& subSample_,
                   const std::vector<unsigned int>& stride_,
                   const std::vector<int>& padding_)
            : subSample(subSample_),
              stride(stride_),
              padding(padding_)
        {
        }
    };

    // Forward
    void forward(const Float_T* alpha,
                 const Tensor<Float_T>& inputs,
                 const Tensor<Float_T>& sharedSynapses,
                 const Descriptor& desc,
                 const Float_T* beta,
                 Tensor<Float_T>& outputs,
                 const Tensor<bool>& maps = Tensor<bool>());
    void forwardBias(const Float_T* alpha,
                     const Tensor<Float_T>& bias,
                     const Float_T* beta,
                     Tensor<Float_T>& outputs);

    // Backward
    void backwardData(const Float_T* alpha,
                      const Tensor<Float_T>& sharedSynapses,
                      const Tensor<Float_T>& diffInputs,
                      const Descriptor& desc,
                      const Float_T* beta,
                      Tensor<Float_T>& diffOutputs,
                      const Tensor<bool>& maps = Tensor<bool>());
    void backwardFilter(const Float_T* alpha,
                        const Tensor<Float_T>& inputs,
                        const Tensor<Float_T>& diffInputs,
                        const Descriptor& desc,
                        const Float_T* beta,
                        Tensor<Float_T>& diffSharedSynapses,
                        const Tensor<bool>& maps = Tensor<bool>());
    void backwardBias(const Float_T* alpha,
                      const Tensor<Float_T>& diffInputs,
                      const Float_T* beta,
                      Tensor<Float_T>& diffBias);
}
}

#endif // N2D2_CONVCELL_FRAME_KERNELS_H
