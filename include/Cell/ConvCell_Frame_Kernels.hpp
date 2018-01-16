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
        unsigned int subSampleX;
        unsigned int subSampleY;
        unsigned int strideX;
        unsigned int strideY;
        int paddingX;
        int paddingY;

        Descriptor(unsigned int subSampleX_,
                   unsigned int subSampleY_,
                   unsigned int strideX_,
                   unsigned int strideY_,
                   int paddingX_,
                   int paddingY_)
            : subSampleX(subSampleX_),
              subSampleY(subSampleY_),
              strideX(strideX_),
              strideY(strideY_),
              paddingX(paddingX_),
              paddingY(paddingY_)
        {
        }
    };

    // Forward
    void forward(const Float_T* alpha,
                 const Tensor4d<Float_T>& inputs,
                 const Tensor4d<Float_T>& sharedSynapses,
                 const Descriptor& desc,
                 const Float_T* beta,
                 Tensor4d<Float_T>& outputs,
                 const Tensor2d<bool>& maps = Tensor2d<bool>());
    void forwardBias(const Float_T* alpha,
                     const Tensor4d<Float_T>& bias,
                     const Float_T* beta,
                     Tensor4d<Float_T>& outputs);

    // Backward
    void backwardData(const Float_T* alpha,
                      const Tensor4d<Float_T>& sharedSynapses,
                      const Tensor4d<Float_T>& diffInputs,
                      const Descriptor& desc,
                      const Float_T* beta,
                      Tensor4d<Float_T>& diffOutputs,
                      const Tensor2d<bool>& maps = Tensor2d<bool>());
    void backwardFilter(const Float_T* alpha,
                        const Tensor4d<Float_T>& inputs,
                        const Tensor4d<Float_T>& diffInputs,
                        const Descriptor& desc,
                        const Float_T* beta,
                        Tensor4d<Float_T>& diffSharedSynapses,
                        const Tensor2d<bool>& maps = Tensor2d<bool>());
    void backwardBias(const Float_T* alpha,
                      const Tensor4d<Float_T>& diffInputs,
                      const Float_T* beta,
                      Tensor4d<Float_T>& diffBias);
}
}

#endif // N2D2_CONVCELL_FRAME_KERNELS_H
