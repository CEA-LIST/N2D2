/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_PADDINGCELL_FRAME_KERNELS_H
#define N2D2_PADDINGCELL_FRAME_KERNELS_H

#include "Cell_Frame.hpp"

namespace N2D2 {
namespace PaddingCell_Frame_Kernels {
    struct Descriptor {
        int leftPad;
        int rightPad;
        int topPad;
        int botPad;

        Descriptor(int leftPad_,
                   int rightPad_,
                   int topPad_,
                   int botPad_)
            : leftPad(leftPad_),
              rightPad(rightPad_),
              topPad(topPad_),
              botPad(botPad_)
        {
        }
    };

    // Forward
    void forward(const Tensor4d<Float_T>& inputs,
                 const Descriptor& desc,
                 const unsigned int nbChannels,
                 const unsigned int inputOffset,
                 const unsigned int outputOffset,
                 Tensor4d<Float_T>& outputs);

}
}

#endif // N2D2_PADDINGCELL_FRAME_KERNELS_H
