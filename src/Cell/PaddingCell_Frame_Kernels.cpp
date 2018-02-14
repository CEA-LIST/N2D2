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

#include "Cell/PaddingCell_Frame_Kernels.hpp"

void N2D2::PaddingCell_Frame_Kernels::forward(const Tensor4d<Float_T>& inputs,
                                           const Descriptor& desc,
                                           const unsigned int nbChannels,
                                           const unsigned int inputOffset,
                                           const unsigned int outputOffset,
                                           Tensor4d<Float_T>& outputs)
{
    const unsigned int size = inputs.dimB() * outputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (inputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)inputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < nbChannels; ++output) {
            for (unsigned int oy = 0; oy < outputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs.dimX(); ++ox) {
                    
                    float outputValue = 0.0;

                    int ix = (int) ox - desc.leftPad;
                    int iy = (int) oy - desc.topPad;

                    if( ix >= 0  && ix < (int) inputs.dimX()
                        && iy >= 0  && iy < (int) inputs.dimY())
                    {
                        outputValue = inputs(ix,
                                             iy, 
                                             output + inputOffset,
                                             batchPos);

                    }

#pragma omp critical
                    outputs(ox, 
                            oy, 
                            output + outputOffset, 
                            batchPos) = outputValue;

                }
            }
        }
    }
}