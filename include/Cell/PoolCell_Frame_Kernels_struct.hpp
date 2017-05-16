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

#ifndef N2D2_POOLCELL_FRAME_KERNELS_STRUCT_H
#define N2D2_POOLCELL_FRAME_KERNELS_STRUCT_H

namespace N2D2 {
namespace PoolCell_Frame_Kernels {
    struct Descriptor {
        unsigned int poolWidth;
        unsigned int poolHeight;
        unsigned int strideX;
        unsigned int strideY;
        int paddingX;
        int paddingY;

        Descriptor(unsigned int poolWidth_,
                   unsigned int poolHeight_,
                   unsigned int strideX_,
                   unsigned int strideY_,
                   int paddingX_,
                   int paddingY_)
            : poolWidth(poolWidth_),
              poolHeight(poolHeight_),
              strideX(strideX_),
              strideY(strideY_),
              paddingX(paddingX_),
              paddingY(paddingY_)
        {
        }
    };

    struct ArgMax {
        unsigned int ix;
        unsigned int iy;
        unsigned int channel;
        bool valid;

        ArgMax(unsigned int ix_ = 0,
               unsigned int iy_ = 0,
               unsigned int channel_ = 0,
               bool valid_ = false)
            : ix(ix_),
              iy(iy_),
              channel(channel_),
              valid(valid_)
        {
        }
    };

    inline bool operator==(const ArgMax& lhs, const ArgMax& rhs) {
        return (lhs.ix == rhs.ix
                && lhs.iy == rhs.iy
                && lhs.channel == rhs.channel
                && lhs.valid == rhs.valid);
    }
}
}

#endif // N2D2_POOLCELL_FRAME_KERNELS_STRUCT_H
