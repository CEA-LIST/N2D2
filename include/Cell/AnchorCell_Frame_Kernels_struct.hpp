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

#ifndef N2D2_ANCHORCELL_FRAME_KERNELS_STRUCT_H
#define N2D2_ANCHORCELL_FRAME_KERNELS_STRUCT_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace N2D2 {
namespace AnchorCell_Frame_Kernels {
    struct BBox_T {
        float x;
        float y;
        float w;
        float h;

        CUDA_HOSTDEV BBox_T() {}
        CUDA_HOSTDEV BBox_T(float x_, float y_, float w_, float h_):
            x(x_), y(y_), w(w_), h(h_) {}
    };

    struct Anchor {
        enum Anchoring {
            TopLeft,
            Centered,
            Original,
            OriginalFlipped
        };

        Anchor(float x0_,
               float y0_,
               float width_,
               float height_)
            : x0(x0_),
              y0(y0_),
              x1(width_ + x0_ - 1),
              y1(height_ + y0_ - 1)
        {
        }
        Anchor(float width, float height, Anchoring anchoring = TopLeft);
        Anchor(unsigned int area,
               double ratio,
               double scale = 1.0,
               Anchoring anchoring = TopLeft);

        float x0;
        float y0;
        float x1;
        float y1;

    private:
        float round(float x);
    };
}
}

#endif // N2D2_ANCHORCELL_FRAME_KERNELS_STRUCT_H
