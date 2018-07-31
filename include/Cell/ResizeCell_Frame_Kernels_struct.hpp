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

#ifndef N2D2_RESIZECELL_FRAME_KERNELS_STRUCT_H
#define N2D2_ANCHORCELL_FRAME_KERNELS_STRUCT_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace N2D2 {
namespace ResizeCell_Frame_Kernels {
        struct PreComputed {
        int low_index;  
        int hight_index;  
        Float_T interpolation;
    };
}
}

#endif // N2D2_RESIZECELL_FRAME_KERNELS_STRUCT_H
