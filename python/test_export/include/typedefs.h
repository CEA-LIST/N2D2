/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_EXPORTC_TYPEDEFS_H
#define N2D2_EXPORTC_TYPEDEFS_H

#include "params.h"

typedef enum {
    Logistic,
    LogisticWithLoss,
    FastSigmoid,
    Tanh,
    TanhLeCun,
    Saturation,
    Rectifier,
    Linear,
    Softplus
} ActivationFunction_T;

typedef enum {
    Convolution,
    Pooling,
    Softmax,
    fcCellProp2D,
    fcCellProp,
    SpatialOutputs
} kernel_T;

typedef enum {
    Max,
    Average
} Pooling_T;

#if NB_BITS == -64
typedef double DATA_T;
typedef double UDATA_T;
typedef double SUM_T;
typedef SUM_T BDATA_T;
#elif NB_BITS == -32 || NB_BITS == -16
typedef float DATA_T;
typedef float UDATA_T;
typedef float SUM_T;
typedef SUM_T BDATA_T;
#elif NB_BITS == 8
typedef char DATA_T;
typedef unsigned char UDATA_T;
typedef int SUM_T;
typedef SUM_T BDATA_T;
#elif NB_BITS == 16
typedef short DATA_T;
typedef unsigned short UDATA_T;
typedef long long int SUM_T;
typedef SUM_T BDATA_T;
#elif NB_BITS == 32 || NB_BITS == 64
typedef int DATA_T;
typedef unsigned int UDATA_T;
typedef long long int SUM_T;
typedef SUM_T BDATA_T;
#else
#define CONCAT(x, y) x##y
#define INT(x) CONCAT(int, x)
#define UINT(x) CONCAT(uint, x)

#include <ap_cint.h>

typedef INT(NB_BITS) DATA_T;
typedef UINT(NB_BITS) UDATA_T;
typedef int SUM_T;
typedef SUM_T BDATA_T;
#endif

typedef DATA_T WDATA_T;

#if NB_BITS < 0
#define DATA_T_MAX 1.0
#define DATA_T_MIN -1.0
#define UDATA_T_MAX 1.0
#define UDATA_T_MIN 0.0
#else
#define DATA_T_MAX ((1LL << (NB_BITS - 1)) - 1)
#define DATA_T_MIN (-(1LL << (NB_BITS - 1)))
#define UDATA_T_MAX ((1LL << NB_BITS) - 1)
#define UDATA_T_MIN 0LL
#endif

#endif // N2D2_EXPORTC_TYPEDEFS_H
