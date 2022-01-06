/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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
#ifndef DNN_UTILS_HPP
#define DNN_UTILS_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <NvInfer.h>
#ifdef ONNX
#include <NvOnnxParser.h>
#endif
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <iterator>
#include <memory> // std::unique_ptr
#include <vector>
#include <string>
#include <algorithm> // std::sort
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <iomanip>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include "typedefs.h"
#include "utils.h"
#include "common_cuda.hpp"

#if NV_TENSORRT_MAJOR < 5
    typedef nvinfer1::DimsNCHW trt_Dims4;
    typedef nvinfer1::DimsCHW trt_Dims3;
    typedef nvinfer1::DimsHW trt_Dims2;
    typedef nvinfer1::DimsHW trt_DimsHW;

#else
    typedef nvinfer1::Dims4 trt_Dims4;
    typedef nvinfer1::Dims3 trt_Dims3;
    typedef nvinfer1::Dims2 trt_Dims2;
    typedef nvinfer1::DimsHW trt_DimsHW;
#endif



static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

static unsigned int prevDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        --v;
    return v;
}

struct LayerActivation {
    bool status;
    nvinfer1::ActivationType type;
    double alpha;
    double beta;
    LayerActivation(bool status_,
               nvinfer1::ActivationType type_,
               double alpha_ = 0.0,
               double beta_ = 0.0)
    : status(status_),
      type(type_),
      alpha(alpha_),
      beta(beta_)
    {
    }
    LayerActivation(bool status_)
    : status(status_)
    {
    }
};


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

template <class T>
inline const T& clamp_export(const T& x, const T& min, const T& max);


template <class T>
const T& clamp_export(const T& x, const T& min, const T& max)
{
    return (x < min) ? min : (x > max) ? max : x;
}

struct ROI {
    int i;
    int j;
    int k;
    int b;
    ROI(unsigned int i_ = 0,
           unsigned int j_ = 0,
           unsigned int k_ = 0,
           unsigned int b_ = 0)
        : i(i_), j(j_), k(k_), b(b_) {}

};

struct Anchor {
    float x0;
    float y0;
    float x1;
    float y1;
};

template <class T1, class T2, class Pred = std::less<T2> >
struct PairSecondPred : public std::binary_function
                        <std::pair<T1, T2>, std::pair<T1, T2>, bool> {
    bool operator()(const std::pair<T1, T2>& left,
                    const std::pair<T1, T2>& right) const
    {
        Pred p;
        return p(left.second, right.second);
    }
};

struct BBox_T {
    float x;
    float y;
    float w;
    float h;
    float s;

    BBox_T() {}
    BBox_T(float x_, float y_, float w_, float h_, float s_):
        x(x_), y(y_), w(w_), h(h_), s(s_) {}
};


#endif
