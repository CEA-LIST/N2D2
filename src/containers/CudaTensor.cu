/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "containers/CudaTensor.hpp"

template <typename T>
void N2D2::thrust_fill(T* devData, size_t size, T value)
{
    thrust::device_ptr<T> thrustPtr(devData);
    thrust::fill(thrustPtr, thrustPtr + size, value);
}

template void N2D2::thrust_fill<double>(double* devData, size_t size, double value);
template void N2D2::thrust_fill<float>(float* devData, size_t size, float value);
template void N2D2::thrust_fill<char>(char* devData, size_t size, char value);
template void N2D2::thrust_fill<int>(int* devData, size_t size, int value);
template void N2D2::thrust_fill<long long int>(long long int* devData, size_t size, long long int value);
template void N2D2::thrust_fill<unsigned int>(unsigned int* devData, size_t size, unsigned int value);
template void N2D2::thrust_fill<unsigned long long int>(unsigned long long int* devData, size_t size, unsigned long long int value);

template <>
void N2D2::thrust_copy(double* srcData, float* dstData, size_t size)
{
    thrust::device_ptr<double> thrustSrcPtr(srcData);
    thrust::device_ptr<float> thrustDstPtr(dstData);
    thrust::copy(thrustSrcPtr, thrustSrcPtr + size, thrustDstPtr);
}

template <>
void N2D2::thrust_copy(float* srcData, double* dstData, size_t size)
{
    thrust::device_ptr<float> thrustSrcPtr(srcData);
    thrust::device_ptr<double> thrustDstPtr(dstData);
    thrust::copy(thrustSrcPtr, thrustSrcPtr + size, thrustDstPtr);
}
