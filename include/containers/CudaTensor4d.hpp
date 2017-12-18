/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_CUDATENSOR4D_H
#define N2D2_CUDATENSOR4D_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "CudaUtils.hpp"
#include "containers/Tensor4d.hpp"

/**
*Tensor4d class :
*
*   Represents data through a 4 dimensional vector.
*
*       Chunk 1
*       +---------+
*       |  im 1   |
*       |  ...    |              x--------x
*       |         |    with     /  im i  /
*       |  im n   |            /        /  w
*       +---------+           x--------x
*       Chunk 2                   h
*       +---------+
*       |         |
*       |         |
*           ***
*
*   To access pixel (i, j) of the kth image of the lth chunk assuming chunk of
*size n :
*       pix(i, j, k, l) = data[n*h*w*l + h*w*k + w*i + j]
*/

namespace N2D2 {
template <typename T> class CudaTensor4d : public Tensor4d<T> {
public:
    CudaTensor4d();
    CudaTensor4d(Tensor4d<T>* base);
    CudaTensor4d(const CudaTensor4d<T>& tensor);
    CudaTensor4d(unsigned int dimX,
                 unsigned int dimY,
                 unsigned int dimZ,
                 unsigned int dimB);
    inline void reserve(unsigned int dimX,
                        unsigned int dimY,
                        unsigned int dimZ,
                        unsigned int dimB);
    inline void resize(unsigned int dimX,
                       unsigned int dimY = 1,
                       unsigned int dimZ = 1,
                       unsigned int dimB = 1,
                       const T& value = T());
    inline void assign(unsigned int dimX,
                       unsigned int dimY,
                       unsigned int dimZ,
                       unsigned int dimB,
                       const T& value);
    inline void push_back(const Tensor3d<T>& frame);
    inline void clear();

    /** Synchronize Device To Host */
    void synchronizeDToH() const;
    void synchronizeDToH(unsigned int index, unsigned int length) const;
    void synchronizeDToH(unsigned int i,
                         unsigned int j,
                         unsigned int k,
                         unsigned int b,
                         unsigned int length) const;
    void synchronizeDToH(unsigned int ijk,
                         unsigned int b,
                         unsigned int length) const;

    /** Synchronize Host To Device */
    void synchronizeHToD() const;
    void synchronizeHToD(unsigned int index, unsigned int length) const;
    void synchronizeHToD(unsigned int i,
                         unsigned int j,
                         unsigned int k,
                         unsigned int b,
                         unsigned int length) const;
    void synchronizeHToD(unsigned int ijk,
                         unsigned int b,
                         unsigned int length) const;

    /** Synchronize Device To Host-based data  */
    void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    void synchronizeHToDBased() const;

    void setDevicePtr(T* dataDevice)
    {
        mDataDevice = dataDevice;
    }
    T* getDevicePtr()
    {
        return mDataDevice;
    }
    cudnnTensorDescriptor_t& getCudnnTensorDesc()
    {
        return mTensor;
    }

    /** DEBUG */
    /*
            void dump(const std::string& fileName);
            void dumpData(int input_id);
            void dumpParam(const std::string& name);
            void dumpWeight();
    */
    ~CudaTensor4d();

protected:
    using Tensor4d<T>::mDimX;
    using Tensor4d<T>::mDimY;
    using Tensor4d<T>::mDimZ;
    using Tensor4d<T>::mDimB;
    using Tensor4d<T>::mData;

    cudnnTensorDescriptor_t mTensor;
    T* mDataDevice;
    bool mHostBased;
};
}

template <typename T>
N2D2::CudaTensor4d<T>::CudaTensor4d()
    : Tensor4d<T>(),
      mDataDevice(NULL),
      mHostBased(false)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
}

template <typename T>
N2D2::CudaTensor4d<T>::CudaTensor4d(Tensor4d<T>* base)
    : Tensor4d<T>(*base),
      mDataDevice(NULL),
      mHostBased(true)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));

    const unsigned int size = base->dimX() * base->dimY() * base->dimZ()
                              * base->dimB();

    if (size > 0) {
        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      base->dimB(),
                                                      base->dimZ(),
                                                      base->dimY(),
                                                      base->dimX()));
    }
}

template <typename T>
N2D2::CudaTensor4d<T>::CudaTensor4d(const CudaTensor4d<T>& tensor)
    : Tensor4d<T>(tensor.dimX(),
                  tensor.dimY(),
                  tensor.dimZ(),
                  tensor.dimB(),
                  tensor.begin(),
                  tensor.end()),
      mDataDevice(NULL),
      mHostBased(tensor.mHostBased)
{
    // copy-ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));

    const unsigned int size = tensor.dimX() * tensor.dimY() * tensor.dimZ()
                              * tensor.dimB();

    if (size > 0) {
        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      tensor.dimB(),
                                                      tensor.dimZ(),
                                                      tensor.dimY(),
                                                      tensor.dimX()));
    }
}

template <typename T>
N2D2::CudaTensor4d<T>::CudaTensor4d(unsigned int dimX,
                                    unsigned int dimY,
                                    unsigned int dimZ,
                                    unsigned int dimB)
    : Tensor4d<T>(dimX, dimY, dimZ, dimB),
      mDataDevice(NULL),
      mHostBased(false)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));

    const unsigned int size = dimX * dimY * dimZ * dimB;

    if (size > 0) {
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      dimB,
                                                      dimZ,
                                                      dimY,
                                                      dimX));

        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
    }
}

template <typename T>
void N2D2::CudaTensor4d<T>::reserve(unsigned int dimX,
                                    unsigned int dimY,
                                    unsigned int dimZ,
                                    unsigned int dimB)
{
    Tensor4d<T>::reserve(dimX, dimY, dimZ, dimB);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    const unsigned int size = dimX * dimY * dimZ * dimB;

    if (size > 0) {
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      dimB,
                                                      dimZ,
                                                      dimY,
                                                      dimX));

        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
    }
}

template <typename T>
void N2D2::CudaTensor4d<T>::resize(unsigned int dimX,
                                   unsigned int dimY,
                                   unsigned int dimZ,
                                   unsigned int dimB,
                                   const T& value)
{
    Tensor4d<T>::resize(dimX, dimY, dimZ, dimB, value);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    const unsigned int size = dimX * dimY * dimZ * dimB;

    if (size > 0) {
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      dimB,
                                                      dimZ,
                                                      dimY,
                                                      dimX));

        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
        synchronizeHToD();
    }
}

template <typename T>
void N2D2::CudaTensor4d<T>::assign(unsigned int dimX,
                                   unsigned int dimY,
                                   unsigned int dimZ,
                                   unsigned int dimB,
                                   const T& value)
{
    Tensor4d<T>::assign(dimX, dimY, dimZ, dimB, value);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    const unsigned int size = dimX * dimY * dimZ * dimB;

    if (size > 0) {
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mTensor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      dimB,
                                                      dimZ,
                                                      dimY,
                                                      dimX));

        CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size * sizeof(T)));
        synchronizeHToD();
    }
}

template <typename T>
void N2D2::CudaTensor4d<T>::push_back(const Tensor3d<T>& frame)
{
    Tensor4d<T>::push_back(frame);

    reserve(mDimX, mDimY, mDimZ, mDimB); // Resize device tensor accordingly
    synchronizeHToD(); // Copy data into device memory
}

template <typename T> void N2D2::CudaTensor4d<T>::clear()
{
    Tensor4d<T>::clear();

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }
}

/**
*   Synchronize data from device to host / Par morceau
*/
template <typename T> void N2D2::CudaTensor4d<T>::synchronizeDToH() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[0],
                                 mDataDevice,
                                 mDimX * mDimY * mDimZ * mDimB * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T>
void N2D2::CudaTensor4d
    <T>::synchronizeDToH(unsigned int index, unsigned int length) const
{
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[0] + index,
                                 mDataDevice + index,
                                 length * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T>
void N2D2::CudaTensor4d<T>::synchronizeDToH(unsigned int i,
                                            unsigned int j,
                                            unsigned int k,
                                            unsigned int b,
                                            unsigned int length) const
{
    const unsigned int index = i + mDimX * (j + mDimY * (k + mDimZ * b));
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[0] + index,
                                 mDataDevice + index,
                                 length * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T>
void N2D2::CudaTensor4d<T>::synchronizeDToH(unsigned int ijk,
                                            unsigned int b,
                                            unsigned int length) const
{
    const unsigned int index = ijk + b * mDimX * mDimY * mDimZ;
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[0] + index,
                                 mDataDevice + index,
                                 length * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

/**
*   Synchronize data from host to device / Par morceau
*/
template <typename T> void N2D2::CudaTensor4d<T>::synchronizeHToD() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice,
                                 &(*mData)[0],
                                 mDimX * mDimY * mDimZ * mDimB * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T>
void N2D2::CudaTensor4d
    <T>::synchronizeHToD(unsigned int index, unsigned int length) const
{
    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice + index,
                                 &(*mData)[0] + index,
                                 length * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T>
void N2D2::CudaTensor4d<T>::synchronizeHToD(unsigned int i,
                                            unsigned int j,
                                            unsigned int k,
                                            unsigned int b,
                                            unsigned int length) const
{
    const unsigned int index = i + mDimX * (j + mDimY * (k + mDimZ * b));
    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice + index,
                                 &(*mData)[0] + index,
                                 length * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T>
void N2D2::CudaTensor4d<T>::synchronizeHToD(unsigned int ijk,
                                            unsigned int b,
                                            unsigned int length) const
{
    const unsigned int index = ijk + b * mDimX * mDimY * mDimZ;
    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice + index,
                                 &(*mData)[0] + index,
                                 length * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

/**
*   Synchronize data from valid host to device / Par morceau
*/
template <typename T> void N2D2::CudaTensor4d<T>::synchronizeDToHBased() const
{
    if (mHostBased)
        synchronizeDToH();
}

template <typename T> void N2D2::CudaTensor4d<T>::synchronizeHBasedToD() const
{
    if (mHostBased)
        synchronizeHToD();
}

template <typename T> void N2D2::CudaTensor4d<T>::synchronizeDBasedToH() const
{
    if (!mHostBased)
        synchronizeDToH();
}

template <typename T> void N2D2::CudaTensor4d<T>::synchronizeHToDBased() const
{
    if (!mHostBased)
        synchronizeHToD();
}

/*
template<typename T>
void N2D2::CudaTensor4d<T>::dump(const std::string& fileName) {
    std::ofstream outfile(fileName.c_str(), std::ios::out);
    outfile << "Dump interface parameters : " << mData.size() << std::endl;
    outfile << "Nb tensor : " << mData.size() << std::endl;
    outfile << "mDimX : " << mDimX << std::endl;
    outfile << "mDimY : " << mDimY << std::endl;
    outfile << "mDimZ : " << mDimZ << std::endl;
    outfile << "mDimB : " << mDimB << std::endl;

    for (unsigned int l = 0; l < mDimB; ++l) {
        for (unsigned int i = 0; i < mDimZ; ++i) {
            for (unsigned int j = 0; j < mDimY; ++j) {
                for (unsigned int k = 0; k < mDimX; ++k)
                    outfile  << std::fixed << std::setprecision(2) << (*this)(k,
j, i, l) << " ";

                outfile << std::endl;
            }
            outfile << std::endl;
        }
        outfile << std::endl;
    }
}

template<typename T>
void N2D2::CudaTensor4d<T>::dumpData(int input_id) {
    std::cout << "\nDump Tensor : " << std::endl;
    for (unsigned int j = 0; j < mDimZ; j++) {
        for (unsigned int k = 0; k < mDimY; k++) {
            for (unsigned int l = 0; l < mDimX; l++) {
                std::cout << std::fixed << std::setprecision(2)
                    << mData[mDimZ*mDimX*mDimY*input_id + j*mDimX*mDimY +
k*mDimX + l] << " ";
            }

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template<typename T>
void N2D2::CudaTensor4d<T>::dumpParam(const std::string& name) {
    std::cout << "\nDump tensor parameters : " << name << std::endl;
    std::cout << "\tdimX : " << mDimX << std::endl;
    std::cout << "\tdimY : " << mDimY << std::endl;
    std::cout << "\tdimZ : " << mDimZ << std::endl;
    std::cout << "\tdimB : " << mDimB << std::endl;
}

template<typename T>
void N2D2::CudaTensor4d<T>::dumpWeight() {
    std::cout << "\nDump Weight : " << std::endl;
    for (unsigned int j = 0; j < mDimZ; j++) {
        for (unsigned int k = 0; k < mDimB; k++)
            std::cout << std::fixed << std::setprecision(2) << mData[j*mDimB+k]
<< " ";

        std::cout << std::endl;
    }
}
*/
template <typename T> N2D2::CudaTensor4d<T>::~CudaTensor4d()
{
    clear();
    CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mTensor));
}

#endif // N2D2_CUDATENSOR4D_H
