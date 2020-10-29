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

#include "containers/Tensor.hpp"

#include <complex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "NodeEnv.hpp"
#include "Synapse.hpp"
#include "Cell/NodeOut.hpp"
#include "Cell/AnchorCell_Frame_Kernels_struct.hpp"
#include "Cell/PoolCell_Frame_Kernels.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"

namespace {
    template<class U>
    U* getDataPtr(std::vector<U>& v) {
        return v.data();
    }

    bool* getDataPtr(std::vector<bool>& /*v*/) {
        throw std::runtime_error("Can't get the data() from a vector<bool>.");
    }

    template<typename To, typename From,
             typename std::enable_if<std::is_convertible<From, To>::value>::type* = nullptr>
    To convertValue(const From& value) {
        return static_cast<To>(value);
    }

    template<typename To, typename From,
             typename std::enable_if<!std::is_convertible<From, To>::value>::type* = nullptr>
    To convertValue(const From& /*value*/) {
        throw std::runtime_error("Can't convert value, types are incompatibles.");
    }

    template<typename T, typename Enable = void>
    struct try_make_signed {
        typedef typename std::make_signed<T>::type type;
    };

    template<typename T>
    struct try_make_signed<T,
        typename std::enable_if<std::is_floating_point<T>::value>::type> {
        typedef T type;
    };

    template<typename T, typename Enable = void>
    struct try_make_unsigned {
        typedef typename std::make_unsigned<T>::type type;
    };

    template<typename T>
    struct try_make_unsigned<T,
        typename std::enable_if<std::is_floating_point<T>::value>::type> {
        typedef T type;
    };
}


/**
 * BaseTensor
 */
void N2D2::BaseTensor::reserve(std::initializer_list<size_t> dims)
{
    reserve(std::vector<size_t>(dims));
}

void N2D2::BaseTensor::resize(std::initializer_list<size_t> dims)
{
    resize(std::vector<size_t>(dims));
}

void N2D2::BaseTensor::reshape(std::initializer_list<size_t> dims)
{
    reshape(std::vector<size_t>(dims));
}

void N2D2::BaseTensor::reshape(const std::vector<size_t>& dims)
{
    const size_t oldSize = size();
    const std::vector<size_t> oldDims = mDims;
    mDims = dims;
    const size_t newSize = computeSize();

    if (newSize != oldSize) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::reshape(): new size (" << mDims << " = "
            << newSize << ") does not match current size (" << oldDims << " = "
            << oldSize << ")" << std::endl;

        throw std::runtime_error(errorStr.str());
    }
}


/**
 * Tensor
 */
template <class T>
N2D2::Tensor<T>::Tensor()
    : BaseTensor(),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>())),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(std::initializer_list<size_t> dims,
                            const T& value)
    : BaseTensor(dims),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>(computeSize(),
                                                            value))),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                            const T& value)
    : BaseTensor(dims),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>(computeSize(),
                                                            value))),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<unsigned int>& dims,
                            const T& value)
    : BaseTensor(std::vector<size_t>(dims.begin(), dims.end())),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>(computeSize(),
                                                            value))),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                        const std::shared_ptr<DataTensor<T> >& data,
                        const std::shared_ptr<bool>& valid,
                        size_t dataOffset,
                        size_t size,
                        size_t sizeM1)
    : BaseTensor(dims, valid, size, sizeM1),
      mData(data),
      mDataOffset(dataOffset)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims, T* dataPtr)
    : BaseTensor(dims),
      mData(std::make_shared<DataTensor<T> >(
          std::vector<T>(dataPtr, dataPtr + computeSize()))),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const cv::Mat& mat, bool signedMapping)
    : BaseTensor(std::vector<size_t>(), std::make_shared<bool>(true)),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>())),
      mDataOffset(0)
{
    // ctor
    mDims.reserve(2);
    mDims.push_back(mat.cols);
    mDims.push_back(mat.rows);

    if (mat.channels() > 1)
        mDims.push_back(mat.channels());

    (*mData)().reserve(computeSize());

    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    for (std::vector<cv::Mat>::const_iterator itChannel = channels.begin();
         itChannel != channels.end();
         ++itChannel)
    {
        switch ((*itChannel).depth()) {
        case CV_8U:
            convert<unsigned char>(*itChannel, (*mData)(), signedMapping);
            break;
        case CV_8S:
            convert<char>(*itChannel, (*mData)());
            break;
        case CV_16U:
            convert<unsigned short>(*itChannel, (*mData)(), signedMapping);
            break;
        case CV_16S:
            convert<short>(*itChannel, (*mData)());
            break;
        case CV_32S:
            convert<int>(*itChannel, (*mData)());
            break;
        case CV_32F:
            convert<float>(*itChannel, (*mData)());
            break;
        case CV_64F:
            convert<double>(*itChannel, (*mData)());
            break;
        default:
            throw std::runtime_error(
                "Cannot convert cv::Mat to Tensor: incompatible types.");
        }
    }

    assert((*mData)().size() == static_cast<std::size_t>(mat.rows * mat.cols * mat.channels()));
    assert((*mData)().size() == size());
}


template <class T>
void N2D2::Tensor<T>::reserve(const std::vector<size_t>& dims)
{
    assert(mData.unique());

    mDims = dims;
    (*mData)().reserve(computeSize());
}

template <class T>
void N2D2::Tensor<T>::resize(const std::vector<size_t>& dims)
{
    assert(mData.unique());

    mDims = dims;
    (*mData)().resize(computeSize());
}

template <class T>
void N2D2::Tensor<T>::resize(std::initializer_list<size_t> dims,
                               const T& value)
{
    resize(std::vector<size_t>(dims), value);
}

template <class T>
void N2D2::Tensor<T>::resize(const std::vector<size_t>& dims,
                               const T& value)
{
    assert(mData.unique());

    mDims = dims;
    (*mData)().resize(computeSize(), value);
}

template <class T>
void N2D2::Tensor<T>::assign(std::initializer_list<size_t> dims,
                               const T& value)
{
    assign(std::vector<size_t>(dims), value);
}

template <class T>
void N2D2::Tensor<T>::assign(const std::vector<size_t>& dims,
                               const T& value)
{
    assert(mData.unique());

    mDims = dims;
    (*mData)().assign(computeSize(), value);
}

template <typename T>
void N2D2::Tensor<T>::fill(const T& value)
{
    std::fill((*mData)().begin() + mDataOffset,
              (*mData)().begin() + mDataOffset + size(), value);
}

template <class T>
void N2D2::Tensor<T>::push_back(const T& value)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims.resize(1, 0);
    }
    else if (mDims.size() != 1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor must be 1D to push back a"
            " single value, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    ++mDims.back();
    computeSize();
    (*mData)().push_back(value);
}

template <class T>
void N2D2::Tensor<T>::push_back(const std::vector<T>& vec)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims.resize(1, vec.size());
        mDims.push_back(0);
    }
    else if (mDims.size() != 2) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor must be 2D to push back a"
            " vector, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }
    else {
        if (mDims[0] != vec.size()) {
            std::stringstream errorStr;
            errorStr << "Tensor<T>::push_back(): tensor first dimension must"
                " match the vector size (" << vec.size() << "), but is "
                << mDims[0] << " (" << mDims << ")" << std::endl;

            throw std::runtime_error(errorStr.str());
        }
    }

    ++mDims.back();
    computeSize();
    (*mData)().insert((*mData)().end(), vec.begin(), vec.end());
}

template <class T>
void N2D2::Tensor<T>::push_back(const Tensor<T>& frame)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims = frame.dims();
        mDims.push_back(0);
    }
    else if (mDims.size() != frame.nbDims() + 1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor must be "
            << (frame.nbDims() + 1) << "D to push back a " << frame.nbDims()
            << "D tensor, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }
    else {
        for (unsigned int dim = 0; dim < frame.nbDims(); ++dim) {
            if (mDims[dim] != frame.dims()[dim]) {
                std::stringstream errorStr;
                errorStr << "Tensor<T>::push_back(): tensors dimension #"
                    << dim << " must match, but tensor dimension is "
                    << mDims << " and tensor to push back is "
                    << frame.dims() << std::endl;

                throw std::runtime_error(errorStr.str());
            }
        }
    }

    ++mDims.back();
    computeSize();
    (*mData)().insert((*mData)().end(), frame.begin(), frame.end());
}

template <class T>
void N2D2::Tensor<T>::append(const std::vector<T>& vec)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims.resize(1, 0);
    }
    else if (mDims.size() != 1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::append(): tensor must be 1D to append a"
            " vector, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    mDims.back() += vec.size();
    computeSize();
    (*mData)().insert((*mData)().end(), vec.begin(), vec.end());
}

template <class T>
void N2D2::Tensor<T>::append(const Tensor<T>& frame)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims = frame.dims();
    }
    else if (mDims.size() != frame.nbDims()) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::append(): tensor must be "
            << frame.nbDims() << "D to append a " << frame.nbDims()
            << "D tensor, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }
    else {
        for (unsigned int dim = 0; dim < frame.nbDims() - 1; ++dim) {
            if (mDims[dim] != frame.dims()[dim]) {
                std::stringstream errorStr;
                errorStr << "Tensor<T>::append(): tensors dimension #"
                    << dim << " must match, but tensor dimension is "
                    << mDims << " and tensor to append is "
                    << frame.dims() << std::endl;

                throw std::runtime_error(errorStr.str());
            }
        }

        mDims.back() += frame.dims().back();
    }

    computeSize();
    (*mData)().insert((*mData)().end(), frame.begin(), frame.end());
}

template <class T>
void N2D2::Tensor<T>::clear()
{
    assert(mData.unique());

    mDims.clear();
    mSize = 0;
    mSizeM1 = 0;
    (*mData)().clear();
}

template <class T>
void N2D2::Tensor<T>::save(std::ostream& stream) const
{
    const size_t dimsSize = mDims.size();
    stream.write(reinterpret_cast<const char*>(&dimsSize), sizeof(dimsSize));

    for (std::vector<size_t>::const_iterator it = mDims.begin();
        it != mDims.end(); ++it)
    {
        stream.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    stream.write(reinterpret_cast<const char*>(&mSize), sizeof(mSize));

    for (typename std::vector<T>::const_iterator it = (*mData)().begin();
        it != (*mData)().end(); ++it)
    {
        const T value = (*it);
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

template <class T>
void N2D2::Tensor<T>::load(std::istream& stream)
{
    size_t dimsSize;
    stream.read(reinterpret_cast<char*>(&dimsSize), sizeof(dimsSize));

    std::vector<size_t> dims(dimsSize);

    for (std::vector<size_t>::iterator it = dims.begin();
        it != dims.end(); ++it)
    {
        stream.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    // Only call resize() if the stored size is different, as resize() implies
    // mData shared_ptr unicity
    if (dims != mDims)
        resize(dims);

    size_t dataSize;
    stream.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

    if (dataSize != mSize)
        throw std::runtime_error("Tensor<T>::load(): mismatch in tensor size!");

    for (typename std::vector<T>::iterator it = (*mData)().begin();
        it != (*mData)().end(); ++it)
    {
        T value;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        (*it) = value;
    }
}

template <class T>
void N2D2::Tensor<T>::swap(Tensor<T>& tensor)
{
    assert(mData.unique());
    assert(mDataOffset == 0);
    assert(tensor.mDataOffset == 0);

    // BaseTensor
    mDims.swap(tensor.mDims);
    std::swap((*mValid), (*tensor.mValid));
    std::swap(mSize, tensor.mSize);
    std::swap(mSizeM1, tensor.mSizeM1);
    mDataTensors.swap(tensor.mDataTensors);

    // Tensor<T>
    (*mData)().swap((*tensor.mData)());

    assert((*mData)().size() == size());
    assert((*tensor.mData)().size() == tensor.size());
}

template <class T>
N2D2::Tensor<T> N2D2::Tensor<T>::clone() const {
    return Tensor<T>(mDims,
                     std::make_shared<DataTensor<T> >(
                                                std::vector<T>(begin(), end())),
                     mValid,
                     0,
                     mSize,
                     mSizeM1);
}

template <class T>
N2D2::Tensor<T> N2D2::Tensor<T>::operator[](size_t i)
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();
    return Tensor<T>(newDims, mData, mValid, mDataOffset + i * mSizeM1,
                mSizeM1, (newDims.back() > 0) ? mSizeM1 / newDims.back() : 0);
}

template <class T>
const N2D2::Tensor<T> N2D2::Tensor<T>::operator[](size_t i) const
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();
    return Tensor<T>(newDims, mData, mValid, mDataOffset + i * mSizeM1,
                mSizeM1, (newDims.back() > 0) ? mSizeM1 / newDims.back() : 0);
}

template <class T>
N2D2::Tensor<T> N2D2::Tensor<T>::rows(size_t j0,
                                          size_t nb)
{
    assert(mDims.size() > 1);
    assert(j0 + nb <= mDims.back());

    std::vector<size_t> newDims = mDims;
    newDims.back() = nb;
    return Tensor<T>(newDims, mData, mValid, mDataOffset + j0 * mSizeM1,
                     nb * mSizeM1, mSizeM1);
}

template <class T>
const N2D2::Tensor<T> N2D2::Tensor<T>::rows(size_t j0,
                                                size_t nb) const
{
    assert(mDims.size() > 1);
    assert(j0 + nb <= mDims.back());

    std::vector<size_t> newDims = mDims;
    newDims.back() = nb;
    return Tensor<T>(newDims, mData, mValid, mDataOffset + j0 * mSizeM1,
                     nb * mSizeM1, mSizeM1);
}

template <class T>
N2D2::BaseTensor& N2D2::Tensor<T>::operator=(const BaseTensor& base)
{
    assert(mDims.size() == base.nbDims());

    for (unsigned int dim = 0; dim < mDims.size(); ++dim) {
        assert(mDims[dim] == base.dims()[dim]);
    }

    const Tensor<T>& tensor = tensor_cast<T>(base);

    if (tensor.mData != mData || tensor.mDataOffset != mDataOffset) {
        // Actual copy only if data is different
        std::copy(tensor.begin(), tensor.end(),
                  (*mData)().begin() + mDataOffset);
    }

    return *this;
}

template <class T>
N2D2::Tensor<T>& N2D2::Tensor<T>::operator=(const Tensor<T>& tensor)
{
    assert(mDims.size() == tensor.nbDims());

    for (unsigned int dim = 0; dim < mDims.size(); ++dim) {
        assert(mDims[dim] == tensor.dims()[dim]);
    }

    if (tensor.mData != mData || tensor.mDataOffset != mDataOffset) {
        // Actual copy only if data is different
        std::copy(tensor.begin(), tensor.end(),
                  (*mData)().begin() + mDataOffset);
    }

    return *this;
}

template <class T>
N2D2::Tensor<T>::operator cv::Mat() const
{
    const int type = (std::is_same<T, char>::value)         ? CV_8SC1 :
                 (std::is_same<T, unsigned char>::value)    ? CV_8UC1 :
                 (std::is_same<T, short>::value)            ? CV_16SC1 :
                 (std::is_same<T, unsigned short>::value)   ? CV_16UC1 :
                 (std::is_same<T, int>::value)              ? CV_32SC1 :
                 (std::is_same<T, float>::value)            ? CV_32FC1 :
                 (std::is_same<T, double>::value)           ? CV_64FC1 :
                                                              -1;
    if (type == -1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::operator cv::Mat(): Cannot convert Tensor to "
            "cv::Mat: Tensor type is not supported (" << typeid(T).name()
            << ")." << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    if (mDims.size() < 3) {
        return cv::Mat((int)((mDims.size() > 1) ? mDims[1] :
                             (mDims.size() > 0) ? 1 : 0),
                       (int)((mDims.size() > 0) ? mDims[0] : 0),
                       type,
                       getDataPtr((*mData)()) + mDataOffset);
    }
    else if (mDims.size() == 3) {
        std::vector<cv::Mat> channels;

        for (size_t k = 0; k < mDims[2]; ++k) {
            channels.push_back(cv::Mat((int)mDims[1],
                                       (int)mDims[0],
                                       type,
                                       getDataPtr((*mData)()) + mDataOffset + k * mDims[0] * mDims[1]));
        }

        cv::Mat mat;
        cv::merge(channels, mat);
        return mat;
    }
    else {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::operator cv::Mat(): Cannot convert Tensor to "
            "cv::Mat: tensor dimension (" << mDims.size() << ": " << mDims
            << ") is > 3." << std::endl;

        throw std::runtime_error(errorStr.str());
    }
}

template <class T>
typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::operator()(const Index& index)
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    return (*mData)()[mDataOffset + offset];
}

template <class T>
typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::operator()(const Index& index) const
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    return (*mData)()[mDataOffset + offset];
}

//TODO: Generalize this to different data types and subtensors?
template <class T>
double N2D2::Tensor<T>::sum(bool valAbs) const
{
    assert(mDims.size() > 1);

    double sum = 0.0;

    for (typename std::vector<T>::iterator it = (*mData)().begin();
        it != (*mData)().end(); ++it)
    {
        if (valAbs) sum += abs(convertValue<double>(*it));
        else sum += convertValue<double>(*it);
    }
    return sum;
}

//TODO: Generalize this to different data types and subtensors?
template <class T>
double N2D2::Tensor<T>::mean(bool valAbs) const
{
    return sum(valAbs)/(*mData)().size();
}

template <class T>
double N2D2::Tensor<T>::std() const
{
    double m = mean();
    
    double var = 0.0;

    for (typename std::vector<T>::iterator it = (*mData)().begin();
        it != (*mData)().end(); ++it)
    {
        var += pow(convertValue<double>(*it) - m, 2);
    }
    var = var/(*mData)().size();

    return sqrt(var);
}

template <class T>
bool N2D2::Tensor<T>::operator==(const Tensor& other) const {
    if(mDims != other.mDims) {
        return false;
    }

    if(mData.get() == other.mData.get() && mDataOffset == other.mDataOffset) {
        return true;
    }

    assert((*mData)().size() == (*other.mData)().size());
    return std::equal(begin(), end(), other.begin());
}

template <class T>
bool N2D2::Tensor<T>::operator!=(const Tensor& other) const {
    return !(*this == other);
}

template <class T>
template <class CV_T, class U,
          typename std::enable_if<std::is_arithmetic<U>::value &&
                                  !std::is_same<U, bool>::value>::type*>
void N2D2::Tensor<T>::convert(const cv::Mat& mat, std::vector<U>& data,
                              bool signedMapping)
{
    const CV_T srcRange = (std::numeric_limits<CV_T>::is_integer)
                              ? ((signedMapping)
                                    ? static_cast<CV_T>(-std::numeric_limits
                                        <typename try_make_signed<CV_T>::type>
                                                                        ::min())
                                    : std::numeric_limits<CV_T>::max())
                              : CV_T(1.0);
    const T dstRange = (std::numeric_limits<T>::is_integer)
                           ? std::numeric_limits<T>::max()
                           : T(1.0);

    // We know dstRange and srcRange are positive. Convert them to unsigned to avoid
    // a potential 'comparison between signed and unsigned integer expressions" warning.
    if (static_cast<typename try_make_unsigned<CV_T>::type>(srcRange) ==
        static_cast<typename try_make_unsigned<U>::type>(dstRange))
    {
        if (mat.isContinuous())
            data.insert(data.end(), (CV_T*) mat.datastart, (CV_T*) mat.dataend);
        else {
            for (int i = 0; i < mat.rows; ++i) {
                data.insert(data.end(),
                            mat.ptr<CV_T>(i), mat.ptr<CV_T>(i) + mat.cols);
            }
        }
    }
    else {
        for (int i = 0; i < mat.rows; ++i) {
            const CV_T* rowPtr = mat.ptr<CV_T>(i);

            for (int j = 0; j < mat.cols; ++j) {
                if (std::numeric_limits<CV_T>::is_integer && signedMapping) {
                    data.push_back(static_cast<T>(
                        ((std::numeric_limits<CV_T>::is_integer
                          && std::numeric_limits<T>::is_integer)
                             ? static_cast<long long int>(dstRange)
                             : static_cast<double>(dstRange))
                            * (rowPtr[j] + std::numeric_limits<
                                  typename try_make_signed<CV_T>::type>::min())
                            / srcRange));
                }
                else {
                    data.push_back(static_cast<T>(
                        ((std::numeric_limits<CV_T>::is_integer
                          && std::numeric_limits<T>::is_integer)
                             ? static_cast<long long int>(dstRange)
                             : static_cast<double>(dstRange))
                            * rowPtr[j] / srcRange));
                }
            }
        }
    }
}

template <class T>
template <class CV_T, class U,
          typename std::enable_if<!(std::is_arithmetic<U>::value &&
                                    !std::is_same<U, bool>::value)>::type*>
void N2D2::Tensor<T>::convert(const cv::Mat& /*mat*/, std::vector<U>& /*data*/,
                              bool /*signedMapping*/)
{
    throw std::runtime_error("Can't convert from or to a non arithmetic Tensor.");
}

#ifdef CUDA

#include "containers/CudaTensor.hpp"
namespace {
    template<typename U,
             typename std::enable_if<std::is_same<typename U::value_type, half_float::half>::value ||
                                     std::is_same<typename U::value_type, float>::value ||
                                     std::is_same<typename U::value_type, double>::value>::type* = nullptr>
    N2D2::CudaTensor<typename U::value_type>* newCudaImpl(const U& value) {
        return new N2D2::CudaTensor<typename U::value_type>(value);
    }

    template<typename U,
             typename std::enable_if<!(std::is_same<typename U::value_type, half_float::half>::value ||
                                       std::is_same<typename U::value_type, float>::value ||
                                       std::is_same<typename U::value_type, double>::value)>::type* = nullptr>
    U* newCudaImpl(const U& ) {
        throw std::runtime_error("Tensor::newCuda(): type not supported");
    }
}

template <class T>
N2D2::BaseTensor* N2D2::Tensor<T>::newCuda() const {
    return newCudaImpl(*this);
}
#endif

template class N2D2::Tensor<float>;
template class N2D2::Tensor<double>;
template class N2D2::Tensor<half_float::half>;
template class N2D2::Tensor<bool>;
template class N2D2::Tensor<char>;
template class N2D2::Tensor<unsigned char>;
template class N2D2::Tensor<short>;
template class N2D2::Tensor<unsigned short>;
template class N2D2::Tensor<int>;
template class N2D2::Tensor<unsigned int>;
template class N2D2::Tensor<long>;
template class N2D2::Tensor<unsigned long>;
template class N2D2::Tensor<long long>;
template class N2D2::Tensor<unsigned long long>;
template class N2D2::Tensor<std::vector<unsigned int>>;
template class N2D2::Tensor<std::pair<unsigned char, unsigned char>>;
template class N2D2::Tensor<std::pair<unsigned long long, char>>;
template class N2D2::Tensor<std::pair<unsigned long long, int>>;
template class N2D2::Tensor<std::complex<double>>;
template class N2D2::Tensor<N2D2::Synapse*>;
template class N2D2::Tensor<N2D2::NodeEnv*>;
template class N2D2::Tensor<N2D2::NodeOut*>;
template class N2D2::Tensor<N2D2::AnchorCell_Frame_Kernels::Anchor>;
template class N2D2::Tensor<N2D2::AnchorCell_Frame_Kernels::BBox_T>;
template class N2D2::Tensor<N2D2::PoolCell_Frame_Kernels::ArgMax>;


namespace N2D2 {
    namespace CNNIP {
        class Instance;
    }
}
template class N2D2::Tensor<std::shared_ptr<N2D2::CNNIP::Instance>>;


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T, typename std::enable_if<!std::is_same<T, bool>::value>::type* = nullptr>
void declare_Tensor_buffer_protocol(py::class_<Tensor<T>, BaseTensor>& tensor) {
    // Buffer protocol
    tensor.def_buffer([](Tensor<T>& b) -> py::buffer_info {
        //assert(mData.unique());

        std::vector<ssize_t> dims;
        std::vector<ssize_t> strides;
        ssize_t stride = sizeof(T);

        for (unsigned int dim = 0; dim < b.nbDims(); ++dim) {
            dims.push_back(b.dims()[dim]);
            strides.push_back(stride);
            stride *= b.dims()[dim];
        }

        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());

        return py::buffer_info(
            &b.data()[0],                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            b.nbDims(),                                      /* Number of dimensions */
            dims,                 /* Buffer dimensions */
            strides             /* Strides (in bytes) for each index */
        );
    })
    .def("__init__", [](Tensor<T>& m, py::array_t<T, py::array::c_style | py::array::forcecast> b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();
/*
        // Some sanity checks... -> not needed with py::array_t<...>
        if (info.format != py::format_descriptor<T>::format())
            throw std::runtime_error("Incompatible format!");

        ssize_t stride = sizeof(T);

        for (unsigned int dim = 0; dim < b.ndim; ++dim) {
            if (stride != info.strides[dim])
                throw std::runtime_error("Incompatible buffer stride!");

            stride *= info.shape[dim];
        }
*/
        const std::vector<size_t> dims(info.shape.begin(), info.shape.end());
        new (&m) Tensor<T>(dims, static_cast<T*>(info.ptr));
    });
}

template<typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
void declare_Tensor_buffer_protocol(py::class_<Tensor<T>, BaseTensor>& /*tensor*/) {
    // No buffer protocol for bool!
}

template<typename T>
void declare_Tensor(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("Tensor_" + typeStr);
    py::class_<Tensor<T>, BaseTensor> tensor(m, pyClassName.c_str(), py::multiple_inheritance(), py::buffer_protocol());
    tensor.def(py::init<>())
    .def(py::init<const std::vector<size_t>&, const T&>(), py::arg("dims"), py::arg("value") = T())
    /// Bare bones interface
    .def("__getitem__", [](const Tensor<T>& b, size_t i) {
        if (i >= b.size()) throw py::index_error();
        return b(i);
    })
    .def("__setitem__", [](Tensor<T>& b, size_t i, T v) {
        if (i >= b.size()) throw py::index_error();
        b(i) = v;
    })
    .def("__len__", [](BaseTensor& b) { return b.size(); })
    /// Optional sequence protocol operations
    .def("__iter__", [](const Tensor<T>& b) { return py::make_iterator(b.begin(), b.end()); },
                        py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
    .def("__contains__", [](const Tensor<T>& b, T v) {
        return (std::find(b.begin(), b.end(), v) != b.end());
    })
    .def("__reversed__", [](const Tensor<T>& b) -> Tensor<T> {
        std::vector<size_t> reversedDims(b.dims());
        std::reverse(reversedDims.begin(), reversedDims.end());

        std::vector<T> reversedData(b.begin(), b.end());
        std::reverse(reversedData.begin(), reversedData.end());

        return Tensor<T>(reversedDims,
                         reversedData.begin(), reversedData.end());
    })
    /// Slicing protocol (optional)
    .def("__getitem__", [](const Tensor<T>& b, py::slice slice) -> Tensor<T>* {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        Tensor<T>* t = new Tensor<T>({slicelength});
        for (size_t i = 0; i < slicelength; ++i) {
            (*t)(i) = b(start); start += step;
        }
        return t;
    })
    .def("__setitem__", [](Tensor<T>& b, py::slice slice, const Tensor<T>& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        if (slicelength != value.size())
            throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value(i); start += step;
        }
    })
    .def("__setitem__", [](Tensor<T>& b, py::slice slice, const T& value) {
        size_t start, stop, step, slicelength;
        if (!slice.compute(b.size(), &start, &stop, &step, &slicelength))
            throw py::error_already_set();
        for (size_t i = 0; i < slicelength; ++i) {
            b(start) = value; start += step;
        }
    });

    declare_Tensor_buffer_protocol(tensor);
}

void init_Tensor(py::module &m) {
    py::class_<BaseTensor>(m, "BaseTensor")
    .def("empty", &BaseTensor::empty)
    .def("dimX", &BaseTensor::dimX)
    .def("dimY", &BaseTensor::dimY)
    .def("dimD", &BaseTensor::dimD)
    .def("dimZ", &BaseTensor::dimZ)
    .def("dimB", &BaseTensor::dimB)
    .def("size", &BaseTensor::size)
    .def("reserve", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::reserve, py::arg("dims"))
    .def("resize", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::resize, py::arg("dims"))
    .def("reshape", (void (BaseTensor::*)(const std::vector<size_t>&)) &BaseTensor::reshape, py::arg("dims"))
    .def("clear", &BaseTensor::clear)
    .def("save", &BaseTensor::save, py::arg("data"))
    .def("load", &BaseTensor::load, py::arg("data"))
    .def("synchronizeDToH", (void (BaseTensor::*)() const) &BaseTensor::synchronizeDToH)
    .def("synchronizeHToD", (void (BaseTensor::*)() const) &BaseTensor::synchronizeHToD)
    .def("synchronizeDToHBased", &BaseTensor::synchronizeDToHBased)
    .def("synchronizeHBasedToD", &BaseTensor::synchronizeHBasedToD)
    .def("synchronizeDBasedToH", &BaseTensor::synchronizeDBasedToH)
    .def("synchronizeHToDBased", &BaseTensor::synchronizeHToDBased)
    .def("nbDims", &BaseTensor::nbDims)
    .def("dims", &BaseTensor::dims)
    .def("isValid", &BaseTensor::isValid)
    .def("setValid", &BaseTensor::setValid)
    .def("clearValid", &BaseTensor::clearValid)
    .def("getType", &BaseTensor::getType);

    declare_Tensor<float>(m, "float");
    declare_Tensor<double>(m, "double");
    declare_Tensor<char>(m, "char");
    declare_Tensor<unsigned char>(m, "unsigned char");
    declare_Tensor<short>(m, "short");
    declare_Tensor<int>(m, "int");
    declare_Tensor<unsigned int>(m, "unsigned int");
    declare_Tensor<long>(m, "long");
    declare_Tensor<unsigned long>(m, "unsigned long");
    declare_Tensor<long long>(m, "long long");
    declare_Tensor<unsigned long long>(m, "unsigned long long");
    declare_Tensor<bool>(m, "bool");
}
}
#endif
