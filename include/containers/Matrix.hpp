/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_MATRIX_H
#define N2D2_MATRIX_H

#include <cassert>
#include <cctype>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "utils/Gnuplot.hpp"

namespace N2D2 {
/**
 * The Matrix class is like a 2D STL vector and is intended to simplify the
 *handling of 2D matrices.
 * It uses *row-major storage*.
 *
 * This class is not intended for high performance computing. To do so, use a
 *linear algebra library instead (such as Eigen or
 * BLAS libraries).
*/
template <class T> class Matrix {
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Matrix() : mNbRows(0), mNbCols(0)
    {
    }
    Matrix(unsigned int nbRows) : mNbRows(nbRows), mNbCols(1), mData(nbRows)
    {
    }
    Matrix(unsigned int nbRows, unsigned int nbCols, const T& value = T())
        : mNbRows(nbRows), mNbCols(nbCols), mData(nbRows * nbCols, value)
    {
    }
    template <typename InputIterator>
    Matrix(unsigned int nbRows,
           unsigned int nbCols,
           InputIterator first,
           InputIterator last);
    bool empty() const
    {
        return mData.empty();
    }
    unsigned int rows() const
    {
        return mNbRows;
    }
    unsigned int cols() const
    {
        return mNbCols;
    }
    unsigned int size() const
    {
        return mData.size();
    }
    iterator begin()
    {
        return mData.begin();
    }
    const_iterator begin() const
    {
        return mData.begin();
    }
    iterator end()
    {
        return mData.end();
    }
    const_iterator end() const
    {
        return mData.end();
    }
    inline void reserve(unsigned int nbRows, unsigned int nbCols);
    inline void
    resize(unsigned int nbRows, unsigned int nbCols, const T& value = T());
    inline void
    assign(unsigned int nbRows, unsigned int nbCols, const T& value);
    inline void clear();
    inline void swap(Matrix<T>& matrix);
    inline std::vector<T> row(unsigned int i) const;
    inline std::vector<T> col(unsigned int j) const;
    inline Matrix<T>
    block(unsigned int i, unsigned int j, unsigned int p, unsigned int q) const;
    inline void insertRow(unsigned int i, const std::vector<T>& row);
    inline void insertCol(unsigned int j, const std::vector<T>& col);
    void appendRow(const std::vector<T>& row)
    {
        insertRow(mNbRows, row);
    }
    void appendCol(const std::vector<T>& col)
    {
        insertCol(mNbCols, col);
    }
    inline void eraseRow(unsigned int i);
    inline void eraseCol(unsigned int j);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    inline reference operator()(unsigned int i, unsigned int j);
    inline const_reference operator()(unsigned int i, unsigned int j) const;
    inline reference operator()(unsigned int index);
    inline const_reference operator()(unsigned int index) const;
    reference at(unsigned int i, unsigned int j)
    {
        return mData.at(i * mNbCols + j);
    }
    const_reference at(unsigned int i, unsigned int j) const
    {
        return mData.at(i * mNbCols + j);
    }
    reference at(unsigned int index)
    {
        return mData.at(index);
    }
    const_reference at(unsigned int index) const
    {
        return mData.at(index);
    }

    friend bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        return (lhs.mData == rhs.mData);
    }
    friend bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        return (lhs.mData != rhs.mData);
    }

    template <class U>
    friend Matrix<U>& operator<<(Matrix<U>& matrix, const std::string& data);
    template <class U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& matrix);
    template <class U>
    friend std::istream& operator>>(std::istream& is, Matrix<U>& matrix);

    Matrix(const cv::Mat& mat);
    inline operator cv::Mat() const;
    inline std::vector<T>& data()
    {
        return mData;
    };
    inline const std::vector<T>& data() const
    {
        return mData;
    };
    inline operator std::vector<T>&();
    inline operator const std::vector<T>&() const;
    void log(const std::string& fileName) const;
    virtual ~Matrix()
    {
    }

protected:
    unsigned int mNbRows;
    unsigned int mNbCols;
    std::vector<T> mData;
};
}

template <class T>
template <typename InputIterator>
N2D2::Matrix<T>::Matrix(unsigned int nbRows,
                        unsigned int nbCols,
                        InputIterator first,
                        InputIterator last)
    : mNbRows(nbRows), mNbCols(nbCols), mData(first, last)
{
    // ctor
    if (mNbRows * mNbCols != mData.size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
void N2D2::Matrix<T>::reserve(unsigned int nbRows, unsigned int nbCols)
{
    mNbRows = nbRows;
    mNbCols = nbCols;
    mData.reserve(nbRows * nbCols);
}

template <class T>
void N2D2::Matrix
    <T>::resize(unsigned int nbRows, unsigned int nbCols, const T& value)
{
    mNbRows = nbRows;
    mNbCols = nbCols;
    mData.resize(nbRows * nbCols, value);
}

template <class T>
void N2D2::Matrix
    <T>::assign(unsigned int nbRows, unsigned int nbCols, const T& value)
{
    mNbRows = nbRows;
    mNbCols = nbCols;
    mData.assign(nbRows * nbCols, value);
}

template <class T> void N2D2::Matrix<T>::clear()
{
    mNbRows = 0;
    mNbCols = 0;
    mData.clear();
}

template <class T> void N2D2::Matrix<T>::swap(Matrix<T>& matrix)
{
    std::swap(mNbRows, matrix.mNbRows);
    std::swap(mNbCols, matrix.mNbCols);
    mData.swap(matrix.mData);

    assert(mData.size() == mNbRows * mNbCols);
    assert(matrix.mData.size() == matrix.mNbRows * matrix.mNbCols);
}

template <class T> std::vector<T> N2D2::Matrix<T>::row(unsigned int i) const
{
    assert(i < mNbRows);

    // Assume row-major storage
    return std::vector
        <T>(mData.begin() + i * mNbCols, mData.begin() + (i + 1) * mNbCols);
}

template <class T> std::vector<T> N2D2::Matrix<T>::col(unsigned int j) const
{
    assert(j < mNbCols);

    std::vector<T> col;
    col.reserve(mNbRows);

    for (unsigned int i = 0; i < mNbRows; ++i)
        col.push_back(mData[i * mNbCols + j]);

    assert(col.size() == mNbRows);
    return col;
}

template <class T>
N2D2::Matrix<T> N2D2::Matrix<T>::block(unsigned int i,
                                       unsigned int j,
                                       unsigned int p,
                                       unsigned int q) const
{
    assert(i <= mNbRows);
    assert(j <= mNbCols);
    assert(p <= mNbRows - i);
    assert(q <= mNbCols - j);

    Matrix<T> block;
    block.reserve(p, q);

    for (unsigned int bi = 0; bi < p; ++bi) {
        for (unsigned int bj = 0; bj < q; ++bj) {
            // [bi*q + bj] in strict ascending order -> push_back()
            block.mData.push_back(mData[(i + bi) * mNbCols + (j + bj)]);
        }
    }

    assert(block.mData.size() == p * q);
    return block;
}

template <class T>
void N2D2::Matrix<T>::insertRow(unsigned int i, const std::vector<T>& row)
{
    assert(i <= mNbRows);

    if (row.size() != mNbCols)
        throw std::out_of_range("Vector size doesn't match matrix row size!");

    ++mNbRows;
    // Assume row-major storage
    mData.insert(mData.begin() + i * mNbCols, row.begin(), row.end());
    assert(mData.size() == mNbRows * mNbCols);
}

template <class T>
void N2D2::Matrix<T>::insertCol(unsigned int j, const std::vector<T>& col)
{
    assert(j <= mNbCols);

    if (col.size() != mNbRows)
        throw std::out_of_range(
            "Vector size doesn't match matrix column size!");

    ++mNbCols;

    for (unsigned int i = 0; i < mNbRows; ++i)
        mData.insert(mData.begin() + i * mNbCols + j, col[i]);

    assert(mData.size() == mNbRows * mNbCols);
}

template <class T> void N2D2::Matrix<T>::eraseRow(unsigned int i)
{
    assert(i < mNbRows);
    assert(mNbRows > 0);

    --mNbRows;
    // Assume row-major storage
    mData.erase(mData.begin() + i * mNbCols, mData.begin() + (i + 1) * mNbCols);
    assert(mData.size() == mNbRows * mNbCols);
}

template <class T> void N2D2::Matrix<T>::eraseCol(unsigned int j)
{
    assert(j < mNbCols);
    assert(mNbCols > 0);

    --mNbCols;

    for (unsigned int i = 0; i < mNbRows; ++i)
        mData.erase(mData.begin() + i * mNbCols + j);

    assert(mData.size() == mNbRows * mNbCols);
}

template <class T>
typename N2D2::Matrix<T>::reference N2D2::Matrix<T>::operator()(unsigned int i,
                                                                unsigned int j)
{
    assert(i < mNbRows);
    assert(j < mNbCols);

    return mData[i * mNbCols + j];
}

template <class T>
typename N2D2::Matrix<T>::const_reference N2D2::Matrix<T>::
operator()(unsigned int i, unsigned int j) const
{
    assert(i < mNbRows);
    assert(j < mNbCols);

    return mData[i * mNbCols + j];
}

template <class T>
typename N2D2::Matrix<T>::reference N2D2::Matrix<T>::
operator()(unsigned int index)
{
    assert(index < mData.size());

    return mData[index];
}

template <class T>
typename N2D2::Matrix<T>::const_reference N2D2::Matrix<T>::
operator()(unsigned int index) const
{
    assert(index < mData.size());

    return mData[index];
}

namespace N2D2 {
template <class T>
N2D2::Matrix<T>& operator<<(Matrix<T>& matrix, const std::string& data)
{
    std::stringstream dataStr(data);

    if (!(dataStr >> matrix))
        throw std::runtime_error("Missing value or unreadable data.");

    // Discard trailing whitespaces
    while (std::isspace(dataStr.peek()))
        dataStr.ignore();

    if (dataStr.get() != std::stringstream::traits_type::eof())
        throw std::runtime_error("Unread additional data remaining.");

    return matrix;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    for (unsigned int i = 0; i < matrix.mNbRows; ++i) {
        // Assume row-major storage
        std::copy(matrix.begin() + i * matrix.mNbCols,
                  matrix.begin() + (i + 1) * matrix.mNbCols,
                  std::ostream_iterator<T>(os, " "));

        os << "\n";
    }

    return os;
}

template <class T> std::istream& operator>>(std::istream& is, Matrix<T>& matrix)
{
    T value;

    // Assume row-major storage
    for (typename Matrix<T>::iterator it = matrix.mData.begin(),
                                      itEnd = matrix.mData.end();
         it != itEnd;
         ++it) {
        if (!(is >> value))
            throw std::runtime_error("Missing value or unreadable data.");

        (*it) = value;
    }

    return is;
}
}

template <class T>
N2D2::Matrix<T>::Matrix(const cv::Mat& mat)
    : mNbRows(mat.rows), mNbCols(mat.cols)
{
    if (mat.channels() != 1 || mat.elemSize() != sizeof(T))
        throw std::runtime_error("Incompatible types.");

    mData.reserve(mNbRows * mNbCols);

    for (int i = 0; i < mat.rows; ++i) {
        const T* rowPtr = mat.ptr<T>(i);
        mData.insert(mData.end(), rowPtr, rowPtr + mat.cols);
    }
}

template <class T> N2D2::Matrix<T>::operator cv::Mat() const
{
#if CV_MINOR_VERSION < 4
    // Segfault in some cases without copy...
    return cv::Mat(mData, true).reshape(0, mNbRows);
#else
    return cv::Mat(mData).reshape(0, mNbRows);
#endif
}

template <class T> N2D2::Matrix<T>::operator std::vector<T>&()
{
    if (mNbRows != 1 && mNbCols != 1)
        throw std::runtime_error("Cannot convert 2D matrix into std::vector.");

    return mData;
}

template <class T> N2D2::Matrix<T>::operator const std::vector<T>&() const
{
    if (mNbRows != 1 && mNbCols != 1)
        throw std::runtime_error("Cannot convert 2D matrix into std::vector.");

    return mData;
}

template <class T> void N2D2::Matrix<T>::log(const std::string& fileName) const
{
    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error(
            "Matrix<T>::log(): Could not create data log file: " + fileName);

    T minVal = (!mData.empty()) ? mData[0] : 0;
    T maxVal = (!mData.empty()) ? mData[0] : 0;

    for (unsigned int i = 0; i < mNbRows; ++i) {
        for (unsigned int j = 0; j < mNbCols; ++j) {
            const T value = (*this)(i, j);
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);

            dataFile << value << " ";
        }

        dataFile << "\n";
    }

    dataFile.close();

    Gnuplot gnuplot;
    gnuplot.set("grid").set("key off");
    gnuplot.set("size ratio 1");
    gnuplot.setXrange(-0.5, mNbCols - 0.5);
    gnuplot.setYrange(-0.5, mNbRows - 0.5, "reverse");
    gnuplot.set("xtics out nomirror");
    gnuplot.set("ytics out nomirror");

    std::stringstream cbRangeStr, paletteStr;
    cbRangeStr << "cbrange [";
    paletteStr << "palette defined (";

    if (minVal < -1.0) {
        cbRangeStr << minVal;
        paletteStr << minVal << " \"blue\", -1 \"cyan\", ";
    } else if (minVal < 0.0) {
        cbRangeStr << minVal;
        paletteStr << minVal << " \"cyan\", ";
    } else
        cbRangeStr << 0.0;

    cbRangeStr << ":";
    paletteStr << "0 \"black\"";

    if (maxVal > 1.0) {
        cbRangeStr << maxVal;
        paletteStr << ", 1 \"white\", " << maxVal << " \"red\"";
    } else if (maxVal > 0.0 || !(minVal < 0)) {
        cbRangeStr << maxVal;
        paletteStr << ", " << maxVal << " \"white\"";
    } else
        cbRangeStr << 0.0;

    cbRangeStr << "]";
    paletteStr << ")";

    gnuplot.set(paletteStr.str());
    gnuplot.set(cbRangeStr.str());

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName, "matrix with image");
}

#endif // N2D2_MATRIX_H
