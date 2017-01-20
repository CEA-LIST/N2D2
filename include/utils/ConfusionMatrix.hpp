/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CONFUSIONMATRIX_H
#define N2D2_CONFUSIONMATRIX_H

#include "containers/Matrix.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
class ConfusionMatrix : public Matrix<unsigned int> {
public:
    void log(const std::string& fileName,
             const std::vector<std::string>& labels = std::vector
             <std::string>()) const;
};

class ConfusionTable {
public:
    ConfusionTable() : mTp(0), mTn(0), mFp(0), mFn(0) {};
    double precision() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFp)) : 0.0;
    };
    double recall() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFn)) : 0.0;
    };
    double F1Score() const
    {
        return (mTp > 0) ? (2 * mTp / (double)(2 * mTp + mFp + mFn)) : 0.0;
    };
    void tp(unsigned int tp)
    {
        mTp += tp;
    };
    void tn(unsigned int tn)
    {
        mTn += tn;
    };
    void fp(unsigned int fp)
    {
        mFp += fp;
    };
    void fn(unsigned int fn)
    {
        mFn += fn;
    };
    unsigned int tp() const
    {
        return mTp;
    };
    unsigned int tn() const
    {
        return mTn;
    };
    unsigned int fp() const
    {
        return mFp;
    };
    unsigned int fn() const
    {
        return mFn;
    };

private:
    unsigned int mTp;
    unsigned int mTn;
    unsigned int mFp;
    unsigned int mFn;
};
}

#endif // N2D2_CONFUSIONMATRIX_H
