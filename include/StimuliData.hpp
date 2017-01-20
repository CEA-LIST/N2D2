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

#ifndef N2D2_STIMULIDATA_H
#define N2D2_STIMULIDATA_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "StimuliProvider.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {
class StimuliData : public Parameterizable {
public:
    struct Size {
        Size(unsigned int dimX_ = 0,
             unsigned int dimY_ = 0,
             unsigned int dimZ_ = 0)
            : dimX(dimX_), dimY(dimY_), dimZ(dimZ_)
        {
        }
        unsigned int dimX;
        unsigned int dimY;
        unsigned int dimZ;
    };

    struct Value {
        Value(Float_T minVal_ = 0.0,
              Float_T maxVal_ = 0.0,
              double mean_ = 0.0,
              double stdDev_ = 0.0)
            : minVal(minVal_), maxVal(maxVal_), mean(mean_), stdDev(stdDev_)
        {
        }
        Float_T minVal;
        Float_T maxVal;
        double mean;
        double stdDev;
    };

    StimuliData(const std::string& name, StimuliProvider& provider);
    void generate(Database::StimuliSetMask setMask = Database::All);
    void displayData() const;
    void clear();

    // Getter
    const Size& getMinSize() const
    {
        return mMinSize;
    }
    const Size& getMaxSize() const
    {
        return mMaxSize;
    }
    Size getMeanSize() const;
    const Value& getGlobalValue() const
    {
        return mGlobalValue;
    }

    // Log
    void logSizeRange() const;
    void logValueRange() const;
    virtual ~StimuliData()
    {
    }

private:
    bool loadDataCache(const std::string& fileName);
    void saveDataCache(const std::string& fileName) const;

    const std::string mName;
    StimuliProvider& mProvider;

    // Parameters
    Parameter<bool> mMeanData;

    // Per-stimulus size
    std::vector<Size> mSize;
    // Global size stats
    Size mMinSize;
    Size mMaxSize;

    // Per-stimulus stats
    std::vector<Value> mValue;
    // Global value stats
    Value mGlobalValue;
};
}

#endif // N2D2_STIMULIDATA_H
