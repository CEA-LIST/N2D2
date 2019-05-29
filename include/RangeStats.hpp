/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thieler@cea.fr)

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

#ifndef N2D2_RANGESTATS_H
#define N2D2_RANGESTATS_H

#include <string>
#include <vector>
#include <iostream>
#include <map>

namespace N2D2 {

class RangeStats {
public:
    RangeStats();
    double minVal() const { return mMinVal; }
    double maxVal() const { return mMaxVal; }
    const std::vector<double>& moments() const { return mMoments; }
    double mean() const;
    double stdDev() const;
    void operator()(double value);
    void save(std::ostream& state) const;
    void load(std::istream& state);

    static void saveOutputsRange(const std::string& fileName,
                               const std::map
                               <std::string, RangeStats>& outputsRange);
    static void loadOutputsRange(const std::string& fileName,
                               std::map<std::string, RangeStats>& outputsRange);
    static void logOutputsRange(const std::string& fileName,
                         const std::map
                         <std::string, RangeStats>& outputsRange);

private:
    double mMinVal;
    double mMaxVal;

    /**
     * mMoments[0] = nb values
     * mMoments[1] = sum of values
     * mMoments[2] = sum of squared values
     */
    std::vector<double> mMoments;
};
}

#endif // N2D2_RANGESTATS_H
