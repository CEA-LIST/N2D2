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

#ifndef N2D2_HISTOGRAM_H
#define N2D2_HISTOGRAM_H

#include <string>
#include <vector>
#include <iostream>
#include <map>

namespace N2D2 {

class Histogram {
public:
    Histogram(double minVal_ = 0.0,
              double maxVal_ = 1000.0,
              unsigned int nbBins_ = 100000);
    void operator()(double value, unsigned long long int count = 1);
    unsigned int enlarge(double value);
    unsigned int truncate(double value);
    inline double getBinWidth() const;
    inline double getBinValue(unsigned int binIdx) const;
    unsigned int getBinIdx(double value) const;
    void log(const std::string& fileName,
             const std::map<std::string, double>& thresholds
                = std::map<std::string, double>()) const;
    Histogram quantize(double newMaxVal,
                       unsigned int newNbBins = 128) const;
    double calibrateKL(unsigned int nbLevels = 128,
                       double maxError = 1.0e-3,
                       unsigned int maxIters = 100) const;
    void save(std::ostream& state) const;
    void load(std::istream& state);

    static double KLDivergence(const Histogram& ref,
                               const Histogram& quant);
    static void saveOutputsHistogram(const std::string& fileName,
                         const std::map
                         <std::string, Histogram>& outputsHistogram);
    static void loadOutputsHistogram(const std::string& fileName,
                         std::map<std::string, Histogram>& outputsHistogram);
    static void logOutputsHistogram(const std::string& fileName,
                         const std::map
                         <std::string, Histogram>& outputsHistogram,
                         unsigned int nbLevels = 128);

private:
    double mMinVal;
    double mMaxVal;
    unsigned int mNbBins;
    std::vector<unsigned long long int> mValues;
    unsigned long long int mNbValues;
    unsigned int mMaxBin;

};
}

double N2D2::Histogram::getBinWidth() const
{
    return ((mMaxVal - mMinVal) / (double)mNbBins);
}

double N2D2::Histogram::getBinValue(unsigned int binIdx) const
{
    return (mMinVal + (binIdx + 0.5) * getBinWidth());
}

#endif // N2D2_HISTOGRAM_H
