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

enum class ClippingMode {
    NONE,
    KL_DIVERGENCE
};

inline ClippingMode parseClippingMode(const std::string& str) {
    if(str == "None") {
        return ClippingMode::NONE;
    }

    if(str == "KL-Divergence") {
        return ClippingMode::KL_DIVERGENCE;
    }

    throw std::runtime_error("Unknown clipping mode '" + str + "'.");
}

inline unsigned int getNbBinsForClippingMode(std::size_t nbBits, 
                                             ClippingMode clippingMode)
{
    switch (clippingMode) {
        case ClippingMode::KL_DIVERGENCE:
            return std::min((1 << nbBits)*32, 65536);
        default:
            return 0;
    }
}

class Histogram {
public:
    Histogram() = default;
    Histogram(double minVal, double maxVal, unsigned int nbBins);

    void operator()(double value, unsigned long long int count = 1);

    unsigned int enlarge(double value);
    unsigned int truncate(double value);

    std::size_t getNbBins() const;
    double getBinWidth() const;
    double getBinValue(unsigned int binIdx) const;
    unsigned int getBinIdx(double value) const;

    double calibrateKLDivergence(std::size_t nbBits) const;
    
    void save(std::ostream& state) const;
    void load(std::istream& state);


    void log(const std::string& fileName,
             const std::unordered_map<std::string, double>& thresholds
                = std::unordered_map<std::string, double>()) const;

    static void saveOutputsHistogram(const std::string& fileName,
                    const std::unordered_map<std::string, Histogram>& outputsHistogram);
    static void loadOutputsHistogram(const std::string& fileName,
                    std::unordered_map<std::string, Histogram>& outputsHistogram);
    static void logOutputsHistogram(const std::string& fileName,
                    const std::unordered_map<std::string, Histogram>& outputsHistogram,
                    std::size_t nbBits);

private:
    static double KLDivergence(const Histogram& ref, const Histogram& quant);

    Histogram quantize(double newMinVal,
                       double newMaxVal,
                       unsigned int newNbBins) const;
private:
    double mMinVal;
    double mMaxVal;
    unsigned int mNbBins;
    std::vector<unsigned long long int> mValues;
    unsigned long long int mNbValues;
    unsigned int mMaxBin;

};
}

inline double N2D2::Histogram::getBinWidth() const {
    return ((mMaxVal - mMinVal) / (double)mNbBins);
}

inline double N2D2::Histogram::getBinValue(unsigned int binIdx) const {
    return (mMinVal + (binIdx + 0.5) * getBinWidth());
}

inline std::size_t N2D2::Histogram::getNbBins() const {
    return mNbBins;
}

#endif // N2D2_HISTOGRAM_H
