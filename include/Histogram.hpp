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

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace N2D2 {

enum class ClippingMode {
    NONE,
    MSE,
    KL_DIVERGENCE
};

inline ClippingMode parseClippingMode(const std::string& str) {
    if(str == "None") {
        return ClippingMode::NONE;
    }

    if(str == "MSE") {
        return ClippingMode::MSE;
    }

    if(str == "KL-Divergence") {
        return ClippingMode::KL_DIVERGENCE;
    }

    throw std::runtime_error("Unknown clipping mode '" + str + "'.");
}

inline std::size_t getNbBinsForClippingMode(std::size_t nbBits, 
                                            ClippingMode clippingMode)
{
    switch (clippingMode) {
        case ClippingMode::MSE:
            return std::min(1 << nbBits, 65536);
        case ClippingMode::KL_DIVERGENCE:
            return std::min((1 << nbBits)*32, 65536);
        default:
            return 0;
    }
}

class Histogram {
public:
    Histogram(double minVal, double maxVal, std::size_t nbBins);

    void operator()(double value, std::size_t count = 1);

    std::size_t getNbBins() const;
    double getBinWidth() const;
    double getBinValue(std::size_t binIdx) const;
    std::size_t getBinIdx(double value) const;

    double getMinVal() const;
    double getMaxVal() const;

    const std::vector<std::size_t>& getBins() const;

    void enlarge(double value);
    void truncate(double value);

    double calibrateMSE(std::size_t nbBits) const;
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

    double MSE(const std::vector<double>& nrmNbValues, const std::vector<double>& midBinValues, 
               double threshold, std::size_t nbBits, bool isUnsigned) const;
                
    Histogram quantize(double newMinVal,
                       double newMaxVal,
                       std::size_t newNbBins) const;
private:
    double mMinVal;
    double mMaxVal;
    std::size_t mNbBins;
    std::vector<std::size_t> mValues;
    std::size_t mNbValues;
    std::size_t mMaxBin;

};
}

#endif // N2D2_HISTOGRAM_H
