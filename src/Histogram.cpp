/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#include "Histogram.hpp"
#include "utils/Utils.hpp"
#include "utils/Gnuplot.hpp"


N2D2::Histogram::Histogram(double minVal, double maxVal, std::size_t nbBins)
    : mMinVal(minVal), mMaxVal(maxVal),
      mNbBins(nbBins),
      mValues(nbBins, 0),
      mNbValues(0)
{
    if(mNbBins <= 0) {
        throw std::runtime_error("Number of bins must be > 0.");
    }

    if(mMaxVal <= mMinVal) {
        throw std::runtime_error("mMinVal must be strictly smaller than mMaxVal.");
    }
}

double N2D2::Histogram::getBinWidth() const {
    assert(mNbBins > 0);
    return (mMaxVal - mMinVal) / mNbBins;
}

double N2D2::Histogram::getBinValue(std::size_t binIdx) const {
    if(binIdx >= mNbBins) {
        throw std::out_of_range("binIdx must be < mNbBins");
    }

    return (mMinVal + (binIdx + 0.5) * getBinWidth());
}

std::size_t N2D2::Histogram::getNbBins() const {
    return mNbBins;
}

std::size_t N2D2::Histogram::getBinIdx(double value) const {
    assert(getBinWidth() > 0);
    
    const double clampedValue = Utils::clamp(value, mMinVal, mMaxVal);
    std::size_t binIdx = static_cast<std::size_t>((clampedValue - mMinVal) / getBinWidth() + 1e-6);
    if(binIdx == mNbBins) {
        binIdx--;
    }

    return binIdx;
}

double N2D2::Histogram::getMinVal() const {
    return mMinVal;
}

double N2D2::Histogram::getMaxVal() const {
    return mMaxVal;
}

const std::vector<std::size_t>& N2D2::Histogram::getBins() const {
    return mValues;
}

void N2D2::Histogram::operator()(double value, std::size_t count) {
    if(value > mMaxVal || value < mMinVal) {
        throw std::out_of_range(std::to_string(value) + " not between [" + 
                                    std::to_string(mMinVal) + ";" + 
                                    std::to_string(mMaxVal) + 
                                "]");
    }

    mValues[getBinIdx(value)] += count;
    mNbValues += count;
}

void N2D2::Histogram::enlarge(double value, bool symetric) {
    const double currBinWidth = getBinWidth();

    
    std::size_t newNbBins;
    if(value < mMinVal) {
        newNbBins = static_cast<std::size_t>(std::ceil((mMinVal - value) / getBinWidth()));
    }
    else if(value > mMaxVal) {
        newNbBins = static_cast<std::size_t>(std::ceil((value - mMaxVal) / getBinWidth()));
    }
    else {
        return;
    }

    
    if(symetric || value < mMinVal) {
        mValues.insert(mValues.begin(), newNbBins, 0);
        mMinVal = mMinVal - newNbBins*currBinWidth;
    }
    
    if(symetric || value > mMaxVal) {
        mValues.insert(mValues.end(), newNbBins, 0);
        mMaxVal = mMaxVal + newNbBins*currBinWidth;
    }

    mNbBins = mValues.size();
}

void N2D2::Histogram::log(const std::string& fileName,
                          const std::unordered_map<std::string, double>& thresholds) const
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    std::ofstream histData(fileName.c_str());

    if (!histData.good()) {
        throw std::runtime_error("Could not open histogram file: " + fileName);
    }

    for (std::size_t bin = 0; bin < mNbBins; ++bin) {
        histData << bin << " " << getBinValue(bin) << " " << mValues[bin]
            << " " << ((mNbValues != 0)?(mValues[bin] / (double)mNbValues):0) << "\n";
    }

    histData.close();

    Gnuplot gnuplot;
    gnuplot.setXlabel("Output value");
    gnuplot.setYlabel("Normalized number of counts");
    gnuplot.set("grid");
    gnuplot.set("key off");
    gnuplot.set("logscale y");

    std::size_t i = 0;

    for (auto it = thresholds.begin(); it != thresholds.end(); ++it) {
        std::stringstream cmdStr;
        cmdStr << "arrow from " << (*it).second << ", "
            "graph 0 to " << (*it).second << ", graph 1 nohead "
            "lt " << (3 + i) << " lw 2";
        gnuplot.set(cmdStr.str());

        cmdStr.str(std::string());
        cmdStr << "label " << (i + 1) << " \" " << (*it).first << " = "
            << (*it).second << "\"";
        gnuplot.set(cmdStr.str());

        cmdStr.str(std::string());
        cmdStr << "label " << (i + 1) << " at " << (*it).second
            << ", graph " << (0.85 - i * 0.05) << " tc lt " << (3 + i);
        gnuplot.set(cmdStr.str());

        ++i;
    }

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName, "using 2:4 with points");
}

double N2D2::Histogram::calibrateMSE(std::size_t nbBits) const {
    if(mNbValues == 0) {
        return 0.0;
    }

    double threshold = Utils::max_abs(mMinVal, mMaxVal);
    double bestThreshold = threshold;
    double bestMSE = std::numeric_limits<double>::max();

    const double threshold_decr_step = threshold/1000.0;
    while(threshold > 0.0) {
        const double mse = MSE(threshold, nbBits);
        if(mse < bestMSE) {
            bestMSE = mse;
            bestThreshold = threshold;
        }
        
        threshold -= threshold_decr_step;
    }

    return bestThreshold;
}

double N2D2::Histogram::MSE(double threshold, std::size_t nbBits) const {
    assert(nbBits > 1);

    const bool isUnsigned = mMinVal >= 0.0;

    const double minVal = isUnsigned?0:-(1LL << (nbBits - 1));
    const double maxVal = isUnsigned?((1ULL << nbBits) - 1):((1ULL << (nbBits - 1)) - 1);
    const double scaling = maxVal/threshold;


    double mse = 0.0;
    for(std::size_t bin = 0; bin < mNbBins; bin++) {
        const double approx = Utils::clamp(std::round(getBinValue(bin)*scaling), minVal, maxVal)/scaling;
        const double normalizedValue = 1.0*mValues[bin]/mNbValues;

        mse += std::pow(getBinValue(bin) - approx, 2) * normalizedValue;
    }

    return mse;
}


double N2D2::Histogram::calibrateKLDivergence(std::size_t nbBits) const {
    if(mNbValues == 0) {
        return 0.0;
    }

    const bool isUnsigned = mMinVal >= 0.0;
    const std::size_t nbQuantizedBins = static_cast<std::size_t>(1) << nbBits;

    double threshold = Utils::max_abs(mMinVal, mMaxVal);
    double bestThreshold = threshold;
    double bestDivergence = std::numeric_limits<double>::max();;

    const double threshold_decr_step = threshold/1000.0;
    while(threshold > 0.0) {
        const double divergence = KLDivergence(*this, quantize(isUnsigned?0:-threshold, 
                                                               threshold, nbQuantizedBins));
        if(divergence < bestDivergence) {
            bestDivergence = divergence;
            bestThreshold = threshold;
        }

        threshold -= threshold_decr_step;
    }

    return bestThreshold;
}

double N2D2::Histogram::getQuantileValue(double quantile) const {
    const size_t quantileNbValues = quantile * mNbValues;
    size_t nbValues = 0;

    for (std::size_t bin = 0; bin < mNbBins; ++bin) {
        nbValues += mValues[bin];

        if (nbValues >= quantileNbValues)
            return getBinValue(bin);
    }

    return getBinValue(mNbBins - 1);
}

N2D2::Histogram N2D2::Histogram::quantize(double newMinVal, double newMaxVal,
                                          std::size_t newNbBins) const
{
    Histogram hist(newMinVal, newMaxVal, newNbBins);

    for (std::size_t bin = 0; bin < mNbBins; ++bin) {
        const double value = Utils::clamp(getBinValue(bin), hist.mMinVal, hist.mMaxVal);
        hist(value, mValues[bin]);
    }

    assert(mNbValues == hist.mNbValues);
    return hist;
}

double N2D2::Histogram::KLDivergence(const Histogram& ref, const Histogram& quant) {
    assert(ref.mNbValues == quant.mNbValues);
    assert(ref.mMaxVal >= quant.mMaxVal);

    double qNorm = 0.0;

    for (std::size_t bin = 0; bin < ref.mNbBins; ++bin) {
        const std::size_t quantIdx = quant.getBinIdx(ref.getBinValue(bin));
        qNorm += quant.mValues[quantIdx];
    }

    double divergence = 0.0;

    for (std::size_t bin = 0; bin < ref.mNbBins; ++bin) {
        const std::size_t quantIdx = quant.getBinIdx(ref.getBinValue(bin));

        // Sum of p and q over bin [0,ref.mNbBins) must be 1
        const double p = (ref.mValues[bin] / (double)ref.mNbValues);
        const double q = (quant.mValues[quantIdx] / qNorm);

        // q = 0 => p = 0
        // p = 0 => contribution = 0
        if (p != 0) {
            assert(q != 0);  // q = 0 => p = 0
            divergence += p * std::log(p / q);
        }
    }

    return divergence;
}

void N2D2::Histogram::save(std::ostream& state) const {
    state.write(reinterpret_cast<const char*>(&mMinVal), sizeof(mMinVal));
    state.write(reinterpret_cast<const char*>(&mMaxVal), sizeof(mMaxVal));
    state.write(reinterpret_cast<const char*>(&mNbBins), sizeof(mNbBins));
    const std::size_t valuesSize = mValues.size();
    state.write(reinterpret_cast<const char*>(&valuesSize), sizeof(valuesSize));
    state.write(reinterpret_cast<const char*>(&mValues[0]),
                valuesSize * sizeof(mValues[0]));
    state.write(reinterpret_cast<const char*>(&mNbValues), sizeof(mNbValues));
}

void N2D2::Histogram::load(std::istream& state) {
    state.read(reinterpret_cast<char*>(&mMinVal), sizeof(mMinVal));
    state.read(reinterpret_cast<char*>(&mMaxVal), sizeof(mMaxVal));
    state.read(reinterpret_cast<char*>(&mNbBins), sizeof(mNbBins));
    std::size_t valuesSize;
    state.read(reinterpret_cast<char*>(&valuesSize), sizeof(valuesSize));
    mValues.resize(valuesSize);
    state.read(reinterpret_cast<char*>(&mValues[0]),
                valuesSize * sizeof(mValues[0]));
    state.read(reinterpret_cast<char*>(&mNbValues), sizeof(mNbValues));
}

void N2D2::Histogram::saveOutputsHistogram(const std::string& fileName,
                        const std::unordered_map<std::string, Histogram>& outputsHistogram)
{
    std::ofstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not create state file: " + fileName);

    const std::size_t mapSize = outputsHistogram.size();
    state.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (auto it = outputsHistogram.begin(); it != outputsHistogram.end(); ++it) {
        const std::size_t nameSize = (*it).first.size();
        const std::string& nameStr = (*it).first;
        state.write(reinterpret_cast<const char*>(&nameSize), sizeof(nameSize));
        state.write(reinterpret_cast<const char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        (*it).second.save(state);
    }

    if (!state.good())
        throw std::runtime_error("Error writing state file: " + fileName);
}

void N2D2::Histogram::loadOutputsHistogram(const std::string& fileName,
                        std::unordered_map<std::string, Histogram>& outputsHistogram)
{
    std::ifstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not open state file: " + fileName);

    std::size_t mapSize;
    state.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    for (size_t n = 0; n < mapSize; ++n) {
        std::size_t nameSize;
        state.read(reinterpret_cast<char*>(&nameSize), sizeof(nameSize));
        std::string nameStr(nameSize, ' ');
        state.read(reinterpret_cast<char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        std::unordered_map<std::string, Histogram>::iterator it;
        std::tie(it, std::ignore)
            = outputsHistogram.insert(std::make_pair(nameStr, Histogram(0, 1, 1)));

        (*it).second.load(state);
    }

    if (state.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in state file: "
            + fileName);
    else if (!state.good())
        throw std::runtime_error("Error while reading state file: "
                                 + fileName);
    else if (state.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "State file size larger than expected: " + fileName);
}

void N2D2::Histogram::logOutputsHistogram(const std::string& dirName,
                        const std::unordered_map<std::string, Histogram>& outputsHistogram,
                        std::size_t nbBits, ClippingMode clippingMode, double quantileValue)
{
    Utils::createDirectories(dirName);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)outputsHistogram.size(); i++) {
        auto it = outputsHistogram.begin();
        std::advance(it, i);

        std::unordered_map<std::string, double> thresholds;

        Histogram hist = (*it).second;
        if(clippingMode == ClippingMode::KL_DIVERGENCE) {
            thresholds["KL"] = hist.calibrateKLDivergence(nbBits);
        }
        else if(clippingMode == ClippingMode::MSE) {
            thresholds["MSE"] = hist.calibrateMSE(nbBits);
        }
        else if(clippingMode == ClippingMode::QUANTILE) {
            thresholds["QUANTILE"] = hist.getQuantileValue(quantileValue);
        }else{
            throw std::runtime_error("In Histogram::logOutputsHistogram : Unknown clipping mode encountered.");
        }

        (*it).second.log(dirName + "/" + Utils::filePath((*it).first) + ".dat", thresholds);
    }
}
