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

#include <cassert>
#include "RangeStats.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Utils.hpp"

N2D2::RangeStats::RangeStats()
    : mMinVal(0.0), mMaxVal(0.0), mMoments(3, 0.0)
{
    // ctor
}

double N2D2::RangeStats::mean() const
{
    return (mMoments[1] / mMoments[0]);
}

double N2D2::RangeStats::stdDev() const
{
    assert(mMoments.size() > 2);
    assert(mMoments[0] > 0.0);

    const double mean = (mMoments[1] / mMoments[0]);
    const double meanSquare = (mMoments[2] / mMoments[0]);
    return std::sqrt(meanSquare - mean * mean);
}

void N2D2::RangeStats::operator()(double value)
{
    if (mMoments[0] > 0) {
        mMinVal = std::min(mMinVal, value);
        mMaxVal = std::max(mMaxVal, value);
    } else {
        mMinVal = value;
        mMaxVal = value;
    }

    double inc = 1.0;

    for (std::vector<double>::iterator it = mMoments.begin(),
                                       itEnd = mMoments.end();
         it != itEnd;
         ++it) {
        (*it) += inc;
        inc *= value;
    }
}

void N2D2::RangeStats::save(std::ostream& state) const {
    state.write(reinterpret_cast<const char*>(&mMinVal), sizeof(mMinVal));
    state.write(reinterpret_cast<const char*>(&mMaxVal), sizeof(mMaxVal));
    const size_t momentsSize = mMoments.size();
    state.write(reinterpret_cast<const char*>(&momentsSize),
                sizeof(momentsSize));
    state.write(reinterpret_cast<const char*>(&mMoments[0]),
                momentsSize * sizeof(mMoments[0]));
}

void N2D2::RangeStats::load(std::istream& state) {
    state.read(reinterpret_cast<char*>(&mMinVal), sizeof(mMinVal));
    state.read(reinterpret_cast<char*>(&mMaxVal), sizeof(mMaxVal));
    size_t momentsSize;
    state.read(reinterpret_cast<char*>(&momentsSize), sizeof(momentsSize));
    mMoments.resize(momentsSize);
    state.read(reinterpret_cast<char*>(&mMoments[0]),
                momentsSize * sizeof(mMoments[0]));
}

void N2D2::RangeStats::saveOutputsRange(const std::string& fileName,
                               const std::unordered_map<std::string, RangeStats>& outputsRange)
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    std::ofstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not create state file: " + fileName);

    const size_t mapSize = outputsRange.size();
    state.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (auto it = outputsRange.begin(); it != outputsRange.end(); ++it) {
        const size_t nameSize = (*it).first.size();
        const std::string& nameStr = (*it).first;
        state.write(reinterpret_cast<const char*>(&nameSize), sizeof(nameSize));
        state.write(reinterpret_cast<const char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        (*it).second.save(state);
    }

    if (!state.good())
        throw std::runtime_error("Error writing state file: " + fileName);
}

void N2D2::RangeStats::loadOutputsRange(const std::string& fileName,
                                        std::unordered_map<std::string, RangeStats>& outputsRange)
{
    std::ifstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not open state file: " + fileName);

    size_t mapSize;
    state.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    for (size_t n = 0; n < mapSize; ++n) {
        size_t nameSize;
        state.read(reinterpret_cast<char*>(&nameSize), sizeof(nameSize));
        std::string nameStr(nameSize, ' ');
        state.read(reinterpret_cast<char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        std::unordered_map<std::string, RangeStats>::iterator it;
        std::tie(it, std::ignore)
            = outputsRange.insert(std::make_pair(nameStr, RangeStats()));

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

void N2D2::RangeStats::logOutputsRange(const std::string& fileName,
                            const std::unordered_map<std::string, RangeStats>& outputsRange)
{
    std::ofstream rangeData(fileName.c_str());

    if (!rangeData.good())
        throw std::runtime_error("Could not open range file: " + fileName);

    for (auto it = outputsRange.begin(); it != outputsRange.end(); ++it) {
        rangeData << (*it).first << " " << (*it).second.mMinVal
                  << " " << (*it).second.mMaxVal << " "
                  << (*it).second.mean() << " "
                  << (*it).second.stdDev() << "\n";
    }

    rangeData.close();

    Gnuplot gnuplot;
    gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
               "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    gnuplot.setYlabel("Range (min, mean, max) with std. dev.");
    gnuplot.setXrange(-0.5, outputsRange.size() - 0.5);
    gnuplot.set("grid");
    gnuplot.set("bmargin 4");
    gnuplot.set("boxwidth 0.2");
    gnuplot << "min(x,y) = (x < y) ? x : y";
    gnuplot << "max(x,y) = (x > y) ? x : y";
    gnuplot.unset("key");
    gnuplot.saveToFile(fileName);
    gnuplot.plot(
        fileName,
        "using ($4):xticlabels(wrap(stringcolumn(1),5)) lt 1,"
        " '' using 0:(max($2,$4-$5)):2:3:(min($3,$4+$5)) with candlesticks lt "
        "1 lw 2 whiskerbars,"
        " '' using 0:($3):($3) with labels offset char 0,1 textcolor lt 1,"
        " '' using 0:($2):($2) with labels offset char 0,-1 textcolor lt 3,"
        " '' using 0:($4):($4) with labels offset char 7,0 textcolor lt -1,"
        " '' using 0:4:4:4:4 with candlesticks lt -1 lw 2 notitle");
}
