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

#include "StimuliData.hpp"

N2D2::StimuliData::StimuliData(const std::string& name,
                               StimuliProvider& provider)
    : mName(name), mProvider(provider), mMeanData(this, "MeanData", false)
{
    // ctor
    Utils::createDirectories(mName);
}

void N2D2::StimuliData::displayData() const
{
    std::stringstream dataStr;
    dataStr << mName << " data:\n"
                        "Number of stimuli: " << mSize.size()
            << "\n"
               "Data width range: [" << mMinSize.dimX << ", " << mMaxSize.dimX
            << "]\n"
               "Data height range: [" << mMinSize.dimY << ", " << mMaxSize.dimY
            << "]\n"
               "Data channels range: [" << mMinSize.dimZ << ", "
            << mMaxSize.dimZ << "]\n"
                                "Value range: [" << mGlobalValue.minVal << ", "
            << mGlobalValue.maxVal << "]\n"
                                      "Value mean: " << mGlobalValue.mean
            << "\n"
               "Value std. dev.: " << mGlobalValue.stdDev << "\n";

    const std::string fileName = mName + "/data.dat";

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create size range log file: "
                                 + fileName);

    data << dataStr.str();

    std::cout << "---\n" << dataStr.str() << "---" << std::endl;
}

void N2D2::StimuliData::clear()
{
    mSize.clear();
    mMinSize = Size();
    mMaxSize = Size();

    mValue.clear();
    mGlobalValue = Value();
}

N2D2::StimuliData::Size N2D2::StimuliData::getMeanSize() const
{
    Size mean;

    for (std::vector<Size>::const_iterator it = mSize.begin(),
                                           itEnd = mSize.end();
         it != itEnd;
         ++it) {
        mean.dimX += (*it).dimX;
        mean.dimY += (*it).dimY;
        mean.dimZ += (*it).dimZ;
    }

    mean.dimX /= mSize.size();
    mean.dimY /= mSize.size();
    mean.dimZ /= mSize.size();

    return mean;
}

void N2D2::StimuliData::logSizeRange() const
{
    const std::string fileName = mName + "/sizeRange.dat";

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create size range log file: "
                                 + fileName);

    unsigned int sumDimX = 0;
    unsigned int sumDimY = 0;

    for (std::vector<Size>::const_iterator it = mSize.begin(),
                                           itEnd = mSize.end();
         it != itEnd;
         ++it) {
        sumDimX += (*it).dimX;
        sumDimY += (*it).dimY;

        data << (*it).dimX << " " << (*it).dimY << " " << (*it).dimZ << "\n";
    }

    data.close();

    // Plot results
    Gnuplot gnuplot;
    gnuplot.set("grid").set("key right bottom");
    gnuplot.set("yrange", "[0:]");
    gnuplot.setXrange(std::min(mMinSize.dimX, mMinSize.dimY) - 1,
                      std::max(mMaxSize.dimX, mMaxSize.dimY) + 1);
    gnuplot.setXlabel("Data size");
    gnuplot.setYlabel("Number of stimuli (cumulative)");

    std::ostringstream label;
    label << "\"Average size: " << sumDimX / mSize.size() << "x"
          << sumDimY / mSize.size() << "\" at graph 0.5, graph 0.1 front";

    gnuplot.saveToFile(fileName, "-cumul");
    gnuplot.set("label", label.str());
    gnuplot.plot(fileName,
                 "using ($1+$0/1.0e12):(1.0) smooth cumulative with steps "
                 "title 'width', "
                 "'' using ($2+$0/1.0e12):(1.0) smooth cumulative with steps "
                 "title 'height'");

    gnuplot.saveToFile(fileName, "-cumul-width");
    gnuplot.set("label", label.str());
    gnuplot.plot("< LC_ALL=C sort -k1g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'width', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'height'");

    gnuplot.saveToFile(fileName, "-cumul-height");
    gnuplot.set("label", label.str());
    gnuplot.plot("< LC_ALL=C sort -k2g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'width', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'height'");
}

void N2D2::StimuliData::logValueRange() const
{
    const std::string fileName = mName + "/valueRange.dat";

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create value range log file: "
                                 + fileName);

    for (std::vector<Value>::const_iterator it = mValue.begin(),
                                            itEnd = mValue.end();
         it != itEnd;
         ++it)
        data << (*it).minVal << " " << (*it).maxVal << " " << (*it).mean << " "
             << (*it).stdDev << "\n";

    data.close();

    // Plot results
    Gnuplot gnuplot;
    gnuplot.set("grid").set("key right bottom");
    gnuplot.set("yrange", "[0:]");
    gnuplot.setXrange(0.99 * (mGlobalValue.minVal - 0.01),
                      1.01 * (mGlobalValue.maxVal + 0.01));
    gnuplot.setXlabel("Data value");
    gnuplot.setYlabel("Number of stimuli (cumulative)");

    gnuplot.saveToFile(fileName, "-cumul");
    gnuplot.plot(
        fileName,
        "using ($1+$0/1.0e12):(1.0) smooth cumulative with steps title 'min', "
        "'' using ($3+$0/1.0e12):(1.0) smooth cumulative with steps title "
        "'mean', "
        "'' using ($2+$0/1.0e12):(1.0) smooth cumulative with steps title "
        "'max'");

    gnuplot.saveToFile(fileName, "-min");
    gnuplot.plot("< LC_ALL=C sort -k1g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'min', "
                 "'' using ($3+$0/1.0e12):($0+1) with steps title 'mean', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'max', "
                 "'' using ($3+$0/1.0e12):($0+1):4 with xerrorbars notitle");

    gnuplot.saveToFile(fileName, "-max");
    gnuplot.plot("< LC_ALL=C sort -k2g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'min', "
                 "'' using ($3+$0/1.0e12):($0+1) with steps title 'mean', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'max', "
                 "'' using ($3+$0/1.0e12):($0+1):4 with xerrorbars notitle");

    gnuplot.saveToFile(fileName, "-mean");
    gnuplot.plot("< LC_ALL=C sort -k3g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'min', "
                 "'' using ($3+$0/1.0e12):($0+1) with steps title 'mean', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'max', "
                 "'' using ($3+$0/1.0e12):($0+1):4 with xerrorbars notitle");

    gnuplot.saveToFile(fileName, "-stddev");
    gnuplot.plot("< LC_ALL=C sort -k4g " + fileName,
                 "using ($1+$0/1.0e12):($0+1) with steps title 'min', "
                 "'' using ($3+$0/1.0e12):($0+1) with steps title 'mean', "
                 "'' using ($2+$0/1.0e12):($0+1) with steps title 'max', "
                 "'' using ($3+$0/1.0e12):($0+1):4 with xerrorbars notitle");
}

void N2D2::StimuliData::generate(Database::StimuliSetMask setMask)
{
    clear();

    const std::string& cacheName = mName + "/_cache";

    if (!loadDataCache(cacheName)) {
        const unsigned int batchSize = mProvider.getBatchSize();
        mProvider.setBatchSize(0);

        const std::string cachePath = mProvider.getCachePath();
        mProvider.setCachePath();

        long double sum = 0.0;
        long double sqSum = 0.0;
        unsigned long long int count = 0;

        const std::vector<Database::StimuliSet> stimuliSets
            = mProvider.getDatabase().getStimuliSets(setMask);

        // For progression visualization
        std::cout << mName << " processing(1/2)" << std::flush;
        unsigned int toLoad = 0;

        for (std::vector<Database::StimuliSet>::const_iterator it
             = stimuliSets.begin(),
             itEnd = stimuliSets.end();
             it != itEnd;
             ++it) {
            toLoad += mProvider.getDatabase().getNbStimuli(*it);
        }

        unsigned int loaded = 0;
        unsigned int progress = 0, progressPrev = 0;

        // First loop: compute frame stats + global mean
        for (std::vector<Database::StimuliSet>::const_iterator it
             = stimuliSets.begin(),
             itEnd = stimuliSets.end();
             it != itEnd;
             ++it) {
            const unsigned int nbStimuli
                = mProvider.getDatabase().getNbStimuli(*it);
            const bool rawData = (mProvider.getNbTransformations(*it) == 0);

            mSize.reserve(mSize.size() + nbStimuli);
            mValue.reserve(mValue.size() + nbStimuli);

            for (unsigned int index = 0; index < nbStimuli; ++index) {
                if (!rawData)
                    mProvider.readStimulus(*it, index, 0);

                const Tensor3d<Float_T> data
                    = (rawData) ? mProvider.readRawData(*it, index)
                                : mProvider.getData()[0];

                assert(!data.empty());

                // Size
                const Size size(data.dimX(), data.dimY(), data.dimZ());

                if (mSize.empty()) {
                    mMinSize = size;
                    mMaxSize = size;
                } else {
                    mMinSize.dimX = std::min(mMinSize.dimX, data.dimX());
                    mMinSize.dimY = std::min(mMinSize.dimY, data.dimY());
                    mMinSize.dimZ = std::min(mMinSize.dimZ, data.dimZ());
                    mMaxSize.dimX = std::max(mMaxSize.dimX, data.dimX());
                    mMaxSize.dimY = std::max(mMaxSize.dimY, data.dimY());
                    mMaxSize.dimZ = std::max(mMaxSize.dimZ, data.dimZ());
                }

                mSize.push_back(size);

                // Value
                const std::pair<std::vector<Float_T>::const_iterator,
                                std::vector<Float_T>::const_iterator> minMaxIt
                    = std::minmax_element(data.begin(), data.end());
                // A new vector must be created because data.data() contains the
                // full content of the original 4D tensor
                const std::pair<double, double> meanStdDev
                    = Utils::meanStdDev(data.begin(), data.end());

                if (mValue.empty()) {
                    mGlobalValue.minVal = *minMaxIt.first;
                    mGlobalValue.maxVal = *minMaxIt.second;
                } else {
                    mGlobalValue.minVal
                        = std::min(mGlobalValue.minVal, *minMaxIt.first);
                    mGlobalValue.maxVal
                        = std::max(mGlobalValue.maxVal, *minMaxIt.second);
                }

                mValue.push_back(Value(*minMaxIt.first,
                                       *minMaxIt.second,
                                       meanStdDev.first,
                                       meanStdDev.second));

                sum += meanStdDev.first * data.size();
                count += data.size();

                // Progress bar
                progress = (unsigned int)(20.0 * (++loaded) / (double)toLoad);

                if (progress > progressPrev) {
                    std::cout << std::string(progress - progressPrev, '.')
                              << std::flush;
                    progressPrev = progress;
                }
            }
        }

        std::cout << std::endl;

        mGlobalValue.mean = sum / count;

        const std::string& meanDataFile = mName + "/meanData.bin";
        cv::Mat meanData;
        bool computeMeanData = false;

        if (mMeanData) {
            if (mMinSize.dimX == mMaxSize.dimX && mMinSize.dimY == mMaxSize.dimY
                && mMinSize.dimZ == mMaxSize.dimZ) {
                if (!std::ifstream(meanDataFile.c_str()).good())
                    computeMeanData = true;
            } else {
                std::cout << Utils::cwarning
                          << "Warning: StimuliData::generate(): cannot compute "
                             "mean data on images of different"
                             " sizes: min = [" << mMinSize.dimX << "x"
                          << mMinSize.dimY << "x" << mMinSize.dimZ << "], "
                                                                      "max = ["
                          << mMaxSize.dimX << "x" << mMaxSize.dimY << "x"
                          << mMaxSize.dimZ << "]" << Utils::cdef << std::endl;
            }
        }

        std::cout << mName << " processing(2/2)" << std::flush;
        loaded = 0;
        progress = 0;
        progressPrev = 0;

        // Second loop: compute global std.dev. + mean data if enabled
        for (std::vector<Database::StimuliSet>::const_iterator it
             = stimuliSets.begin(),
             itEnd = stimuliSets.end();
             it != itEnd;
             ++it) {
            const unsigned int nbStimuli
                = mProvider.getDatabase().getNbStimuli(*it);
            const bool rawData = (mProvider.getNbTransformations(*it) == 0);

            for (unsigned int index = 0; index < nbStimuli; ++index) {
                if (!rawData)
                    mProvider.readStimulus(*it, index, 0);

                const Tensor3d<Float_T> data
                    = (rawData) ? mProvider.readRawData(*it, index)
                                : mProvider.getData()[0];

                for (std::vector<Float_T>::const_iterator it = data.begin(),
                                                          itEnd = data.end();
                     it != itEnd;
                     ++it) {
                    const double v = (*it) - mGlobalValue.mean;
                    sqSum += v * v;
                }

                if (computeMeanData) {
                    cv::Mat matData;
                    ((cv::Mat)data).convertTo(matData, CV_64F);

                    if (meanData.empty())
                        meanData = matData;
                    else
                        meanData += matData;
                }

                // Progress bar
                progress = (unsigned int)(20.0 * (++loaded) / (double)toLoad);

                if (progress > progressPrev) {
                    std::cout << std::string(progress - progressPrev, '.')
                              << std::flush;
                    progressPrev = progress;
                }
            }
        }

        std::cout << std::endl;

        mGlobalValue.stdDev = std::sqrt(sqSum / count);

        if (computeMeanData) {
            meanData /= (double)mSize.size();
            BinaryCvMat::write(meanDataFile, meanData);

            StimuliProvider::logData(Utils::fileBaseName(meanDataFile) + ".dat",
                                     Tensor3d<Float_T>(meanData));
        }

        mProvider.setBatchSize(batchSize);
        mProvider.setCachePath(cachePath);
        saveDataCache(cacheName);
    }
}

bool N2D2::StimuliData::loadDataCache(const std::string& fileName)
{
    std::ifstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        return false;

    unsigned int sizeLength;
    data.read(reinterpret_cast<char*>(&sizeLength), sizeof(sizeLength));

    if (!data.good())
        throw std::runtime_error("Error reading cache file: " + fileName);

    mSize.resize(sizeLength);
    data.read(reinterpret_cast<char*>(&mSize[0]),
              sizeLength * sizeof(mSize[0]));
    data.read(reinterpret_cast<char*>(&mMinSize), sizeof(mMinSize));
    data.read(reinterpret_cast<char*>(&mMaxSize), sizeof(mMaxSize));

    unsigned int valueLength;
    data.read(reinterpret_cast<char*>(&valueLength), sizeof(valueLength));

    if (!data.good())
        throw std::runtime_error("Error reading cache file: " + fileName);

    mValue.resize(valueLength);
    data.read(reinterpret_cast<char*>(&mValue[0]),
              valueLength * sizeof(mValue[0]));
    data.read(reinterpret_cast<char*>(&mGlobalValue), sizeof(mGlobalValue));

    if (!data.good())
        throw std::runtime_error("Error reading cache file: " + fileName);

    return true;
}

void N2D2::StimuliData::saveDataCache(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not create cache file: " + fileName);

    const unsigned int sizeLength = mSize.size();
    data.write(reinterpret_cast<const char*>(&sizeLength), sizeof(sizeLength));
    data.write(reinterpret_cast<const char*>(&mSize[0]),
               sizeLength * sizeof(mSize[0]));
    data.write(reinterpret_cast<const char*>(&mMinSize), sizeof(mMinSize));
    data.write(reinterpret_cast<const char*>(&mMaxSize), sizeof(mMaxSize));

    const unsigned int valueLength = mValue.size();
    data.write(reinterpret_cast<const char*>(&valueLength),
               sizeof(valueLength));
    data.write(reinterpret_cast<const char*>(&mValue[0]),
               valueLength * sizeof(mValue[0]));
    data.write(reinterpret_cast<const char*>(&mGlobalValue),
               sizeof(mGlobalValue));

    if (!data.good())
        throw std::runtime_error("Error writing cache file: " + fileName);
}
