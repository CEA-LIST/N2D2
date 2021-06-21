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
#include "StimuliProvider.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Gnuplot.hpp"

N2D2::StimuliData::StimuliData(const std::string& name,
                               StimuliProvider& provider)
    : mName(name),
      mProvider(provider),
      mMeanData(this, "MeanData", false),
      mStdDevData(this, "StdDevData", false)
{
    // ctor
    Utils::createDirectories(mName);
}

N2D2::StimuliData::StimuliData(const StimuliData& stimuliData)
    : Parameterizable(),
      mName(stimuliData.mName),
      mProvider(stimuliData.mProvider),
      mMeanData(this, "MeanData", stimuliData.mMeanData),
      mStdDevData(this, "StdDevData", stimuliData.mStdDevData)
{
    // copy-ctor
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

    if (!mSize.empty()) {
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
    }

    return mean;
}

void N2D2::StimuliData::logSizeRange() const
{
    if (mValue.empty())
        return;

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
    if (mValue.empty())
        return;

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

unsigned int N2D2::StimuliData::generate(Database::StimuliSetMask setMask,
                                         bool noRaw)
{
    clear();

    const std::string& cacheName = mName + "/_cache";

    // For progression visualization
    unsigned int toLoad = 0;

    const std::vector<Database::StimuliSet> stimuliSets
        = mProvider.getDatabase().getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
            = stimuliSets.begin(),
            itEnd = stimuliSets.end();
            it != itEnd;
            ++it) {
        toLoad += mProvider.getDatabase().getNbStimuli(*it);
    }

    std::cout << mName << " processing " << toLoad << " stimuli"
        << std::flush;

    if (toLoad > 0 && !loadDataCache(cacheName)) {
        const unsigned int batchSize = mProvider.getBatchSize();
        mProvider.setBatchSize(0);

        const std::string cachePath = mProvider.getCachePath();
        mProvider.setCachePath();

        unsigned int minSizeX = std::numeric_limits<unsigned int>::max();
        unsigned int minSizeY = std::numeric_limits<unsigned int>::max();
        unsigned int minSizeZ = std::numeric_limits<unsigned int>::max();
        unsigned int maxSizeX = 0U;
        unsigned int maxSizeY = 0U;
        unsigned int maxSizeZ = 0U;

        Float_T globalValueMin = std::numeric_limits<Float_T>::max();
        Float_T globalValueMax = -std::numeric_limits<Float_T>::max();

        const std::string& meanDataFile = mName + "/meanData.bin";
        bool computeMeanData = mMeanData
                                && !std::ifstream(meanDataFile.c_str()).good();
        cv::Mat meanData;

        const std::string& stdDevDataFile = mName + "/stdDevData.bin";
        bool computeStdDevData = mStdDevData
                               && !std::ifstream(stdDevDataFile.c_str()).good();
        cv::Mat M2Data;

        mSize.resize(toLoad);
        mValue.resize(toLoad);

        // to compute global mean
        long double sumMean = 0.0;
        unsigned long long int sumCount = 0;
        // to compute global stdDev
        long double sumM2 = 0.0;
        std::vector<double> mean(toLoad, 0.0);
        std::vector<unsigned int> count(toLoad, 0U);

        unsigned int loaded = 0;
        unsigned int progress = 0, progressPrev = 0;

#ifdef CUDA
        int dev = 0;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif
        // First loop: compute frame stats + global mean
        for (std::vector<Database::StimuliSet>::const_iterator it
             = stimuliSets.begin(),
             itEnd = stimuliSets.end();
             it != itEnd;
             ++it) {
            const unsigned int nbStimuli
                = mProvider.getDatabase().getNbStimuli(*it);
            const bool rawData = (mProvider.getNbTransformations(*it) == 0
                                    && !noRaw);

#pragma omp parallel for schedule(dynamic) reduction(+:sumMean,sumCount,sumM2)
// min and max reduction not supported by MSVC, using double-checked locking instead.
//reduction(min:minSizeX,minSizeY,minSizeZ,globalValueMin) reduction(max:maxSizeX,maxSizeY,maxSizeZ,globalValueMax)
            for (int index = 0; index < (int)nbStimuli; ++index) {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif 
                StimuliProvider provider = mProvider.cloneParameters();

                if (!rawData)
                    provider.readStimulus(*it, index, 0);

                const Tensor<Float_T> data
                    = (rawData) ? provider.readRawData(*it, index)
                                : provider.getData()[0];

                assert(!data.empty());

                double dataMean = 0.0;
                double dataM2 = 0.0;
                Float_T minVal = data(0);
                Float_T maxVal = data(0);

                // Use Welford's method to compute std. dev. in one pass
                for (unsigned int k = 0, kSize = data.size(); k < kSize; ++k) {
                    const double x = data(k);

                    const double delta = (x - dataMean);
                    dataMean += delta / (k + 1);
                    const double delta2 = (x - dataMean);
                    dataM2 += delta * delta2;

                    if (x < minVal)
                        minVal = x;

                    if (x > maxVal)
                        maxVal = x;
                }

                const double dataStdDev = std::sqrt(dataM2 / (data.size() - 1));

                // to compute global mean
                sumMean += dataMean * data.size();
                sumCount += data.size();
                // to compute global stdDev
                sumM2 += dataM2;
                mean[loaded + index] = dataMean;
                count[loaded + index] = data.size();

                // Size
                // Use dimD to get the true 3rd dimension, as dimZ has a special
                // meaning!
                const Size size(data.dimX(), data.dimY(), data.dimD());

                // Value
                const Value value(minVal, maxVal, dataMean, dataStdDev);

                if (data.dimX() < minSizeX) {
#pragma omp critical(StimuliData__generate_minSizeX)
                    if (data.dimX() < minSizeX)
                        minSizeX = data.dimX();
                }

                if (data.dimY() < minSizeY) {
#pragma omp critical(StimuliData__generate_minSizeY)
                    if (data.dimY() < minSizeY)
                        minSizeY = data.dimY();
                }

                if (data.dimD() < minSizeZ) {
#pragma omp critical(StimuliData__generate_minSizeZ)
                    if (data.dimD() < minSizeZ)
                        minSizeZ = data.dimD();
                }

                if (minVal < globalValueMin) {
#pragma omp critical(StimuliData__generate_globalValueMin)
                    if (minVal < globalValueMin)
                        globalValueMin = minVal;
                }

                if (data.dimX() > maxSizeX) {
#pragma omp critical(StimuliData__generate_maxSizeX)
                    if (data.dimX() > maxSizeX)
                        maxSizeX = data.dimX();
                }

                if (data.dimY() > maxSizeY) {
#pragma omp critical(StimuliData__generate_maxSizeY)
                    if (data.dimY() > maxSizeY)
                        maxSizeY = data.dimY();
                }

                if (data.dimD() > maxSizeZ) {
#pragma omp critical(StimuliData__generate_maxSizeZ)
                    if (data.dimD() > maxSizeZ)
                        maxSizeZ = data.dimD();
                }

                if (maxVal > globalValueMax) {
#pragma omp critical(StimuliData__generate_globalValueMax)
                    if (maxVal > globalValueMax)
                        globalValueMax = maxVal;
                }

                mSize[loaded + index] = size;
                mValue[loaded + index] = value;

                if (computeMeanData || computeStdDevData) {
                    cv::Mat matData;
                    // Use double for maximum precision
                    ((cv::Mat)data).convertTo(matData, CV_64F);

#pragma omp critical(StimuliData__generate_meanData)
                    if (computeMeanData || computeStdDevData) {
                        if (meanData.empty()) {
                            meanData = cv::Mat::zeros(matData.size(),
                                                      matData.type());
                        }

                        if (M2Data.empty() && computeStdDevData) {
                            M2Data = cv::Mat::zeros(matData.size(),
                                                    matData.type());
                        }

                        if (matData.size() == meanData.size()
                            && matData.type() == meanData.type())
                        {
                            const cv::Mat delta = (matData - meanData);
                            meanData += delta / (loaded + index + 1);

                            if (computeStdDevData) {
                                const cv::Mat delta2 = (matData - meanData);
                                M2Data += delta.mul(delta2);
                            }
                        }
                        else {
                            computeMeanData = false;
                            computeStdDevData = false;
                        }
                    }
                }

                // Progress bar
                progress = (unsigned int)(20.0 * (loaded + index)
                                                            / (double)toLoad);

                if (progress > progressPrev) {
#pragma omp critical(StimuliData__generate)
                    if (progress > progressPrev) {
                        std::cout << std::string(progress - progressPrev, '.')
                                  << std::flush;
                        progressPrev = progress;
                    }
                }
            }

            loaded += nbStimuli;
        }

        assert(loaded == toLoad);

        mMinSize = Size(minSizeX, minSizeY, minSizeZ);
        mMaxSize = Size(maxSizeX, maxSizeY, maxSizeZ);

        mGlobalValue.minVal = globalValueMin;
        mGlobalValue.maxVal = globalValueMax;

        // Compute global mean
        const double globalMean = (sumCount > 0) ? sumMean / sumCount : 0.0;
        mGlobalValue.mean = globalMean;

        // Compute global stdDev
        // source: https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
        double sqSum = 0.0;

        for (unsigned int k = 0; k < loaded; ++k) {
            const double delta = (mean[k] - globalMean);
            sqSum += count[k] * delta * delta;
        }

        mGlobalValue.stdDev = (sumCount > 1)
            ? std::sqrt((sumM2 + sqSum) / (sumCount - 1)) : 0.0;

        if (computeMeanData || computeStdDevData) {
            BinaryCvMat::write(meanDataFile, meanData);
            StimuliProvider::logData(Utils::fileBaseName(meanDataFile) + ".dat",
                                     Tensor<Float_T>(meanData));

            if (computeStdDevData) {
                M2Data /= (double)(toLoad - 1);

                cv::Mat stdDevData;
                cv::sqrt(M2Data, stdDevData);

                const int nonZero = cv::countNonZero(stdDevData.reshape(1));
                assert(nonZero <= (int)stdDevData.reshape(1).total());

                if (nonZero < (int)stdDevData.reshape(1).total()) {
                    std::cout << Utils::cwarning
                        << "Warning: StimuliData::generate(): beware that"
                        " StdDevData matrix contains "
                        << (stdDevData.reshape(1).total() - nonZero) << " 0s."
                        << Utils::cdef << std::endl;
                }

                BinaryCvMat::write(stdDevDataFile, stdDevData);
                StimuliProvider::logData(Utils::fileBaseName(stdDevDataFile)
                                            + ".dat",
                                         Tensor<Float_T>(stdDevData));
            }
        }
        else if ((mMeanData && !std::ifstream(meanDataFile.c_str()).good())
            || (mStdDevData && !std::ifstream(stdDevDataFile.c_str()).good()))
        {
            std::cout << Utils::cwarning
                      << "Warning: StimuliData::generate(): cannot compute "
                         "mean/std.dev. data on images of different"
                         " sizes: min = [" << mMinSize.dimX << "x"
                      << mMinSize.dimY << "x" << mMinSize.dimZ << "], "
                                                                  "max = ["
                      << mMaxSize.dimX << "x" << mMaxSize.dimY << "x"
                      << mMaxSize.dimZ << "]" << Utils::cdef << std::endl;
        }

        mProvider.setBatchSize(batchSize);
        mProvider.setCachePath(cachePath);
        saveDataCache(cacheName);
    }

    std::cout << std::endl;

    return toLoad;
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
