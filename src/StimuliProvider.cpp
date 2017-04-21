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

#include "StimuliProvider.hpp"

N2D2::StimuliProvider::StimuliProvider(Database& database,
                                       unsigned int sizeX,
                                       unsigned int sizeY,
                                       unsigned int nbChannels,
                                       unsigned int batchSize,
                                       bool compositeStimuli)
    : // Variables
      mDatabase(database),
      mSizeX(sizeX),
      mSizeY(sizeY),
      mNbChannels(nbChannels),
      mBatchSize(batchSize),
      mCompositeStimuli(compositeStimuli),
      mCachePath("_cache"),
      mBatch(batchSize),
      mFutureBatch(batchSize),
      mData(sizeX, sizeY, nbChannels, batchSize),
      mFutureData(sizeX, sizeY, nbChannels, batchSize),
      mLabelsROI(batchSize, std::vector<std::shared_ptr<ROI> >()),
      mFutureLabelsROI(batchSize, std::vector<std::shared_ptr<ROI> >()),
      mFuture(false)
{
    // ctor
    Utils::createDirectories(mCachePath); // Create default cache directory

    if (mCompositeStimuli) {
        mLabelsData.resize(sizeX, sizeY, 1, batchSize);
        mFutureLabelsData.resize(sizeX, sizeY, 1, batchSize);
    } else {
        mLabelsData.resize(1, 1, 1, batchSize);
        mFutureLabelsData.resize(1, 1, 1, batchSize);
    }
}

void N2D2::StimuliProvider::addChannel(const CompositeTransformation
                                       & /*transformation*/)
{
    if (mChannelsTransformations.empty())
        mNbChannels = 1;
    else
        ++mNbChannels;

    mData.resize(mSizeX, mSizeY, mNbChannels, mBatchSize);
    mFutureData.resize(mSizeX, mSizeY, mNbChannels, mBatchSize);

    if (!mChannelsTransformations.empty()) {
        mLabelsData.resize(mLabelsData.dimX(),
                           mLabelsData.dimY(),
                           mLabelsData.dimZ() + 1,
                           mBatchSize);
        mFutureLabelsData.resize(mFutureLabelsData.dimX(),
                                 mFutureLabelsData.dimY(),
                                 mFutureLabelsData.dimZ() + 1,
                                 mBatchSize);
    }

    mChannelsTransformations.push_back(TransformationsSets());
}

void N2D2::StimuliProvider::addTransformation(const CompositeTransformation
                                              & transformation,
                                              Database::StimuliSetMask setMask)
{
    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mTransformations(*it).cacheable.push_back(transformation);
}

void N2D2::StimuliProvider::addOnTheFlyTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mTransformations(*it).onTheFly.push_back(transformation);
}

void N2D2::StimuliProvider::addChannelTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    addChannel(transformation);

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mChannelsTransformations.back()(*it)
            .cacheable.push_back(transformation);
}

void N2D2::StimuliProvider::addChannelOnTheFlyTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    addChannel(transformation);

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mChannelsTransformations.back()(*it).onTheFly.push_back(transformation);
}

void N2D2::StimuliProvider::addChannelTransformation(
    unsigned int channel,
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    if (channel >= mChannelsTransformations.size())
        throw std::runtime_error("StimuliProvider::addChannelTransformation(): "
                                 "the channel does not exist");

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mChannelsTransformations[channel](*it)
            .cacheable.push_back(transformation);
}

void N2D2::StimuliProvider::addChannelOnTheFlyTransformation(
    unsigned int channel,
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    if (channel >= mChannelsTransformations.size())
        throw std::runtime_error("StimuliProvider::"
                                 "addChannelOnTheFlyTransformation(): the "
                                 "channel does not exist");

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
        mChannelsTransformations[channel](*it)
            .onTheFly.push_back(transformation);
}

void N2D2::StimuliProvider::addChannelsTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it) {
        for (std::vector<TransformationsSets>::iterator itTrans
             = mChannelsTransformations.begin(),
             itTransEnd = mChannelsTransformations.end();
             itTrans != itTransEnd;
             ++itTrans) {
            (*itTrans)(*it).cacheable.push_back(transformation);
        }
    }
}

void N2D2::StimuliProvider::addChannelsOnTheFlyTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it) {
        for (std::vector<TransformationsSets>::iterator itTrans
             = mChannelsTransformations.begin(),
             itTransEnd = mChannelsTransformations.end();
             itTrans != itTransEnd;
             ++itTrans) {
            (*itTrans)(*it).onTheFly.push_back(transformation);
        }
    }
}

void N2D2::StimuliProvider::future()
{
    mFuture = true;
}

void N2D2::StimuliProvider::synchronize()
{
    if (mFuture) {
        mBatch.swap(mFutureBatch);
        mData.swap(mFutureData);
        mLabelsData.swap(mFutureLabelsData);
        mLabelsROI.swap(mFutureLabelsROI);
        mFuture = false;
    }
}

unsigned int N2D2::StimuliProvider::getRandomIndex(Database::StimuliSet set)
{
    return Random::randUniform(0, mDatabase.getNbStimuli(set) - 1);
}

N2D2::Database::StimulusID
N2D2::StimuliProvider::getRandomID(Database::StimuliSet set)
{
    return mDatabase.getStimulusID(set, getRandomIndex(set));
}

void N2D2::StimuliProvider::readRandomBatch(Database::StimuliSet set)
{
    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    for (unsigned int batchPos = 0; batchPos < mBatchSize; ++batchPos)
        batchRef[batchPos] = getRandomID(set);

#pragma omp parallel for if (mBatchSize > 1)
    for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos)
        readStimulus(batchRef[batchPos], set, batchPos);
}

N2D2::Database::StimulusID
N2D2::StimuliProvider::readRandomStimulus(Database::StimuliSet set,
                                          unsigned int batchPos)
{
    const Database::StimulusID id = getRandomID(set);
    readStimulus(id, set, batchPos);
    return id;
}

void N2D2::StimuliProvider::readBatch(Database::StimuliSet set,
                                      unsigned int startIndex)
{
    const unsigned int batchSize
        = std::min(mBatchSize, mDatabase.getNbStimuli(set) - startIndex);
    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        batchRef[batchPos]
            = mDatabase.getStimulusID(set, startIndex + batchPos);

#pragma omp parallel for if (batchSize > 1)
    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos)
        readStimulus(batchRef[batchPos], set, batchPos);

    std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
}

void N2D2::StimuliProvider::readStimulus(Database::StimulusID id,
                                         Database::StimuliSet set,
                                         unsigned int batchPos)
{
    std::stringstream dataCacheFile, labelsCacheFile;
    dataCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                  << id << "_data_" << set << ".bin";
    labelsCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                    << id << "_labels_" << set << ".bin";

    std::vector<std::shared_ptr<ROI> >& labelsROI
        = (mFuture) ? mFutureLabelsROI[batchPos] : mLabelsROI[batchPos];
    labelsROI = mDatabase.getStimulusROIs(id);

    std::vector<cv::Mat> rawChannelsData;
    std::vector<cv::Mat> rawChannelsLabels;

    // 1. Cached data
    if (!mCachePath.empty() && std::ifstream(dataCacheFile.str()).good()) {
        // Cache present, load the pre-processed data
        rawChannelsData = loadDataCache(dataCacheFile.str());
        rawChannelsLabels = loadDataCache(labelsCacheFile.str());
    } else {
        // Cache not present, load the raw stimuli from the database
        cv::Mat rawData
            = mDatabase.getStimulusData(id)
                  .clone(); // make sure the database image will not be altered
        cv::Mat rawLabels
            = mDatabase.getStimulusLabelsData(id)
                  .clone(); // make sure the database image will not be altered

        // Apply global cacheable transformation
        mTransformations(set)
            .cacheable.apply(rawData, rawLabels, labelsROI, id);

        if (mTransformations(set).onTheFly.empty()
            && !mChannelsTransformations.empty()) {
            // If no global on-the-fly transformation, apply the cacheable
            // channels transformations
            for (std::vector<TransformationsSets>::iterator it
                 = mChannelsTransformations.begin(),
                 itEnd = mChannelsTransformations.end();
                 it != itEnd;
                 ++it) {
                cv::Mat channelData = rawData.clone();
                cv::Mat channelLabels = rawLabels.clone();
                (*it)(set).cacheable.apply(channelData, channelLabels, id);
                rawChannelsData.push_back(channelData);
                rawChannelsLabels.push_back(channelLabels);
            }
        } else {
            rawChannelsData.push_back(rawData);
            rawChannelsLabels.push_back(rawLabels);
        }

        // Save the pre-processed data
        if (!mCachePath.empty()) {
            saveDataCache(dataCacheFile.str(), rawChannelsData);
            saveDataCache(labelsCacheFile.str(), rawChannelsLabels);
        }
    }

    // 2. On-the-fly processing
    if (!mTransformations(set).onTheFly.empty())
        mTransformations(set).onTheFly.apply(
            rawChannelsData[0], rawChannelsLabels[0], labelsROI, id);

    Tensor3d<Float_T> data = (mChannelsTransformations.empty())
                                 ? Tensor3d<Float_T>(rawChannelsData[0])
                                 : Tensor3d<Float_T>();
    Tensor3d<int> labels = (mChannelsTransformations.empty()) ? Tensor3d
                               <int>(rawChannelsLabels[0])
                                                              : Tensor3d<int>();

    // 2.1 Process channels
    if (!mChannelsTransformations.empty()) {
        for (std::vector<TransformationsSets>::iterator it
             = mChannelsTransformations.begin(),
             itBegin = mChannelsTransformations.begin(),
             itEnd = mChannelsTransformations.end();
             it != itEnd;
             ++it) {
            cv::Mat channelData((rawChannelsData.size() > 1)
                                    ? rawChannelsData[it - itBegin].clone()
                                    : rawChannelsData[0].clone());
            cv::Mat channelLabels
                = ((rawChannelsLabels.size() > 1)
                       ? rawChannelsLabels[it - itBegin].clone()
                       : rawChannelsLabels[0].clone());

            if (!mTransformations(set).onTheFly.empty())
                (*it)(set).cacheable.apply(channelData, channelLabels, id);

            (*it)(set).onTheFly.apply(channelData, channelLabels, id);
            data.push_back(Tensor2d<Float_T>(channelData));
            labels.push_back(Tensor2d<int>(channelLabels));
        }
    }

    Tensor4d<Float_T>& dataRef = (mFuture) ? mFutureData : mData;
    Tensor4d<int>& labelsRef = (mFuture) ? mFutureLabelsData : mLabelsData;

    if (mBatchSize > 0) {
        if (dataRef.dimX() != data.dimX() || dataRef.dimY() != data.dimY()
            || dataRef.dimZ() != data.dimZ()) {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected data size is "
                << dataRef.dimX() << "x" << dataRef.dimY() << "x"
                << dataRef.dimZ() << ", but size after transformations is "
                << data.dimX() << "x" << data.dimY() << "x" << data.dimZ();

            throw std::runtime_error(msg.str());
        }

        dataRef[batchPos] = data;

        if (labelsRef.dimX() != labels.dimX()
            || labelsRef.dimY() != labels.dimY()
            || labelsRef.dimZ() != labels.dimZ()) {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected labels size is "
                << labelsRef.dimX() << "x" << labelsRef.dimY() << "x"
                << labelsRef.dimZ() << ", but size after transformations is "
                << labels.dimX() << "x" << labels.dimY() << "x"
                << labels.dimZ();

            throw std::runtime_error(msg.str());
        }

        labelsRef[batchPos] = labels;
    } else {
        dataRef.clear();
        dataRef.push_back(data);
        labelsRef.clear();
        labelsRef.push_back(labels);
    }
}

N2D2::Database::StimulusID N2D2::StimuliProvider::readStimulus(
    Database::StimuliSet set, unsigned int index, unsigned int batchPos)
{
    const Database::StimulusID id = mDatabase.getStimulusID(set, index);
    readStimulus(id, set, batchPos);
    return id;
}

void N2D2::StimuliProvider::streamStimulus(const cv::Mat& mat,
                                           Database::StimuliSet set,
                                           unsigned int batchPos)
{
    Tensor4d<Float_T>& dataRef = (mFuture) ? mFutureData : mData;

    // Apply global transformation
    cv::Mat rawData = mat.clone();
    mTransformations(set).cacheable.apply(rawData);
    mTransformations(set).onTheFly.apply(rawData);

    Tensor3d<Float_T> data = (mChannelsTransformations.empty())
                                 ? Tensor3d<Float_T>(rawData)
                                 : Tensor3d<Float_T>();

    if (!mChannelsTransformations.empty()) {
        // Apply channels transformations
        for (std::vector<TransformationsSets>::iterator it
             = mChannelsTransformations.begin(),
             itEnd = mChannelsTransformations.end();
             it != itEnd;
             ++it) {
            cv::Mat channelData(rawData.clone());
            (*it)(set).cacheable.apply(channelData);
            (*it)(set).onTheFly.apply(channelData);
            data.push_back(Tensor2d<Float_T>(channelData));
        }
    }

    dataRef[batchPos] = data;
}

void N2D2::StimuliProvider::reverseLabels(const cv::Mat& mat,
                                          Database::StimuliSet set,
                                          Tensor2d<int>& labels,
                                          std::vector
                                          <std::shared_ptr<ROI> >& labelsROIs)
{
    std::vector<cv::Mat> frameSteps(1, mat.clone());

    // Forward
    if (!mTransformations(set).onTheFly.empty()) {
        frameSteps.push_back(frameSteps.back().clone());
        mTransformations(set).cacheable.apply(frameSteps.back());
    }

    // Reverse
    cv::Mat labelsMat = (cv::Mat)labels;

    if (!mTransformations(set).onTheFly.empty()) {
        mTransformations(set)
            .onTheFly.reverse(frameSteps.back(), labelsMat, labelsROIs);
        frameSteps.pop_back();
    }

    mTransformations(set)
        .cacheable.reverse(frameSteps.back(), labelsMat, labelsROIs);
    labels = Tensor2d<int>(labelsMat);
}

void N2D2::StimuliProvider::setBatchSize(unsigned int batchSize)
{
    mBatchSize = batchSize;

    if (mBatchSize > 0) {
        const unsigned int nbChannels = (!mChannelsTransformations.empty())
                                            ? mChannelsTransformations.size()
                                            : mNbChannels;
        const unsigned int nbChannelsLabels
            = (!mChannelsTransformations.empty())
                  ? mChannelsTransformations.size()
                  : 1;

        mData.resize(mSizeX, mSizeY, nbChannels, mBatchSize);
        mFutureData.resize(mSizeX, mSizeY, nbChannels, mBatchSize);

        if (mCompositeStimuli) {
            mLabelsData.resize(mSizeX, mSizeY, nbChannelsLabels, batchSize);
            mFutureLabelsData.resize(
                mSizeX, mSizeY, nbChannelsLabels, batchSize);
        } else {
            mLabelsData.resize(1, 1, nbChannelsLabels, batchSize);
            mFutureLabelsData.resize(1, 1, nbChannelsLabels, batchSize);
        }
    }
}

N2D2::Tensor3d<N2D2::Float_T>
N2D2::StimuliProvider::readRawData(Database::StimulusID id) const
{
    const cv::Mat mat = mDatabase.getStimulusData(id);

    cv::Mat mat64F;
    mat.convertTo(mat64F, CV_64F);
    return Tensor3d<Float_T>(mat64F);
}

void N2D2::StimuliProvider::setCachePath(const std::string& path)
{
    if (!path.empty())
        Utils::createDirectories(path);

    mCachePath = path;
}

unsigned int
N2D2::StimuliProvider::getNbTransformations(Database::StimuliSet set) const
{
    unsigned int nbTransformations = mTransformations(set).cacheable.size()
                                     + mTransformations(set).onTheFly.size();

    for (std::vector<TransformationsSets>::const_iterator it
         = mChannelsTransformations.begin(),
         itEnd = mChannelsTransformations.end();
         it != itEnd;
         ++it) {
        nbTransformations += (*it)(set).cacheable.size()
                             + (*it)(set).onTheFly.size();
    }

    return nbTransformations;
}

const N2D2::Tensor2d<N2D2::Float_T>
N2D2::StimuliProvider::getData(unsigned int channel,
                               unsigned int batchPos) const
{
    return Tensor2d<Float_T>(mData[batchPos][channel]);
}

const N2D2::Tensor2d<int>
N2D2::StimuliProvider::getLabelsData(unsigned int channel,
                                     unsigned int batchPos) const
{
    return Tensor2d<int>(mLabelsData[batchPos][channel]);
}

void N2D2::StimuliProvider::logData(const std::string& fileName,
                                    const Tensor2d<Float_T>& data)
{
    logData(fileName,
            Tensor3d
            <Float_T>(data.dimX(), data.dimY(), 1, data.begin(), data.end()));
}

void N2D2::StimuliProvider::logData(const std::string& fileName,
                                    const Tensor3d<Float_T>& data)
{
    if (data.dimX() == 1 && data.dimY() == 1 && data.dimZ() > 1) {
        logData(fileName,
                Tensor2d<Float_T>(data.dimZ(), 1, data.begin(), data.end()));
        return;
    }

    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data log file: " + fileName);

    std::vector<Float_T> minVal(data.dimZ(), data(0));
    std::vector<Float_T> maxVal(data.dimZ(), data(0));

    unsigned int dimX = data.dimX();
    unsigned int dimY = data.dimY();

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        const Tensor2d<Float_T> channel = data[z];

        if (dimX > 1 && dimY > 1) {
            // 2D data
            for (unsigned int y = 0; y < dimY; ++y) {
                for (unsigned int x = 0; x < dimX; ++x) {
                    minVal[z] = std::min(minVal[z], channel(x, y));
                    maxVal[z] = std::max(maxVal[z], channel(x, y));

                    dataFile << channel(x, y) << " ";
                }

                dataFile << "\n";
            }
        } else {
            // 1D data
            const unsigned int size = dimX * dimY;
            dimX = dimY = std::ceil(std::sqrt((double)size));
            unsigned int index = 0;

            for (unsigned int y = 0; y < dimY; ++y) {
                for (unsigned int x = 0; x < dimX; ++x) {
                    if (index < size) {
                        minVal[z] = std::min(minVal[z], channel(index));
                        maxVal[z] = std::max(maxVal[z], channel(index));

                        dataFile << channel(index) << " ";
                        ++index;
                    } else
                        dataFile << "0 ";
                }

                dataFile << "\n";
            }
        }

        dataFile << "\n";
    }

    dataFile.close();

    Gnuplot gnuplot(fileName + ".gnu");
    gnuplot.set("grid").set("key off");
    gnuplot.set("size ratio 1");
    gnuplot.setXrange(-0.5, dimX - 0.5);
    gnuplot.setYrange(-0.5, dimY - 0.5, "reverse");

    gnuplot << "if (!exists(\"multiplot\")) set xtics out nomirror";
    gnuplot << "if (!exists(\"multiplot\")) set ytics out nomirror";

    const std::string dirName = Utils::fileBaseName(fileName);
    const std::string baseName = Utils::baseName(dirName);

    if (data.dimZ() > 1)
        Utils::createDirectories(dirName);

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        std::stringstream cbRangeStr, paletteStr;
        cbRangeStr << "cbrange [";
        paletteStr << "palette defined (";

        if (minVal[z] < -1.0) {
            cbRangeStr << minVal[z];
            paletteStr << minVal[z] << " \"blue\", -1 \"cyan\", ";
        } else if (minVal[z] < 0.0) {
            cbRangeStr << -1.0;
            paletteStr << "-1 \"cyan\", ";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << ":";
        paletteStr << "0 \"black\"";

        if (maxVal[z] > 1.0) {
            cbRangeStr << maxVal[z];
            paletteStr << ", 1 \"white\", " << maxVal[z] << " \"red\"";
        } else if (maxVal[z] > 0.0 || !(minVal[z] < 0)) {
            cbRangeStr << 1.0;
            paletteStr << ", 1 \"white\"";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << "]";
        paletteStr << ")";

        gnuplot.set(paletteStr.str());
        gnuplot.set(cbRangeStr.str());

        if (data.dimZ() > 1) {
            std::stringstream channelName;
            channelName << dirName << "/" << baseName << "[" << z << "]."
                        << Utils::fileExtension(fileName);
            gnuplot.saveToFile(channelName.str());

            std::stringstream plotCmd;
            plotCmd << "index " << z << " matrix with image";
            gnuplot.plot(fileName, plotCmd.str());
        } else {
            gnuplot.saveToFile(fileName);
            gnuplot.plot(fileName, "matrix with image");
        }
    }

    gnuplot.close();

    if (data.dimZ() > 1) {
        const unsigned int size = std::ceil(std::sqrt((double)data.dimZ()));

        std::stringstream termStr;
        termStr << "if (!exists(\"multiplot\")) set term png size "
                << 100 * size << "," << 100 * size << " enhanced";

        Gnuplot multiplot;
        multiplot.saveToFile(fileName + ".dat");
        multiplot << termStr.str();
        multiplot.setMultiplot(size, size);
        multiplot.set("lmargin 0.1");
        multiplot.set("tmargin 0.1");
        multiplot.set("rmargin 0.1");
        multiplot.set("bmargin 0.1");
        multiplot.unset("xtics");
        multiplot.unset("ytics");
        multiplot.set("format x \"\"");
        multiplot.set("format y \"\"");
        multiplot.unset("colorbox");
        multiplot.readCmd(fileName + ".gnu");
    }
}

std::vector<cv::Mat> N2D2::StimuliProvider::loadDataCache(const std::string
                                                          & fileName) const
{
    std::ifstream is(fileName.c_str(), std::ios::binary);

    if (!is.good())
        throw std::runtime_error("Could not read cache file: " + fileName);

    std::vector<cv::Mat> data;

    do {
        data.push_back(cv::Mat());
        BinaryCvMat::read(is, data.back());
    } while (is.peek() != EOF);

    return data;
}

void N2D2::StimuliProvider::saveDataCache(const std::string& fileName,
                                          const std::vector
                                          <cv::Mat>& data) const
{
    std::ofstream os(fileName.c_str(), std::ios::binary);

    if (!os.good())
        throw std::runtime_error("Could not create cache file: " + fileName);

    for (std::vector<cv::Mat>::const_iterator it = data.begin(),
                                              itEnd = data.end();
         it != itEnd;
         ++it)
        BinaryCvMat::write(os, *it);
}
