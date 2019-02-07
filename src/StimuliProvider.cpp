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
#include "Solver/SGDSolver_Kernels.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Gnuplot.hpp"

N2D2::StimuliProvider::StimuliProvider(Database& database,
                                       const std::vector<size_t>& size,
                                       unsigned int batchSize,
                                       bool compositeStimuli)
    : // Variables
      mDataSignedMapping(this, "DataSignedMapping", false),
      mQuantizationLevels(this, "QuantizationLevels", 0U),
      mQuantizationMin(this, "QuantizationMin", 0.0),
      mQuantizationMax(this, "QuantizationMax", 1.0),
      mDatabase(database),
      mSize(size),
      mBatchSize(batchSize),
      mCompositeStimuli(compositeStimuli),
      mCachePath("_cache"),
      mBatch(batchSize),
      mFutureBatch(batchSize),
      mLabelsROI(batchSize, std::vector<std::shared_ptr<ROI> >()),
      mFutureLabelsROI(batchSize, std::vector<std::shared_ptr<ROI> >()),
      mFuture(false)
{
    // ctor
    Utils::createDirectories(mCachePath); // Create default cache directory

    std::vector<size_t> dataSize(mSize);
    dataSize.push_back(batchSize);

    mData.resize(dataSize);
    mFutureData.resize(dataSize);

    std::vector<size_t> labelSize(mSize);

    if (mCompositeStimuli) {
        // Last dimension is channel, mCompositeStimuli assumes unique label
        // for all channels by default
        labelSize.back() = 1;
    }
    else
        std::fill(labelSize.begin(), labelSize.end(), 1U);

    labelSize.push_back(batchSize);
    mLabelsData.resize(labelSize);
    mFutureLabelsData.resize(labelSize);
}

N2D2::StimuliProvider::StimuliProvider(const StimuliProvider& sp)
    : Parameterizable(sp),
      // Variables
      mDataSignedMapping(this, "DataSignedMapping", sp.mDataSignedMapping),
      mQuantizationLevels(this, "QuantizationLevels", sp.mQuantizationLevels),
      mQuantizationMin(this, "QuantizationMin", sp.mQuantizationMin),
      mQuantizationMax(this, "QuantizationMax", sp.mQuantizationMax),
      mDatabase(sp.mDatabase),
      mSize(sp.mSize),
      mBatchSize(sp.mBatchSize),
      mCompositeStimuli(sp.mCompositeStimuli),
      mCachePath(sp.mCachePath),
      mTransformations(sp.mTransformations),
      mChannelsTransformations(sp.mChannelsTransformations),
      mBatch(sp.mBatch),
      mFutureBatch(sp.mFutureBatch),
      mData(sp.mData),
      mFutureData(sp.mFutureData),
      mLabelsData(sp.mLabelsData),
      mFutureLabelsData(sp.mFutureLabelsData),
      mLabelsROI(sp.mLabelsROI),
      mFutureLabelsROI(sp.mFutureLabelsROI),
      mFuture(sp.mFuture)
{
    // copy-ctor
}

void N2D2::StimuliProvider::addChannel(const CompositeTransformation
                                       & /*transformation*/)
{
    if (mChannelsTransformations.empty())
        mSize.back() = 1;
    else
        ++mSize.back();

    std::vector<size_t> dataSize(mSize);
    dataSize.push_back(mBatchSize);

    mData.resize(dataSize);
    mFutureData.resize(dataSize);

    if (!mChannelsTransformations.empty()) {
        std::vector<size_t> labelSize(mLabelsData.dims());
        labelSize.pop_back();
        ++labelSize.back();
        labelSize.push_back(mBatchSize);

        mLabelsData.resize(labelSize);
        mFutureLabelsData.resize(labelSize);
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

    unsigned int exceptCatch = 0;

#pragma omp parallel for schedule(dynamic) if (mBatchSize > 1)
    for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
        try {
            readStimulus(batchRef[batchPos], set, batchPos);
        }
        catch (const std::exception& e)
        {
            #pragma omp critical(StimuliProvider__readRandomBatch)
            {
                std::cout << Utils::cwarning << e.what() << Utils::cdef
                    << std::endl;
                ++exceptCatch;
            }
        }
    }

    if (exceptCatch > 0) {
        std::cout << "Retry without multi-threading..." << std::endl;

        for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos)
            readStimulus(batchRef[batchPos], set, batchPos);
    }
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

#pragma omp parallel for schedule(dynamic) if (batchSize > 1)
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

    Tensor<Float_T> data = (mChannelsTransformations.empty())
                       ? Tensor<Float_T>(rawChannelsData[0], mDataSignedMapping)
                       : Tensor<Float_T>(std::vector<size_t>(mSize.size(), 0));
    Tensor<int> labels = (mChannelsTransformations.empty())
                        ? Tensor<int>(rawChannelsLabels[0])
                        : Tensor<int>(std::vector<size_t>(mSize.size(), 0));

    if (data.nbDims() < mSize.size()) {
        // rawChannelsData[0] can be 2D or 3D
        std::vector<size_t> dataSize(data.dims());
        dataSize.resize(mSize.size(), 1);
        data.reshape(dataSize);
    }

    if (labels.nbDims() < mSize.size()) {
        std::vector<size_t> labelsSize(labels.dims());
        labelsSize.resize(mSize.size(), 1);
        labels.reshape(labelsSize);
    }

    // 2.1 Process channels
    if (!mChannelsTransformations.empty()) {
        for (std::vector<TransformationsSets>::iterator it
             = mChannelsTransformations.begin(),
             itBegin = mChannelsTransformations.begin(),
             itEnd = mChannelsTransformations.end();
             it != itEnd;
             ++it) {
            cv::Mat channelDataMat((rawChannelsData.size() > 1)
                                    ? rawChannelsData[it - itBegin].clone()
                                    : rawChannelsData[0].clone());
            cv::Mat channelLabelsMat
                = ((rawChannelsLabels.size() > 1)
                       ? rawChannelsLabels[it - itBegin].clone()
                       : rawChannelsLabels[0].clone());

            if (!mTransformations(set).onTheFly.empty())
                (*it)(set).cacheable.apply(channelDataMat, channelLabelsMat, id);

            (*it)(set).onTheFly.apply(channelDataMat, channelLabelsMat, id);

            Tensor<Float_T> channelData(channelDataMat, mDataSignedMapping);
            Tensor<int> channelLabels(channelLabelsMat);

            if (channelData.nbDims() < mSize.size() - 1) {
                std::vector<size_t> dataSize(channelData.dims());
                dataSize.resize(mSize.size() - 1, 1);
                channelData.reshape(dataSize);
            }

            if (channelLabels.nbDims() < mSize.size() - 1) {
                std::vector<size_t> labelsSize(channelLabels.dims());
                labelsSize.resize(mSize.size() - 1, 1);
                channelLabels.reshape(labelsSize);
            }

            data.push_back(channelData);
            labels.push_back(channelLabels);
        }
    }

    Tensor<Float_T>& dataRef = (mFuture) ? mFutureData : mData;
    Tensor<int>& labelsRef = (mFuture) ? mFutureLabelsData : mLabelsData;

    if (mBatchSize > 0) {
        if (!std::equal(data.dims().begin(), data.dims().end(),
                   dataRef.dims().begin()))
        {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected data size is "
                << dataRef.dims() << ", but size after transformations is "
                << data.dims();

            throw std::runtime_error(msg.str());
        }

        if (mQuantizationLevels > 0) {
            quantize(data,
                     data,
                     (Float_T)mQuantizationMin,
                     (Float_T)mQuantizationMax,
                     mQuantizationLevels,
                     true);
        }

        dataRef[batchPos] = data;

        if (!std::equal(labels.dims().begin(), labels.dims().end(),
                   labelsRef.dims().begin()))
        {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected labels size is "
                << labelsRef.dims() << ", but size after transformations is "
                << labels.dims();

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
    Tensor<Float_T>& dataRef = (mFuture) ? mFutureData : mData;

    // Apply global transformation
    cv::Mat rawData = mat.clone();
    mTransformations(set).cacheable.apply(rawData);
    mTransformations(set).onTheFly.apply(rawData);

    Tensor<Float_T> data = (mChannelsTransformations.empty())
                                 ? Tensor<Float_T>(rawData)
                                 : Tensor<Float_T>();

    if (data.nbDims() < mSize.size()
            && mChannelsTransformations.empty()) {
        // rawChannelsData[0] can be 2D or 3D
        std::vector<size_t> dataSize(data.dims());
        dataSize.resize(mSize.size(), 1);
        data.reshape(dataSize);
    }

    if (!mChannelsTransformations.empty()) {
        // Apply channels transformations
        for (std::vector<TransformationsSets>::iterator it
             = mChannelsTransformations.begin(),
             itEnd = mChannelsTransformations.end();
             it != itEnd;
             ++it) {
            cv::Mat channelDataMat(rawData.clone());
            (*it)(set).cacheable.apply(channelDataMat);
            (*it)(set).onTheFly.apply(channelDataMat);

            Tensor<Float_T> channelData(channelDataMat);

            if (channelData.nbDims() < mSize.size() - 1) {
                std::vector<size_t> dataSize(channelData.dims());
                dataSize.resize(mSize.size() - 1, 1);
                channelData.reshape(dataSize);
            }

            data.push_back(channelData);
        }
    }

    dataRef[batchPos] = data;
}

void N2D2::StimuliProvider::reverseLabels(const cv::Mat& mat,
                                          Database::StimuliSet set,
                                          Tensor<int>& labels,
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
    labels = Tensor<int>(labelsMat);
}

void N2D2::StimuliProvider::setBatchSize(unsigned int batchSize)
{
    mBatchSize = batchSize;

    if (mBatchSize > 0) {
        std::vector<size_t> dataSize(mData.dims());
        dataSize.back() = mBatchSize;

        mData.resize(dataSize);
        mFutureData.resize(dataSize);

        std::vector<size_t> labelSize(mLabelsData.dims());
        labelSize.back() = mBatchSize;

        mLabelsData.resize(labelSize);
        mFutureLabelsData.resize(labelSize);
    }
}

N2D2::Tensor<N2D2::Float_T>
N2D2::StimuliProvider::readRawData(Database::StimulusID id) const
{
    const cv::Mat mat = mDatabase.getStimulusData(id);

    cv::Mat matF;
    mat.convertTo(matF, opencv_data_type<Float_T>::value);
    return Tensor<Float_T>(matF);
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

const N2D2::Tensor<N2D2::Float_T>
N2D2::StimuliProvider::getData(unsigned int channel,
                               unsigned int batchPos) const
{
    return Tensor<Float_T>(mData[batchPos][channel]);
}

const N2D2::Tensor<int>
N2D2::StimuliProvider::getLabelsData(unsigned int channel,
                                     unsigned int batchPos) const
{
    return Tensor<int>(mLabelsData[batchPos][channel]);
}

void N2D2::StimuliProvider::logData(const std::string& fileName,
                                    Tensor<Float_T> data)
{
    if (data.dims().size() == 2)
        data.reshape({data.dimX(), data.dimY(), 1});
    else if (data.dims().size() == 3
             && (data.dimX() == 1 && data.dimY() == 1 && data.dimZ() > 1))
    {
        data.reshape({data.dimZ(), 1, 1});
    }
    else if (data.dims().size() > 3) {
        throw std::runtime_error("Could not log Tensor of dimension > 3");
    }

    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data log file: " + fileName);

    std::vector<Float_T> minVal(data.dimZ(), data(0));
    std::vector<Float_T> maxVal(data.dimZ(), data(0));

    unsigned int dimX = data.dimX();
    unsigned int dimY = data.dimY();

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        const Tensor<Float_T>& channel = data[z];

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

// This implementation plots grey level matrices. Only used in spiking version
void N2D2::StimuliProvider::logData(const std::string& fileName,
                                    Tensor<Float_T> data,
                                    const double minValue,
                                    const double maxValue)
{
    if (data.dims().size() == 2)
        data.reshape({data.dimX(), data.dimY(), 1});
    else if (data.dims().size() == 3
             && (data.dimX() == 1 && data.dimY() == 1 && data.dimZ() > 1))
    {
        data.reshape({data.dimZ(), 1, 1});
    }
    else if (data.dims().size() > 3) {
        throw std::runtime_error("Could not log Tensor of dimension > 3");
    }

    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data log file: " + fileName);

    std::vector<Float_T> minVal(data.dimZ(), data(0));
    std::vector<Float_T> maxVal(data.dimZ(), data(0));

    unsigned int dimX = data.dimX();
    unsigned int dimY = data.dimY();

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        const Tensor<Float_T>& channel = data[z];

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

         if (minValue < 0.0) {
            cbRangeStr << minValue;
            paletteStr << minValue << " \"cyan\", ";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << ":";
        paletteStr << "0 \"black\"";

        if (maxValue > 0.0 || !(minValue < 0)) {
            cbRangeStr << maxValue;
            paletteStr << ", " << maxValue << " \"white\"";
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


void N2D2::StimuliProvider::logDataMatrix(const std::string& fileName,
                                    const Tensor<Float_T>& data,
                                    const double minValue,
                                    const double maxValue)
{
    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data log file: " + fileName);

    std::vector<Float_T> minVal(data.dimZ(), data(0));
    std::vector<Float_T> maxVal(data.dimZ(), data(0));

    unsigned int dimX = data.dimX();
    unsigned int dimY = data.dimY();

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        const Tensor<Float_T> channel = data[z];

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
        }
        else {
            throw std::runtime_error("StimuliProvider::logDataMatrix: "
                                     "dimension should be bigger than 1");
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
        cbRangeStr << "cbrange ["; //<< minValue << ":" << maxValue << "]";
        paletteStr << "palette defined ("; //<< minValue << "" << ")";

        if (minValue < 0.0) {
            cbRangeStr << minValue;
            paletteStr << minValue << " \"black\", ";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << ":";
        paletteStr << "0 \"grey\"";

        if (maxValue > 0.0 || !(minValue < 0)) {
            cbRangeStr << maxValue;
            paletteStr << ", " << maxValue << " \"white\"";
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

        }
        else {
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


// Required for IF class with colours
// Not reimplemented yet
/*
void N2D2::StimuliProvider::logRgbData(const std::string& fileName,
                                    const Tensor4d<Float_T>& data)
{
    Gnuplot plotFilter(fileName + "-filters.gnu");
    plotFilter.set("grid").set("key off");
    plotFilter.set("size ratio 1");
    plotFilter.setXrange(-0.5, data.dimX() - 0.5);
    plotFilter.setYrange(data.dimY() - 0.5, -0.5);
    /// WARNING: Bug in gnuplot 5 makes 'reverse' useless
    //plotFilter.setYrange(-0.5, data.dimY() - 0.5, "reverse");
    plotFilter << "if (!exists(\"multiplot\")) set xtics out nomirror";
    plotFilter << "if (!exists(\"multiplot\")) set ytics out nomirror";

    for (unsigned int ftr=0; ftr<data.dimB(); ++ftr){
        //std::ofstream dataFile(fileName.c_str());
        //if (!dataFile.good())
        //    throw std::runtime_error("Could not create data log file: " + fileName);

        //unsigned colors[] = {"red", "green", "blue"};

        std::stringstream fileNameFilter;
        fileNameFilter << fileName << "-" << ftr;
        std::string file = fileNameFilter.str();

        const std::string dirName = Utils::fileBaseName(file);
        const std::string baseName = Utils::baseName(dirName);

        if (data.dimZ() > 1)
            Utils::createDirectories(dirName);

        unsigned int dimX = data.dimX();
        unsigned int dimY = data.dimY();

        Gnuplot gnuplot(file + ".gnu");
        gnuplot.set("grid").set("key off");
        gnuplot.set("size ratio 1");
        gnuplot.setXrange(-0.5, dimX - 0.5);
        gnuplot.setYrange(dimY - 0.5, -0.5);
        /// WARNING: Bug in gnuplot 5 makes 'reverse' useless
        //gnuplot.setYrange(-0.5, dimY - 0.5, "reverse");

        gnuplot << "if (!exists(\"multiplot\")) set xtics out nomirror";
        gnuplot << "if (!exists(\"multiplot\")) set ytics out nomirror";

        // Gnuplot object for filter plot
        //Gnuplot plotFilter(fileName + "-filters.gnu");
        //plotFilter.set("grid").set("key off");
        //plotFilter.set("size ratio 1");
        //plotFilter.setXrange(-0.5, dimX - 0.5);
        //plotFilter.setYrange(-0.5, dimY - 0.5, "reverse");

        //plotFilter << "if (!exists(\"multiplot\")) set xtics out nomirror";
        //plotFilter << "if (!exists(\"multiplot\")) set ytics out nomirror";

        for (unsigned int z = 0; z < data.dimZ(); ++z) {

            std::stringstream channelName;
            channelName << dirName << "/" << baseName << "[" << z << "]" << ".dat";
            std::ofstream rgbChannelData(channelName.str());
            if (!rgbChannelData.good())
                throw std::runtime_error("Could not create data log file: "
                                    + channelName.str());

            // Generate rgb data for each channel and save in file
            for (unsigned int y = 0; y < dimY; ++y) {
               for (unsigned int x = 0; x < dimX; ++x) {
                    rgbChannelData << x << " " << y << " ";
                    const Tensor2d<Float_T> channel = data[ftr][z];
                    for (unsigned int k=0; k<3; ++k){
                        if (z == k){
                            rgbChannelData << (int) channel(x, y) << " ";
                        }
                        else {
                            rgbChannelData << "0" << " ";
                        }
                    }
                    rgbChannelData << "\n";
                }
            }
            rgbChannelData.close();

            // Add output command to plot file
            gnuplot.saveToFile(channelName.str());

            // Add plot command to plot file
            std::stringstream plotCmd;
            plotCmd << " using 1:2:3:4:5 with rgbimage";
            gnuplot.plot(channelName.str(), plotCmd.str());
        }

        //gnuplot.close();

        //std::string colors[] = {"red", "green", "blue"};

        std::stringstream rgbFileName;
        rgbFileName << dirName << "/" << baseName << "-rgb.dat";
        std::ofstream rgbDataFile(rgbFileName.str());
        if (!rgbDataFile.good())
            throw std::runtime_error("Could not create data log file: "
                                    + rgbFileName.str());

        // rgb data
        for (unsigned int y = 0; y < dimY; ++y) {
            for (unsigned int x = 0; x < dimX; ++x) {
                rgbDataFile << x << " " << y << " ";
                for (unsigned int z = 0; z < data.dimZ(); ++z) {
                    const Tensor2d<Float_T> channel = data[ftr][z];
                    rgbDataFile << (int) channel(x, y) << " ";
                }
                rgbDataFile << "\n";
            }
        }

        //std::stringstream gnuName;
        //gnuName << dirName << "/" << baseName << "-rgb.gnu";
        //Gnuplot rgbPlot(gnuName.str());
        //rgbPlot.set("grid").set("key off");
        //rgbPlot.set("size ratio 1");
        //rgbPlot.setXrange(-0.5, dimX - 0.5);
        //rgbPlot.setYrange(-0.5, dimY - 0.5, "reverse");

        //rgbPlot << "if (!exists(\"multiplot\")) set xtics out nomirror";
        //rgbPlot << "if (!exists(\"multiplot\")) set ytics out nomirror";

        std::stringstream channelName;
        channelName << dirName << "/" << baseName << "-rgb.dat";
        gnuplot.saveToFile(channelName.str());
        plotFilter.saveToFile(channelName.str());

        std::stringstream plotCmd;
        plotCmd << "using 1:2:3:4:5 with rgbimage";
        std::stringstream rgbImageName;
        rgbImageName << channelName.str();
        gnuplot.plot(rgbImageName.str(), plotCmd.str());
        plotFilter.plot(rgbImageName.str(), plotCmd.str());

        gnuplot.close();
        //plotFilter.close();


        // Make 4x4 multiplots
        const unsigned int size = std::ceil(std::sqrt((double)data.dimZ()));

        std::stringstream termStr;
        termStr << "if (!exists(\"multiplot\")) set term png size "
                << 100 * size << "," << 100 * size << " enhanced";

        // This opens the gnu file again, and executes the plots there again
        // with the multiplot variables
        Gnuplot multiplot;
        multiplot.saveToFile(file + ".dat");
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
        multiplot.readCmd(file + ".gnu");
        multiplot.close();
    }

    plotFilter.close();

    // Plot all filters in multiplot
    const unsigned int size = std::ceil(std::sqrt((double)data.dimB()));

    std::stringstream termStr;
    termStr << "if (!exists(\"multiplot\")) set term png size "
            << 100 * size << "," << 100 * size << " enhanced";

    // This opens the gnu file again, and executes the plots there again
    // with the multiplot variables

    Gnuplot filterMultiplot;
    filterMultiplot.saveToFile(fileName + "-filters.dat");
    filterMultiplot << termStr.str();
    filterMultiplot.setMultiplot(size, size);
    filterMultiplot.set("lmargin 0.1");
    filterMultiplot.set("tmargin 0.1");
    filterMultiplot.set("rmargin 0.1");
    filterMultiplot.set("bmargin 0.1");
    filterMultiplot.unset("xtics");
    filterMultiplot.unset("ytics");
    filterMultiplot.set("format x \"\"");
    filterMultiplot.set("format y \"\"");
    filterMultiplot.unset("colorbox");
    filterMultiplot.readCmd(fileName + "-filters.gnu");
    filterMultiplot.close();


}
*/

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
