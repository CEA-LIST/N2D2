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
#include "utils/GraphViz.hpp"

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
      mCachePath(""),
      mBatch(batchSize),
      mFutureBatch(batchSize),
#ifdef CUDA
      // mData and mFutureData are host-based by default.
      // This can be changed with the hostBased() method if data is directly
      // supplied to mData's device pointer.
      mData(true),
      mFutureData(true),
      mTargetData(true),
      mFutureTargetData(true),
#endif
      mLabelsROI(std::max(batchSize, 1u), std::vector<std::shared_ptr<ROI> >()),
      mFutureLabelsROI(std::max(batchSize, 1u), std::vector<std::shared_ptr<ROI> >()),
      mFuture(false)
{
    // ctor
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

N2D2::StimuliProvider::StimuliProvider(StimuliProvider&& other)
    : mDataSignedMapping(this, "DataSignedMapping", other.mDataSignedMapping),
      mQuantizationLevels(this, "QuantizationLevels", other.mQuantizationLevels),
      mQuantizationMin(this, "QuantizationMin", other.mQuantizationMin),
      mQuantizationMax(this, "QuantizationMax", other.mQuantizationMax),
      mDatabase(other.mDatabase),
      mSize(std::move(other.mSize)),
      mBatchSize(other.mBatchSize),
      mCompositeStimuli(other.mCompositeStimuli),
      mCachePath(std::move(other.mCachePath)),
      mTransformations(other.mTransformations),
      mChannelsTransformations(std::move(other.mChannelsTransformations)),
      mBatch(std::move(other.mBatch)),
      mFutureBatch(std::move(other.mFutureBatch)),
      mData(other.mData),
      mFutureData(other.mFutureData),
      mLabelsData(other.mLabelsData),
      mFutureLabelsData(other.mFutureLabelsData),
      mTargetData(other.mTargetData),
      mFutureTargetData(other.mFutureTargetData),
      mLabelsROI(std::move(other.mLabelsROI)),
      mFutureLabelsROI(std::move(other.mFutureLabelsROI)),
      mFuture(other.mFuture)
{
}

N2D2::StimuliProvider N2D2::StimuliProvider::cloneParameters() const {
    StimuliProvider sp(mDatabase, mSize, mBatchSize, mCompositeStimuli);
    sp.mDataSignedMapping = mDataSignedMapping;
    sp.mQuantizationLevels = mQuantizationLevels;
    sp.mQuantizationMin = mQuantizationMin;
    sp.mQuantizationMax = mQuantizationMax;
    sp.mCachePath = mCachePath;
    sp.mTransformations = mTransformations;
    sp.mChannelsTransformations = mChannelsTransformations;

    return sp;
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

void N2D2::StimuliProvider::addTopTransformation(const CompositeTransformation& transformation,
                                                 Database::StimuliSetMask setMask)
{
    if(!mChannelsTransformations.empty()) {
        addChannelsOnTheFlyTransformation(transformation, setMask);
    }
    else {
        addOnTheFlyTransformation(transformation, setMask);
    }
}

void N2D2::StimuliProvider::logTransformations(const std::string& fileName)
    const
{
    GraphViz graph("Transformations", "", true);

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(Database::All);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(), itEnd = stimuliSets.end(); it != itEnd; ++it)
    {
        auto transParams = [](GraphViz& setGraph,
                              const std::string& nodeName,
                              const std::map<std::string, std::string>& params)
        {
            if (!params.empty()) {
                std::ostringstream paramsStr;
                paramsStr << "{";

                for (std::map<std::string, std::string>::const_iterator itParams
                    = params.begin(), itParamsEnd = params.end();
                    itParams != itParamsEnd; ++itParams)
                {
                    if (itParams != params.begin())
                        paramsStr << "|";

                    paramsStr << (*itParams).first;
                }

                paramsStr << "}|{";

                for (std::map<std::string, std::string>::const_iterator itParams
                    = params.begin(), itParamsEnd = params.end();
                    itParams != itParamsEnd; ++itParams)
                {
                    if (itParams != params.begin())
                        paramsStr << "|";

                    paramsStr << (*itParams).second;
                }

                paramsStr << "}";

                setGraph.node(nodeName + "_params", paramsStr.str());
                setGraph.attr(nodeName + "_params", "shape", "record");
                setGraph.attr(nodeName + "_params", "color", "gray");
                setGraph.attr(nodeName + "_params", "fontsize", "8");
            }
        };

        auto transNode = [transParams](GraphViz& graph,
                                       std::shared_ptr<Transformation> trans,
                                       unsigned int& width,
                                       unsigned int& height,
                                       const std::string& prevNodeName,
                                       const std::string& nodeName,
                                       bool isCacheable)
        {
            std::tie(width, height) = trans->getOutputsSize(width, height);

            std::ostringstream labelStr;
            labelStr << trans->getType() << "\n";

            if (width != 0 && height != 0)
                labelStr << width << "x" << height;
            else
                labelStr << "?x?";

            GraphViz nodeGraph("cluster" + nodeName);
            nodeGraph.attr("cluster" + nodeName, "style", "invis");
            nodeGraph.node(nodeName, labelStr.str());

            if (trans->getType() == std::string("ChannelExtraction")) {
                nodeGraph.attr(nodeName, "shape", "invtrapezium");
                nodeGraph.attr(nodeName, "width", "2");
                nodeGraph.attr(nodeName, "fixedsize", "true");
            }
            else
                nodeGraph.attr(nodeName, "shape", "rect");

            if (isCacheable)
                nodeGraph.attr(nodeName, "style", "filled");

            transParams(nodeGraph, nodeName, trans->getParameters());

            graph.subgraph(nodeGraph);
            graph.edge(prevNodeName, nodeName);
        };

        unsigned int width = 0;
        unsigned int height = 0;

        // Dataset node
        std::string nodeName = Utils::toString(*it);
        std::string prevNodeName;

        graph.node(nodeName);
        graph.attr(nodeName, "shape", "cylinder");
        graph.attr(nodeName, "style", "filled");
        graph.attr(nodeName, "fillcolor", "yellow");

        // Output DATA node
        const std::string dataNodeName = "DATA" + nodeName;

        std::ostringstream labelStr;
        labelStr << "DATA\n"
            << mSize;

        graph.node(dataNodeName, labelStr.str());

        if (mSize.back() > 1)
            graph.attr(dataNodeName, "shape", "box3d");
        else
            graph.attr(dataNodeName, "shape", "box");

        graph.attr(dataNodeName, "height", "0.75");
        graph.attr(dataNodeName, "width", "1.5");
        graph.attr(dataNodeName, "fixedsize", "true");

        // Global transformations
        const CompositeTransformation& trans = mTransformations(*it).cacheable;

        for (size_t i = 0; i < trans.size(); ++i) {
            std::swap(prevNodeName, nodeName);
            nodeName = Utils::toString(*it)
                + "_cacheable_" + std::to_string(i);

            transNode(graph, trans[i], width, height,
                      prevNodeName, nodeName, true);

            if (mSize.back() > 1) {
                graph.attr(prevNodeName + "->" + nodeName,
                           "color", "black:invis:black");
            }
        }

        const CompositeTransformation& transFly
            = mTransformations(*it).onTheFly;

        for (size_t i = 0; i < transFly.size(); ++i) {
            std::swap(prevNodeName, nodeName);
            nodeName = Utils::toString(*it)
                + "_onthefly_" + std::to_string(i);

            transNode(graph, transFly[i], width, height,
                      prevNodeName, nodeName, false);

            if (mSize.back() > 1) {
                graph.attr(prevNodeName + "->" + nodeName,
                           "color", "black:invis:black");
            }
        }

        const std::string globalNodeName = nodeName;

        // Channel transformations
        for (size_t channel = 0; channel < mChannelsTransformations.size();
            ++channel)
        {
            nodeName = globalNodeName;

            const CompositeTransformation& chTrans 
                = mChannelsTransformations[channel](*it).cacheable;

            for (size_t i = 0; i < chTrans.size(); ++i) {
                std::swap(prevNodeName, nodeName);
                nodeName = Utils::toString(*it)
                    + "_ch" + std::to_string(channel)
                    + "_cacheable_" + std::to_string(i);

                transNode(graph, chTrans[i], width, height,
                          prevNodeName, nodeName, true);
            }

            const CompositeTransformation& chTransFly
                = mChannelsTransformations[channel](*it).onTheFly;

            for (size_t i = 0; i < chTransFly.size(); ++i) {
                std::swap(prevNodeName, nodeName);
                nodeName = Utils::toString(*it)
                    + "_ch" + std::to_string(channel)
                    + "_onthefly_" + std::to_string(i);

                transNode(graph, chTransFly[i], width, height,
                          prevNodeName, nodeName, false);
            }

            graph.edge(nodeName, dataNodeName);
        }

        if (mChannelsTransformations.empty())
            graph.edge(nodeName, dataNodeName);
    }

    graph.render(fileName);
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
        mTargetData.swap(mFutureTargetData);
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
    if (startIndex >= mDatabase.getNbStimuli(set)) {
        std::stringstream msg;
        msg << "StimuliProvider::readBatch(): startIndex (" << startIndex
            << ") is higher than the number of stimuli in the " << set
            << " set (" << mDatabase.getNbStimuli(set) << ")";

        throw std::runtime_error(msg.str());
    }

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

void N2D2::StimuliProvider::streamBatch(int startIndex) {
    if (startIndex < 0)
        startIndex = mBatch.back() + 1;

    std::iota(mBatch.begin(), mBatch.end(), startIndex);
}

void N2D2::StimuliProvider::readStimulusBatch(Database::StimuliSet set,
                                              Database::StimulusID id)
{
    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    readStimulus(id, set, 0);

    batchRef[0] = id;
    std::fill(batchRef.begin() + 1, batchRef.end(), -1);
}

void N2D2::StimuliProvider::readStimulus(Database::StimulusID id,
                                         Database::StimuliSet set,
                                         unsigned int batchPos)
{
    std::stringstream dataCacheFile, labelsCacheFile, validCacheFile;
    dataCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                  << id << "_data_" << set << ".bin";
    labelsCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                    << id << "_labels_" << set << ".bin";
    validCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                    << id << "_" << set << ".valid";

    std::vector<std::shared_ptr<ROI> >& labelsROI
        = (mFuture) ? mFutureLabelsROI[batchPos] : mLabelsROI[batchPos];
    labelsROI = mDatabase.getStimulusROIs(id);

    std::vector<cv::Mat> rawChannelsData;
    std::vector<cv::Mat> rawChannelsLabels;

    // 1. Cached data
    if (!mCachePath.empty() && std::ifstream(validCacheFile.str()).good()) {
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
            std::ofstream(validCacheFile.str());
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
    Tensor<Float_T> targetData = (!mTargetSize.empty())
        ? Tensor<Float_T>(mDatabase.getStimulusTargetData(id,
                                                          rawChannelsData[0],
                                                          rawChannelsLabels[0],
                                                          labelsROI)
            .clone())  // make sure the database image will not be altered
        : Tensor<Float_T>();

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

    if (targetData.nbDims() < mTargetSize.size()) {
        std::vector<size_t> targetDataSize(targetData.dims());
        targetDataSize.resize(mTargetSize.size(), 1);
        targetData.reshape(targetDataSize);
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

    TensorData_T& dataRef = (mFuture) ? mFutureData : mData;
    Tensor<int>& labelsRef = (mFuture) ? mFutureLabelsData : mLabelsData;
    TensorData_T& targetDataRef = (mFuture) ? mFutureTargetData : mTargetData;

    if (mBatchSize > 0) {
        TensorData_T dataRefPos = dataRef[batchPos];
        Tensor<int> labelsRefPos = labelsRef[batchPos];

        if (data.dims() != dataRefPos.dims()) {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected data size is "
                << dataRefPos.dims() << ", but size after transformations is "
                << data.dims() << " for stimulus: "
                << mDatabase.getStimulusName(id);

#pragma omp critical
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

        dataRefPos = data;

        if (labels.dims() != labelsRefPos.dims()) {
            std::stringstream msg;
            msg << "StimuliProvider::readStimulus(): expected labels size is "
                << labelsRefPos.dims() << ", but size after transformations is "
                << labels.dims() << " for stimulus: "
                << mDatabase.getStimulusName(id);

#pragma omp critical
            throw std::runtime_error(msg.str());
        }

        labelsRefPos = labels;

        if (!targetDataRef.empty()) {
            TensorData_T targetDataRefPos = targetDataRef[batchPos];

            if (targetData.dims() != targetDataRefPos.dims()) {
                std::stringstream msg;
                msg << "StimuliProvider::readStimulus(): expected target data "
                    "size is " << targetDataRefPos.dims() << ", but size is "
                    << targetData.dims() << " for stimulus: "
                    << mDatabase.getStimulusName(id);

#pragma omp critical
                throw std::runtime_error(msg.str());
            }

            targetDataRefPos = targetData;
        }
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
    TensorData_T& dataRef = (mFuture) ? mFutureData : mData;

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

void N2D2::StimuliProvider::setTargetSize(const std::vector<size_t>& size) {
    mTargetSize = size;

    std::vector<size_t> targetSize(size);
    targetSize.push_back(mBatchSize);

    mTargetData.resize(targetSize);
    mFutureTargetData.resize(targetSize);
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
    if (!path.empty()) {
        if (!Utils::createDirectories(path)) {
            throw std::runtime_error("StimuliProvider::setCachePath(): "
                                     "Could not create directory: " + path);
        }
    }

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

const N2D2::StimuliProvider::TensorData_T
N2D2::StimuliProvider::getData(unsigned int channel,
                               unsigned int batchPos) const
{
    return TensorData_T(mData[batchPos][channel]);
}

const N2D2::Tensor<int>
N2D2::StimuliProvider::getLabelsData(unsigned int channel,
                                     unsigned int batchPos) const
{
    return Tensor<int>(mLabelsData[batchPos][channel]);
}

const N2D2::StimuliProvider::TensorData_T
N2D2::StimuliProvider::getTargetData(unsigned int channel,
                                     unsigned int batchPos) const
{
    return TensorData_T((!mTargetData.empty())
        ? mTargetData[batchPos][channel]
        : mData[batchPos][channel]);
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

    std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

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

    dirName = Utils::fileBaseName(fileName);
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

    std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

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

    dirName = Utils::fileBaseName(fileName);
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
    std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

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

    dirName = Utils::fileBaseName(fileName);
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


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_StimuliProvider(py::module &m) {
    py::class_<StimuliProvider, std::shared_ptr<StimuliProvider>>(m, "StimuliProvider", py::multiple_inheritance())
    .def(py::init<Database&, const std::vector<size_t>&, unsigned int, bool>(), py::arg("database"), py::arg("size"), py::arg("batchSize") = 1, py::arg("compositeStimuli") = false)
    .def("cloneParameters", &StimuliProvider::cloneParameters)
    .def("logTransformations", &StimuliProvider::logTransformations, py::arg("fileName"))
    .def("future", &StimuliProvider::future)
    .def("synchronize", &StimuliProvider::synchronize)
    .def("getRandomIndex", &StimuliProvider::getRandomIndex, py::arg("set"))
    .def("getRandomID", &StimuliProvider::getRandomID, py::arg("set"))
    .def("readRandomBatch", &StimuliProvider::readRandomBatch, py::arg("set"))
    .def("readRandomStimulus", &StimuliProvider::readRandomStimulus, py::arg("set"), py::arg("batchPos") = 0)
    .def("readBatch", &StimuliProvider::readBatch, py::arg("set"), py::arg("startIndex") = 0)
    .def("streamBatch", &StimuliProvider::streamBatch, py::arg("startIndex") = -1)
    .def("readStimulusBatch", &StimuliProvider::readStimulusBatch, py::arg("set"), py::arg("id"))
    .def("readStimulus", (void (StimuliProvider::*)(Database::StimulusID, Database::StimuliSet, unsigned int)) &StimuliProvider::readStimulus, py::arg("id"), py::arg("set"), py::arg("batchPos") = 0)
    .def("readStimulus", (Database::StimulusID (StimuliProvider::*)(Database::StimuliSet, unsigned int, unsigned int)) &StimuliProvider::readStimulus, py::arg("set"), py::arg("index"), py::arg("batchPos") = 0)
    .def("readRawData", (Tensor<Float_T> (StimuliProvider::*)(Database::StimulusID) const) &StimuliProvider::readRawData, py::arg("id"))
    .def("readRawData", (Tensor<Float_T> (StimuliProvider::*)(Database::StimuliSet, unsigned int) const) &StimuliProvider::readRawData, py::arg("set"), py::arg("index"))
    .def("setBatchSize", &StimuliProvider::setBatchSize, py::arg("batchSize"))
    .def("setCachePath", &StimuliProvider::setCachePath, py::arg("path") = "")
    .def("getDatabase", (Database& (StimuliProvider::*)()) &StimuliProvider::getDatabase)
    .def("getSize", &StimuliProvider::getSize)
    .def("getSizeX", &StimuliProvider::getSizeX)
    .def("getSizeY", &StimuliProvider::getSizeY)
    .def("getSizeD", &StimuliProvider::getSizeD)
    .def("getBatchSize", &StimuliProvider::getBatchSize)
    .def("isCompositeStimuli", &StimuliProvider::isCompositeStimuli)
    .def("getNbChannels", &StimuliProvider::getNbChannels)
    .def("getNbTransformations", &StimuliProvider::getNbTransformations, py::arg("set"))
    .def("getTransformation", &StimuliProvider::getTransformation, py::arg("set"))
    .def("getOnTheFlyTransformation", &StimuliProvider::getOnTheFlyTransformation, py::arg("set"))
    .def("getChannelTransformation", &StimuliProvider::getChannelTransformation, py::arg("channel"), py::arg("set"))
    .def("getChannelOnTheFlyTransformation", &StimuliProvider::getChannelOnTheFlyTransformation, py::arg("channel"), py::arg("set"))
    .def("getBatch", &StimuliProvider::getBatch)
    .def("getData", (StimuliProvider::TensorData_T& (StimuliProvider::*)()) &StimuliProvider::getData)
    .def("getLabelsData", (Tensor<int>& (StimuliProvider::*)()) &StimuliProvider::getLabelsData)
    .def("getLabelsROIs", (const std::vector<std::vector<std::shared_ptr<ROI> > >& (StimuliProvider::*)() const) &StimuliProvider::getLabelsROIs)
    .def("getData", (const StimuliProvider::TensorData_T (StimuliProvider::*)(unsigned int, unsigned int) const) &StimuliProvider::getData, py::arg("channel"), py::arg("batchPos") = 0)
    .def("getLabelsData", (const Tensor<int> (StimuliProvider::*)(unsigned int, unsigned int) const) &StimuliProvider::getLabelsData, py::arg("channel"), py::arg("batchPos") = 0)
    .def("getLabelsROIs", (const std::vector<std::shared_ptr<ROI> >& (StimuliProvider::*)(unsigned int) const) &StimuliProvider::getLabelsROIs, py::arg("batchPos") = 0)
    .def("getCachePath", &StimuliProvider::getCachePath);
}
}
#endif
