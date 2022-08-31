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
#include "Transformation/RangeAffineTransformation.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/GraphViz.hpp"
#include "Adversarial.hpp"

N2D2::StimuliProvider::ProvidedData::ProvidedData(ProvidedData&& other)
    : batch(std::move(other.batch)),
      data(other.data),
      labelsData(other.labelsData),
      targetData(other.targetData),
      labelsROI(std::move(other.labelsROI))
{
}

void N2D2::StimuliProvider::ProvidedData::swap(ProvidedData& other) {
    batch.swap(other.batch);
    data.swap(other.data);
    targetData.swap(other.targetData);
    labelsData.swap(other.labelsData);
    labelsROI.swap(other.labelsROI);
}

N2D2::StimuliProvider::StimuliProvider(Database& database,
                                       const std::vector<size_t>& size,
                                       unsigned int batchSize,
                                       bool compositeStimuli)
    : // Variables
      mDataSignedMapping(this, "DataSignedMapping", false),
      mQuantizationLevels(this, "QuantizationLevels", 0U),
      mQuantizationMin(this, "QuantizationMin", 0.0),
      mQuantizationMax(this, "QuantizationMax", 1.0),
      mStreamTensor(this, "StreamTensor", false),
      mStreamLabel(this, "StreamLabel", false),
      mDatabase(database),
      mSize(size),
      mBatchSize(batchSize),
      mCompositeStimuli(compositeStimuli),
      mCachePath(""),
      mFuture(false)
{
    // ctor
    int count = 1;
#ifdef CUDA
    // Don't use CHECK_CUDA_STATUS because this class should be usable even when
    // N2D2 is compiled with CUDA and there is no device. 
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        count = 1;
#endif

    // mProvidedData is a vector, with one element per device
    mProvidedData.resize(count);
    mFutureProvidedData.resize(count);
    mDevicesInfo.numBatchs.resize(count, -1);
    mDevicesInfo.numFutureBatchs.resize(count, -1);
#ifdef CUDA
    mDevicesInfo.states.resize(count, N2D2::DeviceState::Excluded);
#endif

    // Default construction of the adversarial attack
    // Another attack pointer may be brought by the StimuliProvider generator
    std::shared_ptr<Adversarial> adv(new Adversarial(Adversarial::Attack_T::None));
    setAdversarialAttack(adv);

#ifdef CUDA
    const char* gpuDevices = std::getenv("N2D2_GPU_DEVICES");

    if (gpuDevices != NULL) {
        std::stringstream gpuDevicesStr;
        gpuDevicesStr.str(gpuDevices);

        std::set<int> devices;
        std::copy(std::istream_iterator<int>(gpuDevicesStr),
                  std::istream_iterator<int>(),
                  std::inserter(devices, devices.end()));

        if (devices.lower_bound(0) != devices.begin()) {
            std::stringstream msg;
            msg << "Cannot execute the program with a device number "  
                << "lower than 0 !\n"
                << "Please choose another device number" << std::endl;

            throw std::runtime_error(msg.str());
        }
        if (devices.upper_bound(count-1) != devices.end()) {
            std::stringstream msg;
            msg << "Cannot execute the program with a device number "  
                << "greater than " << count - 1 << " !\n"
                << "Please choose another device number" << std::endl;

            throw std::runtime_error(msg.str());
        }

        int dev;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));

        if (devices.find(dev) == devices.end()){
            std::stringstream msg;
            msg << "Cuda device selected with the dev option "
                << "is not in the N2D2_GPU_DEVICES list: "
                << "[ ";
            std::copy(devices.begin(),
                      devices.end(),
                      std::ostream_iterator<int>(msg, " "));
            msg << "]\n"
                << "Please choose another cuda device" << std::endl;

            throw std::runtime_error(msg.str());
        }

        setDevices(devices);
    }
    else {
#endif
        setDevices();
#ifdef CUDA
    }
#endif
}

N2D2::StimuliProvider::StimuliProvider(StimuliProvider&& other)
    : mDataSignedMapping(this, "DataSignedMapping", other.mDataSignedMapping),
      mQuantizationLevels(this, "QuantizationLevels", other.mQuantizationLevels),
      mQuantizationMin(this, "QuantizationMin", other.mQuantizationMin),
      mQuantizationMax(this, "QuantizationMax", other.mQuantizationMax),
      mStreamTensor(this, "StreamTensor", false),
      mStreamLabel(this, "StreamLabel", false),
      mDatabase(other.mDatabase),
      mSize(std::move(other.mSize)),
      mBatchSize(other.mBatchSize),
      mCompositeStimuli(other.mCompositeStimuli),
      mCachePath(std::move(other.mCachePath)),
      mTransformations(other.mTransformations),
      mChannelsTransformations(std::move(other.mChannelsTransformations)),
      mProvidedData(std::move(other.mProvidedData)),
      mFutureProvidedData(std::move(other.mFutureProvidedData)),
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

void N2D2::StimuliProvider::setDevices(const std::set<int>& devices)
{
    std::vector<size_t> dataSize(mSize);
    dataSize.push_back(mBatchSize);

    std::vector<size_t> labelSize(mSize);

    if (mCompositeStimuli) {
        // Last dimension is channel, mCompositeStimuli assumes unique label
        // for all channels by default
        labelSize.back() = 1;
    }
    else
        std::fill(labelSize.begin(), labelSize.end(), 1U);

    labelSize.push_back(mBatchSize);

    int currentDev = 0;
#ifdef CUDA
    const cudaError_t status = cudaGetDevice(&currentDev);
    if (status != cudaSuccess)
        currentDev = 0;
#endif

    if (devices.empty()) {
        mDevices.clear();
        mDevices.insert(currentDev);
    }
    else
        mDevices = devices;

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            mProvidedData[dev].batch.resize(mBatchSize);
            mProvidedData[dev].data.resize(dataSize);
            mProvidedData[dev].labelsData.resize(labelSize);
            mProvidedData[dev].labelsROI.resize(std::max(mBatchSize, 1u));

            mFutureProvidedData[dev].batch.resize(mBatchSize);
            mFutureProvidedData[dev].data.resize(dataSize);
            mFutureProvidedData[dev].labelsData.resize(labelSize);
            mFutureProvidedData[dev].labelsROI.resize(std::max(mBatchSize, 1u));
#ifdef CUDA
            mDevicesInfo.states[dev] = N2D2::DeviceState::Connected;
#endif
        }
        else {
            mProvidedData[dev].batch.clear();
            mProvidedData[dev].data.clear();
            mProvidedData[dev].labelsData.clear();
            mProvidedData[dev].labelsROI.clear();

            mFutureProvidedData[dev].batch.clear();
            mFutureProvidedData[dev].data.clear();
            mFutureProvidedData[dev].labelsData.clear();
            mFutureProvidedData[dev].labelsROI.clear();
        }
    }

#ifdef CUDA
    if (mDevices.size() > 1) {
        // hostBased() is set to false, which means the data is considered to be
        // originating from the GPU. In this case, there will be no HToD 
        // synchronization on the first layer. We handle it in StimuliProvider, in 
        // the readStimulus() method.
        mProvidedData[currentDev].data.hostBased() = false;
        mProvidedData[currentDev].targetData.hostBased() = false;

        mFutureProvidedData[currentDev].data.hostBased() = false;
        mFutureProvidedData[currentDev].targetData.hostBased() = false;

        std::cout << "Multi-GPU enabled with devices: ";
        std::copy(mDevices.begin(),
                  mDevices.end(),
                  std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }
#endif
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

    int dev = 0;
#ifdef CUDA
    const cudaError_t status = cudaGetDevice(&dev);
    if (status != cudaSuccess)
        dev = 0;
#endif

    std::vector<size_t> labelSize(mProvidedData[dev].labelsData.dims());

    if (!mChannelsTransformations.empty()) {
        labelSize.pop_back();
        ++labelSize.back();
        labelSize.push_back(mBatchSize);
    }

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        mProvidedData[dev].data.resize(dataSize);
        mProvidedData[dev].labelsData.resize(labelSize);

        mFutureProvidedData[dev].data.resize(dataSize);
        mFutureProvidedData[dev].labelsData.resize(labelSize);
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
    {
        mTransformations(*it).cacheable.push_back(transformation);
        mTransformations(*it).cacheable.setStimuliProvider(this);
    }
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
    {
        mTransformations(*it).onTheFly.push_back(transformation);
        mTransformations(*it).onTheFly.setStimuliProvider(this);
    }
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
    {
        mChannelsTransformations.back()(*it)
            .cacheable.push_back(transformation);
        mChannelsTransformations.back()(*it)
            .cacheable.setStimuliProvider(this);
    }
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
    {
        mChannelsTransformations.back()(*it).onTheFly.push_back(transformation);
        mChannelsTransformations.back()(*it).onTheFly.setStimuliProvider(this);
    }
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
    {
        mChannelsTransformations[channel](*it)
            .cacheable.push_back(transformation);
        mChannelsTransformations[channel](*it)
            .cacheable.setStimuliProvider(this);
    }
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
    {
        mChannelsTransformations[channel](*it)
            .onTheFly.push_back(transformation);
        mChannelsTransformations[channel](*it)
            .onTheFly.setStimuliProvider(this);
    }
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
             ++itTrans)
        {
            (*itTrans)(*it).cacheable.push_back(transformation);
            (*itTrans)(*it).cacheable.setStimuliProvider(this);
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
             ++itTrans)
        {
            (*itTrans)(*it).onTheFly.push_back(transformation);
            (*itTrans)(*it).onTheFly.setStimuliProvider(this);
        }
    }
}

void N2D2::StimuliProvider::iterTransformations(
    Database::StimuliSet set,
    std::function<void(const Transformation&)> func) const
{
    mTransformations(set).cacheable.iterTransformations(func);
    mTransformations(set).onTheFly.iterTransformations(func);

    // Channel transformations
    for (size_t channel = 0; channel < mChannelsTransformations.size();
        ++channel)
    {
        mChannelsTransformations[channel](set).cacheable
            .iterTransformations(func);
        mChannelsTransformations[channel](set).onTheFly
            .iterTransformations(func);
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

bool N2D2::StimuliProvider::normalizeIntegersStimuli(int envCvDepth) {
    bool normalizationOccured = false;

    for(Database::StimuliSet stimuliSet: mDatabase.getStimuliSets(Database::All)) {
        int depthSet = envCvDepth;
        depthSet = getTransformation(stimuliSet).getOutputsDepth(depthSet);
        depthSet = getOnTheFlyTransformation(stimuliSet).getOutputsDepth(depthSet);


        if(mChannelsTransformations.empty()) {
            const std::pair<double, double> depthUnityValue
                = Utils::cvMatDepthUnityValue(depthSet, mDataSignedMapping);
            if(depthUnityValue.second != 1.0) {
                normalizationOccured = true;

                addOnTheFlyTransformation(
                    RangeAffineTransformation(RangeAffineTransformation::Plus, depthUnityValue.first,
                                              RangeAffineTransformation::Divides, depthUnityValue.second), 
                    mDatabase.getStimuliSetMask(stimuliSet)
                );
            }
        }
        else {
            for(std::size_t ch = 0; ch < mChannelsTransformations.size(); ch++) {
                int chDepthSet = depthSet;
                chDepthSet = mChannelsTransformations[ch](stimuliSet).cacheable.getOutputsDepth(chDepthSet);
                chDepthSet = mChannelsTransformations[ch](stimuliSet).onTheFly.getOutputsDepth(chDepthSet);


                const std::pair<double, double> depthUnityValue
                    = Utils::cvMatDepthUnityValue(chDepthSet, mDataSignedMapping);
                if(depthUnityValue.second != 1.0) {
                    normalizationOccured = true;

                    addChannelOnTheFlyTransformation(
                        ch,
                        RangeAffineTransformation(RangeAffineTransformation::Plus, depthUnityValue.first,
                                                RangeAffineTransformation::Divides, depthUnityValue.second), 
                        mDatabase.getStimuliSetMask(stimuliSet)
                    );
                }
            }
        }
    }

    return normalizationOccured;
}

void N2D2::StimuliProvider::logTransformations(
    const std::string& fileName,
    Database::StimuliSetMask setMask) const
{
    GraphViz graph("Transformations", "", true);

    const std::vector<Database::StimuliSet> stimuliSets
        = mDatabase.getStimuliSets(setMask);

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

            if (trans->getType() == std::string("RangeAffine")) {
                std::shared_ptr<RangeAffineTransformation> rangeAffineTrans
                    = std::dynamic_pointer_cast<RangeAffineTransformation>(trans);

                labelStr << "\n" << rangeAffineTrans->getFirstOperator() << ": "
                    << rangeAffineTrans->getFirstValue();
                labelStr << "\n" << rangeAffineTrans->getSecondOperator() << ": "
                    << rangeAffineTrans->getSecondValue();
            }

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
/*
void N2D2::StimuliProvider::readEpochBatch( Database::StimuliSet set,
                                            unsigned int startIndex,
                                            unsigned int epochIndex)
{
    const std::vector<std::vector<unsigned int > >& indexes =
                (set == Database::StimuliSet::Learn) ? mDatabaseLearnIndexes :
                (set == Database::StimuliSet::Validation) ? mDatabaseValIndexes :
                mDatabaseTestIndexes;

    if (startIndex >= mDatabase.getNbStimuli(set)) {
        std::stringstream msg;
        msg << "StimuliProvider::readEpochBatch(): startIndex (" << startIndex
            << ") is higher than the number of stimuli in the " << set
            << " set (" << mDatabase.getNbStimuli(set) << ")";

        throw std::runtime_error(msg.str());
    }
    if (epochIndex >= indexes.size()) {
        std::stringstream msg;
        msg << "StimuliProvider::readEpochBatch(): epochIndex (" << epochIndex
            << ") is higher than the number of intialized epoch in the " << set
            << " set (" << mDatabase.getNbStimuli(set) << ")";

        throw std::runtime_error(msg.str());
     }
 
    const unsigned int batchSize
        = std::min(mBatchSize, mDatabase.getNbStimuli(set) - startIndex);
    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
    {
        batchRef[batchPos]
            = mDatabase.getStimulusID(set, indexes[epochIndex][startIndex + batchPos]);
    }
#pragma omp parallel for schedule(dynamic) if (batchSize > 1)
    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos)
        readStimulus(batchRef[batchPos], set, batchPos);

    std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
}
*/

#ifdef CUDA
void N2D2::StimuliProvider::adjustBatchs(Database::StimuliSet set)
{
    std::deque<unsigned int>& indexes = 
                (set == Database::StimuliSet::Learn) ? mIndexesLearn :
                (set == Database::StimuliSet::Validation) ? mIndexesVal :
                mIndexesTest;
    
    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            if (mDevicesInfo.states[dev] == N2D2::DeviceState::Banned) {

                if (mDevicesInfo.numBatchs[dev] != -1) {
                    indexes.push_front(mDevicesInfo.numBatchs[dev]);
                    mDevicesInfo.numBatchs[dev] = -1;
                }
                if (mDevicesInfo.numFutureBatchs[dev] != -1) {
                    indexes.push_front(mDevicesInfo.numFutureBatchs[dev]);
                    mDevicesInfo.numFutureBatchs[dev] = -1;
                }
            }
        }
    }
}
#endif

void N2D2::StimuliProvider::future()
{
    mFuture = true;
}

void N2D2::StimuliProvider::synchronize()
{
    if (mFuture) {
        // Don't swap the vectors directly, as it would invalidate the
        // address to the tensors
        for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
            mProvidedData[dev].swap(mFutureProvidedData[dev]);
            mDevicesInfo.numBatchs[dev] = mDevicesInfo.numFutureBatchs[dev];
            mDevicesInfo.numFutureBatchs[dev] = -1;
        }

        mFuture = false;
    }

    synchronizeToDevices();
}
// unsigned int
//     N2D2::StimuliProvider::setStimuliIndexes(   Database::StimuliSet set,   
//                                                 const unsigned int nbEpochs,
//                                                 bool randPermutation)
// {
//     if(nbEpochs == 0) {
//         std::stringstream msg;
//         msg << "StimuliProvider::DatabasePermutation(): nbEpochs (" << nbEpochs
//             << ") must be higher than 0 ! for set (" << set << ")";

//         throw std::runtime_error(msg.str());
//     }
//     const unsigned int nMax = mDatabase.getNbStimuli(set);
//     std::vector<std::vector<unsigned int > >& indexes =
//             (set == Database::StimuliSet::Learn) ? mDatabaseLearnIndexes :
//             (set == Database::StimuliSet::Validation) ? mDatabaseValIndexes :
//                 mDatabaseTestIndexes;

//     if(nMax == 0)
//     {
//         std::cout << Utils::cwarning << "setStimuliIndexes for set " << set <<
//             " is empty" << Utils::cdef
//             << std::endl;
//         return 0U;
//     }
//     if(indexes.empty())
//     {
//         indexes.resize(nbEpochs, 
//                                 std::vector<unsigned int>(nMax));
//         for(unsigned int epoch = 0; epoch < nbEpochs; ++ epoch)
//         {
//             std::iota(  std::begin(indexes[epoch]),
//                         std::end(indexes[epoch]),
//                         0U);

//             //Sort index of data stimuli under a pseudo random range
//             if(randPermutation) {
//                 for(unsigned int n = 0; n < (nMax - 1); ++n)
//                 {
//                     const unsigned int randIdx = 
//                             Random::mtRand() % (nMax - n);
//                     const unsigned int tmp = indexes[epoch][n];
//                     indexes[epoch][n] = indexes[epoch][(randIdx+n)];
//                     indexes[epoch][(randIdx+n)] = tmp;
//                 }

//             }
//         }
//     }
//     else {
//         std::cout << Utils::cwarning << "indexes for set " << set <<
//             " are already initialized" << Utils::cdef
//             << std::endl;
//     }
//     return nMax;
// } 
unsigned int N2D2::StimuliProvider::getRandomIndex(Database::StimuliSet set)
{
    return Random::randUniform(0, mDatabase.getNbStimuli(set) - 1);
}

N2D2::Database::StimulusID
N2D2::StimuliProvider::getRandomID(Database::StimuliSet set)
{
    return mDatabase.getStimulusID(set, getRandomIndex(set));
}

N2D2::Database::StimulusID
N2D2::StimuliProvider::getRandomIDWithLabel(Database::StimuliSet set, int label)
{
    std::vector<Database::StimulusID> partitionWithLabel;

    for (unsigned int index = 0; index < mDatabase.getNbStimuli(set); ++index) {
        if (mDatabase.getStimulusLabel(set, index) == label
            || mDatabase.getNbROIsWithLabel(label) > 0)
        {
            partitionWithLabel.push_back(mDatabase.getStimulusID(set, index));
        }
    }

    const unsigned int randomIdx
        = Random::randUniform(0, partitionWithLabel.size() - 1);
    return partitionWithLabel[randomIdx];
}

void N2D2::StimuliProvider::readRandomBatch(Database::StimuliSet set)
{
    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            std::vector<int>& batchRef = (mFuture)
                ? mFutureProvidedData[dev].batch
                : mProvidedData[dev].batch;

            for (unsigned int batchPos = 0; batchPos < mBatchSize; ++batchPos)
                batchRef[batchPos] = getRandomID(set);
        }
    }

    unsigned int exceptCatch = 0;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for schedule(dynamic) collapse(2) if (mProvidedData.size() > 1 || mBatchSize > 1)
#else
#pragma omp parallel for schedule(dynamic) if (mProvidedData.size() > 1 || mBatchSize > 1)
#endif
    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
            if (mDevices.find(dev) != mDevices.end()) {
                std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;

                try {
                    readStimulus(batchRef[batchPos], set, batchPos, dev);
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
        }
    }

    if (exceptCatch > 0) {
        std::cout << "Retry without multi-threading..." << std::endl;

        for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
            if (mDevices.find(dev) != mDevices.end()) {
                std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;

                for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
                    readStimulus(batchRef[batchPos], set, batchPos, dev);
                }
            }
        }
    }
}

N2D2::Database::StimulusID
N2D2::StimuliProvider::readRandomStimulus(Database::StimuliSet set,
                                          unsigned int batchPos,
                                          int dev)
{
    const Database::StimulusID id = getRandomID(set);
    readStimulus(id, set, batchPos, dev);
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

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            std::vector<int>& batchRef = (mFuture)
                ? mFutureProvidedData[dev].batch
                : mProvidedData[dev].batch;
            const unsigned int batchSize
                = std::min(mBatchSize, mDatabase.getNbStimuli(set) - startIndex);

            for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
                batchRef[batchPos]
                    = mDatabase.getStimulusID(set, startIndex + batchPos);
            }

            std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
            startIndex += batchSize;
        }
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for schedule(dynamic) collapse(2) if (mProvidedData.size() > 1 || mBatchSize > 1)
#else
#pragma omp parallel for schedule(dynamic) if (mProvidedData.size() > 1 || mBatchSize > 1)
#endif
    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
            if (mDevices.find(dev) != mDevices.end()) {
                std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;

                if (batchRef[batchPos] >= 0) {
                    readStimulus(batchRef[batchPos], set, batchPos, dev);
                }
            }
        }
    }
}

void N2D2::StimuliProvider::streamBatch(int startIndex, int dev) {
    std::vector<int>& batchRef = mProvidedData[getDevice(dev)].batch;

    if (startIndex < 0)
        startIndex = batchRef.back() + 1;

    std::iota(batchRef.begin(), batchRef.end(), startIndex);
}

void N2D2::StimuliProvider::readStimulusBatch(Database::StimulusID id,
                                              Database::StimuliSet set,
                                              int dev)
{
    std::vector<int>& batchRef = (mFuture)
        ? mFutureProvidedData[getDevice(dev)].batch
        : mProvidedData[getDevice(dev)].batch;

    readStimulus(id, set, 0, dev);

    batchRef[0] = id;
    std::fill(batchRef.begin() + 1, batchRef.end(), -1);
}

void N2D2::StimuliProvider::readStimulus(Database::StimulusID id,
                                         Database::StimuliSet set,
                                         unsigned int batchPos,
                                         int dev)
{
    std::stringstream dataCacheFile, labelsCacheFile, validCacheFile;
    dataCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                  << id << "_data_" << set << ".bin";
    labelsCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                    << id << "_labels_" << set << ".bin";
    validCacheFile << mCachePath << "/" << std::setfill('0') << std::setw(7)
                    << id << "_" << set << ".valid";

    dev = getDevice(dev);
#ifdef CUDA

    // Necessary to save the current device id before calling
    // cudaSetDevice(dev) in order to avoid Multi-GPU malfunction
    int currentDev = 0;
    const cudaError_t status = cudaGetDevice(&currentDev);
    if (status != cudaSuccess)
        currentDev = 0;

    // readStimulus() is typically  called in an OpenMP thread.
    // The current CUDA device therefore will not necessarily match dev.
    // However, some transformations may need the correct device, such as
    // BlendingTransformation.
    cudaSetDevice(dev);
#endif

    std::vector<std::shared_ptr<ROI> >& labelsROI = (mFuture)
        ? mFutureProvidedData[dev].labelsROI[batchPos]
        : mProvidedData[dev].labelsROI[batchPos];
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
            .clone(), mDataSignedMapping)  // make sure the database image will not be altered
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

    TensorData_T& dataRef = (mFuture)
        ? mFutureProvidedData[dev].data
        : mProvidedData[dev].data;
    Tensor<int>& labelsRef = (mFuture)
        ? mFutureProvidedData[dev].labelsData
        : mProvidedData[dev].labelsData;
    TensorData_T& targetDataRef = (mFuture)
        ? mFutureProvidedData[dev].targetData
        : mProvidedData[dev].targetData;

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

        if (!targetDataRef.empty()) {
            targetDataRef.clear();
            targetDataRef.push_back(targetData);
        }
    }
#ifdef CUDA
    cudaSetDevice(currentDev);
#endif

}

N2D2::Database::StimulusID N2D2::StimuliProvider::readStimulusBatch(
    Database::StimuliSet set, unsigned int index, int dev)
{
    std::vector<int>& batchRef = (mFuture)
        ? mFutureProvidedData[getDevice(dev)].batch
        : mProvidedData[getDevice(dev)].batch;

    const Database::StimulusID id = readStimulus(set, index, 0, dev);

    batchRef[0] = id;
    std::fill(batchRef.begin() + 1, batchRef.end(), -1);

    return id;
}

N2D2::Database::StimulusID N2D2::StimuliProvider::readStimulus(
    Database::StimuliSet set, unsigned int index, unsigned int batchPos,
    int dev)
{
    const Database::StimulusID id = mDatabase.getStimulusID(set, index);
    readStimulus(id, set, batchPos, dev);
    return id;
}

#ifdef CUDA
/** Determine if a device is able to read a batch. 
 * 
 * Determine if a device is able to read a batch 
 * according to its state and the mode (future or not)
 * 
 * @param set       StimuliSet
 * @param future    Boolean to know if future 
 * @param state     State of a device
 */
bool isReadPossible(N2D2::Database::StimuliSet set,
                    bool future, 
                    N2D2::DeviceState state) 
{
    bool isAuthorized = false;

    if (state == N2D2::DeviceState::Connected)
        isAuthorized = true;
    else {
        if (set == N2D2::Database::StimuliSet::Learn) {
            if (future) {
                if (state == N2D2::DeviceState::Debanned)
                    isAuthorized = true;
            }
        }
    }

    return isAuthorized;
}
#endif

void N2D2::StimuliProvider::readBatch(Database::StimuliSet set)
{
    std::vector<unsigned int>& batchs =
                (set == Database::StimuliSet::Learn) ? mBatchsLearnIndexes :
                (set == Database::StimuliSet::Validation) ? mBatchsValIndexes :
                mBatchsTestIndexes;
    
    std::deque<unsigned int>& indexes = 
                (set == Database::StimuliSet::Learn) ? mIndexesLearn :
                (set == Database::StimuliSet::Validation) ? mIndexesVal :
                mIndexesTest;
    
    if (batchs.size() == 0) {
        std::stringstream msg;
        msg << "indexes for set " << set <<
            " must be initialized first";

        throw std::runtime_error(msg.str());
    }

    if (indexes.empty()) {
        return;
    }

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            int index = -1;
            bool readAllowed = true;
#ifdef CUDA
            // Test if the device is banned
            readAllowed = isReadPossible(set, mFuture, mDevicesInfo.states[dev]);
#endif
            if (readAllowed) {
                if (!indexes.empty()) {
                    index = indexes.front();
                    indexes.pop_front();
                } else {
                    if (set == Database::StimuliSet::Learn) {
                        unsigned int nbBatch 
                            = std::ceil(mDatabase.getNbStimuli(set)/ (double)mBatchSize);
                        index = Random::randUniform(0, nbBatch - 1) * mBatchSize;
                    }
                }
            }
            if (mFuture) mDevicesInfo.numFutureBatchs[dev] = index;
            else mDevicesInfo.numBatchs[dev] = index;

            std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;

            if (index >= 0) {
                unsigned int batchSize = mBatchSize;

                if (set != Database::StimuliSet::Learn)
                    batchSize = std::min(mBatchSize, mDatabase.getNbStimuli(set) - index);

                for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
                    batchRef[batchPos] = mDatabase.getStimulusID(set, batchs[index + batchPos]);
                }
                std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
            } else
                std::fill(batchRef.begin(), batchRef.end(), -1);   
        }
    }

    unsigned int exceptCatch = 0;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for schedule(dynamic) collapse(2) if (mProvidedData.size() > 1 || mBatchSize > 1)
#else
#pragma omp parallel for schedule(dynamic) if (mProvidedData.size() > 1 || mBatchSize > 1)
#endif
    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
            if (mDevices.find(dev) != mDevices.end()) {
                std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;

				if (batchRef[batchPos] >= 0) {
                    try {
                        readStimulus(batchRef[batchPos], set, batchPos, dev);
                    }
                    catch (const std::exception& e)
                    {
                        #pragma omp critical(StimuliProvider__readBatch)
                        {
                            std::cout << Utils::cwarning << e.what() << Utils::cdef
                                << std::endl;
                            ++exceptCatch;
                        }
                    }
                }
            }
        }
    }

    if (exceptCatch > 0) {
        std::cout << "Retry without multi-threading..." << std::endl;

        for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
            if (mDevices.find(dev) != mDevices.end()) {
                std::vector<int>& batchRef = (mFuture)
                    ? mFutureProvidedData[dev].batch
                    : mProvidedData[dev].batch;   

                for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
                    if (batchRef[batchPos] >= 0) 
                        readStimulus(batchRef[batchPos], set, batchPos, dev);
                }
            }
        }
    }

}

void N2D2::StimuliProvider::streamStimulus(const cv::Mat& mat,
                                           Database::StimuliSet set,
                                           unsigned int batchPos,
                                           int dev)
{
    TensorData_T& dataRef = (mFuture)
        ? mFutureProvidedData[getDevice(dev)].data
        : mProvidedData[getDevice(dev)].data;

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


void N2D2::StimuliProvider::setStreamedTensor(TensorData_T& streamedTensor) 
{
    if (mStreamTensor){
        mStreamedTensor = &streamedTensor;
    }else{
        throw std::runtime_error("Error: StreamTensor is False but you try to set a StreamedTensor.");
    }
}
void N2D2::StimuliProvider::setStreamedLabel(Tensor<int>& streamedLabel) 
{
    if (mStreamLabel){
        mStreamedLabel = &streamedLabel;
    }else{
        throw std::runtime_error("Error: StreamLabel is False but you try to set a StreamedLabel.");
    }
}

void N2D2::StimuliProvider::synchronizeToDevices() {
#ifdef CUDA
    int currentDev = 0;
    const cudaError_t status = cudaGetDevice(&currentDev);
    if (status != cudaSuccess)
        currentDev = 0;

    bool setDeviceBack = false;

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        if (mDevices.find(dev) != mDevices.end()) {
            TensorData_T& dataRef = mProvidedData[dev].data;
            TensorData_T& targetDataRef = mProvidedData[dev].targetData;

            if (!mProvidedData[currentDev].data.hostBased()) {
                // If hostBased() is false, multi-GPU is enabled.
                // All the GPU data must be referenced in the mProvidedData[currentDev]
                // element, which is the input of the first layer.
                // The other elements are never synchronized on GPU, they stay on CPU.
                CHECK_CUDA_STATUS(cudaSetDevice(dev));
                setDeviceBack = true;
                mProvidedData[currentDev].data.synchronizeToD(dataRef);

                if (!targetDataRef.empty())
                    mProvidedData[currentDev].targetData.synchronizeToD(targetDataRef);
            }
        }
    }

    if (setDeviceBack)
        CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
#endif
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
        int dev = 0;
#ifdef CUDA
        const cudaError_t status = cudaGetDevice(&dev);
        if (status != cudaSuccess)
            dev = 0;
#endif

        std::vector<size_t> dataSize(mProvidedData[dev].data.dims());
        dataSize.back() = mBatchSize;

        std::vector<size_t> labelSize(mProvidedData[dev].labelsData.dims());
        labelSize.back() = mBatchSize;

        for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
            mProvidedData[dev].data.resize(dataSize);
            mProvidedData[dev].labelsData.resize(labelSize);

            mFutureProvidedData[dev].data.resize(dataSize);
            mFutureProvidedData[dev].labelsData.resize(labelSize);
        }
    }
}

void N2D2::StimuliProvider::setTargetSize(const std::vector<size_t>& size) {
    mTargetSize = size;

    std::vector<size_t> targetSize(size);
    targetSize.push_back(mBatchSize);

    for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
        mProvidedData[dev].targetData.resize(targetSize);
        mFutureProvidedData[dev].targetData.resize(targetSize);
    }
}

void N2D2::StimuliProvider::setBatch(Database::StimuliSet set,
                                      bool randShuffle, 
                                      unsigned int nbMax)
{
    const unsigned int nbStimuli = nbMax > 0 
                                   ? std::min(nbMax, mDatabase.getNbStimuli(set))
                                   : mDatabase.getNbStimuli(set);
    const unsigned int batchSize = getBatchSize();
    const unsigned int nbBatchs = std::ceil(nbStimuli / (double)batchSize);

    if(nbStimuli == 0) {
        std::stringstream msg;
        msg << "setStimuliIndexes for set " << set 
            << " is empty" << std::endl;

        throw std::runtime_error(msg.str());
    }
	std::vector<unsigned int>& batchs =
                (set == Database::StimuliSet::Learn) ? mBatchsLearnIndexes :
                (set == Database::StimuliSet::Validation) ? mBatchsValIndexes :
                mBatchsTestIndexes;
    batchs.clear();
    batchs.resize(nbStimuli);
    std::iota(batchs.begin(),
              batchs.end(),
              0U);
              
    if (set == Database::StimuliSet::Learn) {
        // The last batch might be shorter than the others
        // The following loop is to complete that batch
        unsigned int ind = 0;
        while (batchs.size() < batchSize * nbBatchs) {
            unsigned int index = randShuffle
                                 ? getRandomIndex(set)
                                 : ind++;
            batchs.push_back(index);
        }
    }
    //Sort index of data stimuli under a pseudo random range
    if (randShuffle) {
        std::random_shuffle(batchs.begin(),
                            batchs.end(),
                            Random::randShuffle);
    }
    
    std::deque<unsigned int>& indexes = 
                (set == Database::StimuliSet::Learn) ? mIndexesLearn :
                (set == Database::StimuliSet::Validation) ? mIndexesVal :
                mIndexesTest;
	indexes.clear();
    indexes.resize(nbBatchs);
    std::iota(indexes.begin(),
              indexes.end(),
              0U);
	for (auto& x: indexes)
        x *= batchSize;
    
    if (randShuffle) {
        std::random_shuffle(indexes.begin(),
                            indexes.end(),
                            Random::randShuffle);
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
N2D2::StimuliProvider::getDataChannel(unsigned int channel,
                                      unsigned int batchPos,
                                      int dev) const
{
    return TensorData_T(mProvidedData[getDevice(dev)]
        .data[batchPos][channel]);
}

const N2D2::Tensor<int>
N2D2::StimuliProvider::getLabelsDataChannel(unsigned int channel,
                                            unsigned int batchPos,
                                            int dev) const
{
    return Tensor<int>(mProvidedData[getDevice(dev)]
        .labelsData[batchPos][channel]);
}

const N2D2::StimuliProvider::TensorData_T
N2D2::StimuliProvider::getTargetDataChannel(unsigned int channel,
                                            unsigned int batchPos,
                                            int dev) const
{
    return TensorData_T((!mProvidedData[getDevice(dev)].targetData.empty())
        ? mProvidedData[getDevice(dev)].targetData[batchPos][channel]
        : mProvidedData[getDevice(dev)].data[batchPos][channel]);
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
    unsigned int dimSqrt = 0;

    for (unsigned int z = 0; z < data.dimZ(); ++z) {
        const Tensor<Float_T>& channel = data[z];

        if (dimSqrt == 0 && dimX > 1 && dimY > 1) {
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
            dimSqrt = std::ceil(std::sqrt((double)size));
            unsigned int index = 0;

            for (unsigned int y = 0; y < dimSqrt; ++y) {
                for (unsigned int x = 0; x < dimSqrt; ++x) {
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

    if (dimSqrt > 0) {
        dimX = dimSqrt;
        dimY = dimSqrt;
    }

    Gnuplot gnuplot(fileName + ".gnu");
    gnuplot.set("grid").set("key off");
    gnuplot.set("size ratio 1");
    gnuplot.setXrange(-0.5, dimX - 0.5);
    gnuplot.setYrange(dimY - 0.5, -0.5, "reverse");

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
    gnuplot.setYrange(dimY - 0.5, -0.5, "reverse");

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
