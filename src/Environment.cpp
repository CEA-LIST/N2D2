/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#include "Environment.hpp"
#include "NodeEnv.hpp"
#include "Transformation/FilterTransformation.hpp"

namespace N2D2 {
Database EmptyDatabase;
}

N2D2::Environment::Environment(Network& network,
                               Database& database,
                               const std::vector<size_t>& size,
                               unsigned int batchSize,
                               bool compositeStimuli)
    : StimuliProvider(database, size, batchSize, compositeStimuli),
      mNetwork(network),
      mNodes(mData.dims(), (NodeEnv*)NULL)

{
    // ctor
    // Create default nodes, if the Environment has no transformation
    fillNodes(mNodes);
}

void N2D2::Environment::fillNodes(Tensor<NodeEnv*> nodes, double orientation)
{
    if (nodes.nbDims() > 2) {
        for (size_t i = 0; i < nodes.dims().back(); ++i)
            fillNodes(nodes[i]);
    }
    else {
        const size_t dimY = (nodes.nbDims() > 1) ? nodes.dimY() : 1;

        for (size_t y = 0; y < dimY; ++y) {
            for (size_t x = 0; x < nodes.dimX(); ++x) {
                if (nodes(x, y) == NULL)
                    nodes(x, y) = new NodeEnv(mNetwork, 1.0, orientation, x, y);
            }
        }
    }
}

void N2D2::Environment::addChannel(const CompositeTransformation
                                   & transformation)
{
    // DEPRECATED: unsafe legacy adpatation -->
    const std::shared_ptr<FilterTransformation> filter
        = std::dynamic_pointer_cast<FilterTransformation>(transformation[0]);
    const double orientation = (filter != NULL) ? filter->getOrientation()
                                                : 0.0;
    // <--

    std::vector<size_t> dataSize(mNodes.dims());
    dataSize.pop_back();

    if (!mChannelsTransformations.empty())
        ++dataSize.back();
    else
        dataSize.back() = 1;

    dataSize.push_back(mBatchSize);

    if (!mChannelsTransformations.empty()) {
        Tensor<NodeEnv*> nodes(dataSize, (NodeEnv*)NULL);

        for (unsigned int b = 0; b < nodes.dimB(); ++b) {
            for (unsigned int z = 0; z < nodes.dimZ(); ++z) {
                if (z < mNodes.dimZ())
                    nodes[b][z] = mNodes[b][z];
            }
        }

        mNodes.swap(nodes);
    }
    else {
        // Delete the default nodes created in the constructor, which may not
        // have the right orientation anymore
        std::for_each(mNodes.begin(), mNodes.end(), Utils::Delete());
        mNodes.resize(dataSize, NULL);
    }

    fillNodes(mNodes, orientation);

    StimuliProvider::addChannel(transformation);

    assert(mNodes.nbDims() == mData.nbDims());
    assert(std::equal(mNodes.dims().begin(), mNodes.dims().end(),
                      mData.dims().begin()));
    assert(mNodes.size() == mData.size());
}

void N2D2::Environment::propagate(Time_T start, Time_T end)
{
    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);
    if (aerDatabase) {

        for (std::vector<AerReadEvent>::const_iterator it = mAerData.begin(),
                                                itEnd = mAerData.end(); 
                                                it != itEnd; ++it) {
            unsigned idx = (*it).x + (*it).y * mNodes.dimX() 
                + (*it).channel * mNodes.dimX() * mNodes.dimY()
                + (*it).batch * mNodes.dimX() * mNodes.dimY() * mNodes.dimZ();
            if ((*it).time > end) {
                std::cout << "Warning: event omitted at time: " << (*it).time
                << " since end at: " << end << std::endl;
            }
            mNodes(idx)->incomingSpike(NULL, (*it).time + start, (*it).value);
        }
    }
    else {
        assert(mNodes.size() == mData.size());

        SpikeGenerator::checkParameters();

        for (Tensor<NodeEnv*>::const_iterator it = mNodes.begin(),
                                                itBegin = mNodes.begin(),
                                                itEnd = mNodes.end();
            it != itEnd;
            ++it) {
                std::pair<Time_T, int> event = std::make_pair(start, 0);

                do {
                    SpikeGenerator::nextEvent(event, mData(it - itBegin), start, end);

                    if (event.second != 0)
                        (*it)->incomingSpike(
                            NULL, event.first, (event.second < 0) ? 1 : 0);
                } while (event.second != 0);
        }
    }
}

const std::shared_ptr<N2D2::FilterTransformation>
N2D2::Environment::getFilter(unsigned int channel) const
{
    if (!mChannelsTransformations.empty())
        return std::dynamic_pointer_cast<FilterTransformation>(
            mChannelsTransformations.at(channel)(Database::Learn).cacheable[0]);
    else
        return std::shared_ptr<FilterTransformation>();
}

void
N2D2::Environment::testFrame(unsigned int channel, Time_T start, Time_T end)
{
    const unsigned int size = mNodes.dimX() * mNodes.dimY();
    const Time_T dt = (end - start) / size;

    for (unsigned int batchPos = 0; batchPos < mBatchSize; ++batchPos) {
        const Tensor<NodeEnv*> channelNodes = mNodes[batchPos][channel];

        for (unsigned int i = 0; i < size; ++i)
            channelNodes(i)->incomingSpike(NULL, start + i * dt);
    }
}



N2D2::Database::StimulusID N2D2::Environment::readRandomAerStimulus(
                                            Database::StimuliSet set,
                                            unsigned int batch)
{
    const Database::StimulusID id = getRandomID(set);
    readAerStimulus(id, set, batch);
    return id;
}

void N2D2::Environment::readAerStimulus(Database::StimulusID id,
                                        Database::StimuliSet set,
                                        unsigned int batch)
{
    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);

    if (aerDatabase) {
        // TODO: reuse memory
        mAerData.clear();
        aerDatabase->loadAerStimulusData(mAerData, set, id, batch);
    }
    else {
        throw std::runtime_error("Environment::readAerStimulus: No AER database!");
    }
}



void N2D2::Environment::readRandomAerBatch(Database::StimuliSet set)
{
    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    for (unsigned int batchPos = 0; batchPos < mBatchSize; ++batchPos)
        batchRef[batchPos] = getRandomID(set);

    unsigned int exceptCatch = 0;

#pragma omp parallel for schedule(dynamic) if (mBatchSize > 1)
    for (int batchPos = 0; batchPos < (int)mBatchSize; ++batchPos) {
        try {
            readAerStimulus(batchRef[batchPos], set, batchPos);
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
            readAerStimulus(batchRef[batchPos], set, batchPos);
    }
}


void N2D2::Environment::readAerBatch(Database::StimuliSet set,
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
        readAerStimulus(batchRef[batchPos], set, batchPos);

    std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
}

N2D2::Environment::~Environment()
{
    // dtor
    std::for_each(mNodes.begin(), mNodes.end(), Utils::Delete());
}

/*
void N2D2::Environment::readRandom(Time_T start, Time_T end) {
    std::vector<unsigned int> stimuli;
    const Time_T dt = (end - start);
    const std::vector<NodeEnv*>& nodes = mNodes[0].second[0].second;

    stimuli.reserve(nodes.size());

    for (unsigned int i = 0, size = nodes.size(); i < size; ++i)
        stimuli.push_back(i);

    for (std::vector<NodeEnv*>::const_iterator it = nodes.begin(), itEnd =
nodes.end(); it != itEnd; ++it) {
        unsigned int j = Random::randUniform(0, stimuli.size() - 1);
        (*it)->incomingSpike(NULL, (Time_T) (start + stimuli[j]*dt));
        stimuli.erase(stimuli.begin() + j);
    }
}

void N2D2::Environment::exportStimuli(const std::string& dirName, StimulusSet
set) const {
    const std::string subDir = dirName + ((set == Learn)        ? "/learning" :
                                          (set == Validation)   ? "/validation"
:
                                                                  "/test");

    const unsigned int size = stimuliSet(set).size();
    const unsigned int zeroPad = std::ceil(std::log10(size));

    std::vector<int> params;
    params.push_back(CV_IMWRITE_PXM_BINARY);
    params.push_back(0);

    for (unsigned int index = 0; index < size; ++index) {
        const std::vector<float> stimuli = (!mStimuliCachePath.empty())
            ? loadFrameCache(stimuliSet(set).at(index).first)
            : stimuliSetCache(set).at(index);
        const unsigned int cls = stimuliSet(set).at(index).second;

        std::vector<float>::const_iterator itStimu = stimuli.begin(), itStimuEnd
= stimuli.end();

        std::stringstream baseName;
        baseName << subDir << "/" << getClassName(cls);

        Utils::createDirectories(baseName.str());

        baseName << "/stimuli" << std::setfill('0') << std::setw(zeroPad) <<
index;

        // For each scale ...
        for (unsigned int map = 0, mapSize = mNodes.size(); map < mapSize;
++map) {
            const double scale = mNodes[map].first;

            // ... for each filter
            for (unsigned int filter = 0, filterSize =
mNodes[map].second.size(); filter < filterSize; ++filter) {
                const Matrix<float> stimuliFilter = extractNextFilter(itStimu,
scale, 0.5, 0.5, false, false);
                const cv::Mat img = (cv::Mat) stimuliFilter;

                std::stringstream stimuliName(baseName.str());

                if (mapSize > 1)
                    stimuliName << "_" << map;

                if (filterSize > 1)
                    stimuliName << "_" << filter;

                // Image export
                cv::Mat img8U;
                img.convertTo(img8U, CV_8U, 255.0);
                cv::imwrite(stimuliName.str() + ".pgm", img8U);
                cv::imwrite(stimuliName.str() + ".ascii.pgm", img8U, params);

                // ASCII export
                std::ofstream data((stimuliName.str() + ".dat").c_str());

                if (!data.good())
                    throw std::runtime_error("Could not export stimuli file: " +
stimuliName.str() + ".dat");

                data << stimuliFilter;
            }
        }
    }
}

cv::Mat N2D2::Environment::reconstructFrame(unsigned int index, StimulusSet set)
const {
    if (mDiscardedLateStimuli < 0.0 || mDiscardedLateStimuli > 1.0)
        throw std::domain_error("Environment: DiscardedLateStimuli is out of
range (must be >= 0.0 and <= 1.0)");

    // Load frame
    double firstOrientation = 0.0;
    bool singleOrientation = true;

    for (unsigned int map = 0, mapSize = mNodes.size(); map < mapSize; ++map) {
        // ... for each filter
        for (unsigned int filter = 0, filterSize = mNodes[map].second.size();
filter < filterSize; ++filter) {
            const double orientation =
mNodes[map].second[filter].first.getOrientation();

            if (map == 0 && filter == 0)
                firstOrientation = orientation;
            else if (orientation != firstOrientation)
                singleOrientation = false;
        }
    }

    cv::Mat img(cv::Size(mX, mY), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));

    const std::vector<float> stimuli = (!mStimuliCachePath.empty())
        ? loadFrameCache(stimuliSet(set).at(index).first)
        : stimuliSetCache(set).at(index);

    std::vector<float>::const_iterator itStimu = stimuli.begin(), itStimuEnd =
stimuli.end();

    // Generate frame image
    // For each scale ...
    for (unsigned int map = 0, mapSize = mNodes.size(); map < mapSize; ++map) {
        const double scale = mNodes[map].first;
        const unsigned int width = (unsigned int) (scale*mX);
        const unsigned int height = (unsigned int) (scale*mY);

        cv::Mat tmpMap(cv::Size(width, height), CV_32FC3, cv::Scalar(0.0, 0.0,
0.0));

        // ... for each filter
        for (unsigned int filter = 0, filterSize = mNodes[map].second.size();
filter < filterSize; ++filter) {
            const double orientation =
mNodes[map].second[filter].first.getOrientation();
            const Matrix<float> stimuliFilter = extractNextFilter(itStimu,
scale, 0.5, 0.5, false, false);
            const cv::Mat img = (cv::Mat) stimuliFilter;

            cv::Mat tmpFilterHsv(cv::Size(width, height), CV_32FC3,
cv::Scalar(0.0, 0.0, 0.0));

            for (int i = 0; i < img.rows; ++i) {
                const float* rowPtr = img.ptr<float>(i);

                for (int j = 0; j < img.cols; ++j, ++rowPtr) {
                    if (1.0 - (*rowPtr) <= mDiscardedLateStimuli) {
                        tmpFilterHsv.at<cv::Vec3f>(i,j) =
cv::Vec3f(360.0*orientation,       // Hue scale in OpenCV is 0.0 - 360.0
                            (singleOrientation) ? 0.0 : 1.0,
                            (*rowPtr));
                    }
                }
            }

            cv::Mat tmpFilter;
            cv::cvtColor(tmpFilterHsv, tmpFilter, CV_HSV2BGR);
            tmpFilter/= filterSize;
            tmpMap+= tmpFilter;
        }

        cv::Mat tmpMapResized;

        if (scale < 1.0)
            cv::resize(tmpMap, tmpMapResized, cv::Size(mX, mY));
        else
            tmpMapResized = tmpMap;

        tmpMapResized/= mapSize;
        img+= tmpMapResized;
    }

    cv::Mat img8U;
    img.convertTo(img8U, CV_8U, 255.0);
    return img8U;
}

cv::Mat N2D2::Environment::reconstructMeanFrame(const std::string& className,
bool normalize) const {
    const unsigned int cls = mStimuliCls.at(className);
    const unsigned int length = std::count_if(mStimuli.begin(), mStimuli.end(),
        std::bind(std::equal_to<unsigned int>(), cls,
            std::bind(&Utils::pairSecond<std::string, unsigned int>,
std::placeholders::_1)));

    if (length == 0)
        throw std::runtime_error("No frame in this class");

    cv::Mat img(cv::Size(mX, mY), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));

    for (std::vector<std::pair<std::string, unsigned int> >::const_iterator it =
mStimuli.begin(), itBegin = mStimuli.begin(),
        itEnd = mStimuli.end(); it != itEnd; ++it)
    {
        if ((*it).second == cls) {
            cv::Mat imgFrame;
            reconstructFrame(it - itBegin).convertTo(imgFrame, CV_32F,
1.0/length);
            img+= imgFrame;
        }
    }

    if (normalize) {
        cv::Mat imgNorm;
        cv::normalize(img.reshape(1), imgNorm, 0.0, 255.0, cv::NORM_MINMAX);
        img = imgNorm.reshape(3);
    }

    cv::Mat img8U;
    img.convertTo(img8U, CV_8U);
    return img8U;
}

void N2D2::Environment::reconstructMeanFrames(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    for (std::map<std::string, unsigned int>::const_iterator it =
mStimuliCls.begin(), itEnd = mStimuliCls.end();
        it != itEnd; ++it)
    {
        std::ostringstream fileName;
        fileName << dirName << "/" << (*it).first << ".jpg";

        cv::Mat img;
        cv::resize(reconstructMeanFrame((*it).first), img, cv::Size(512, 512),
0.0, 0.0, cv::INTER_NEAREST);
        cv::imwrite(fileName.str(), img);
    }
}

double N2D2::Environment::baselineTestCompare(const std::string& fileName) const
{
    if (mStimuliTest.empty())
        throw std::runtime_error("Baseline test compare: no test stimuli");

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create compare data file: " +
fileName);

    // Construct the mean image for each class
    std::map<unsigned int, cv::Mat> meanFrames;

    for (std::map<std::string, unsigned int>::const_iterator it =
mStimuliCls.begin(), itEnd = mStimuliCls.end(); it != itEnd; ++it)
        meanFrames.insert(std::make_pair((*it).second,
reconstructMeanFrame((*it).first)));

    // Compute correlation between images and the mean image of each class
    std::vector<std::vector<double> > correlation(mStimuliTest.size(),
std::vector<double>(meanFrames.size(), 0.0));

    unsigned int compareError = 0;

    for (std::vector<std::pair<std::string, unsigned int> >::const_iterator it =
mStimuliTest.begin(),
        itBegin = mStimuliTest.begin(), itEnd = mStimuliTest.end(); it != itEnd;
++it)
    {
        const unsigned int clsIdx = (*it).second;
        cv::Mat frame(reconstructFrame(it - itBegin, Test));

        double minError = 1.0;
        bool badCompare = true;

        data << (it - itBegin);

        for (std::map<unsigned int, cv::Mat>::const_iterator itMean =
meanFrames.begin(), itMeanEnd = meanFrames.end();
            itMean != itMeanEnd; ++itMean)
        {
            const double meanError = cv::norm(frame, (*itMean).second,
CV_L2)/(double) (frame.rows*frame.cols);
            correlation[it - itBegin][(*itMean).first] = meanError;

            if (meanError < minError) {
                minError = meanError;
                badCompare = ((*itMean).first != clsIdx);
            }

            data << "   " << meanError;

            if ((*itMean).first == clsIdx)
                data << "*";
        }

        if (badCompare) {
            data << "   BAD";
            ++compareError;
        }

        data << std::endl;
    }

    const double baselineScore = (mStimuliTest.size() - compareError) /
((double) mStimuliTest.size());

    data << std::endl << "BASELINE SCORE: " << (baselineScore*100) << "%" <<
std::endl;
    return baselineScore;
}

void N2D2::Environment::reconstructFilters(const std::string& dirName) const {
    Utils::createDirectories(dirName);

    // For each scale ...
    for (unsigned int map = 0, mapSize = mNodes.size(); map < mapSize; ++map) {
        // ... sample the image and for each filter
        for (unsigned int filter = 0, filterSize = mNodes[map].second.size();
filter < filterSize; ++filter) {
            std::ostringstream fileName;
            fileName << dirName << "/map_" << map << "_filter_" << filter <<
".jpg";

            mNodes[map].second[filter].first.reconstructFilter(fileName.str());
        }
    }
}
*/
