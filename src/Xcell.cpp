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

#include "Xcell.hpp"

unsigned int N2D2::Xcell::mIdCnt = 1;

N2D2::Xcell::Xcell(Network& net)
    : NetworkObserver(net),
      mSyncClock(this, "SyncClock", 0 * TimeS),
      // Variables
      mId(mIdCnt++)
{
    // ctor
}

void N2D2::Xcell::addLateralInhibition(Xcell& cell)
{
    for (std::vector<NodeNeuron*>::const_iterator it = cell.mNeurons.begin(),
                                                  itEnd = cell.mNeurons.end();
         it != itEnd;
         ++it) {
        for (std::vector<NodeNeuron*>::const_iterator itInner
             = mNeurons.begin(),
             itInnerEnd = mNeurons.end();
             itInner != itInnerEnd;
             ++itInner) {
            (*it)->addLateralBranch(*itInner);
        }
    }
}

void N2D2::Xcell::addInput(Node* node)
{
    if (mSyncClock > 0) {
        mSyncs.push_back(new NodeSync(mNet, *this));
        mSyncs.back()->addLink(node);
        std::for_each(mNeurons.begin(),
                      mNeurons.end(),
                      std::bind(&NodeNeuron::addLink,
                                std::placeholders::_1,
                                mSyncs.back()));
    } else {
        std::for_each(
            mNeurons.begin(),
            mNeurons.end(),
            std::bind(&NodeNeuron::addLink, std::placeholders::_1, node));
    }
}

void N2D2::Xcell::addInput(Xcell& cell)
{
    std::for_each(cell.mNeurons.begin(),
                  cell.mNeurons.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(Node*)>(&Xcell::addInput),
                            this,
                            std::placeholders::_1));
}

void N2D2::Xcell::addInput(Environment& env,
                           unsigned int channel,
                           unsigned int x0,
                           unsigned int length)
{
    addInput(env, channel, x0, 0, length, 1);
}

void N2D2::Xcell::addInput(Environment& env,
                           unsigned int channel,
                           unsigned int x0,
                           unsigned int y0,
                           unsigned int width,
                           unsigned int height)
{
    for (unsigned int x = x0; x < x0 + width; ++x) {
        for (unsigned int y = y0; y < y0 + height; ++y)
            addInput(env.getNode(channel, x, y));
    }
}

void N2D2::Xcell::addInput(Environment& env,
                           unsigned int x0,
                           unsigned int y0,
                           unsigned int width,
                           unsigned int height)
{
    for (unsigned int channel = 0, nbChannels = env.getNbChannels();
         channel < nbChannels;
         ++channel)
        addInput(env, channel, x0, y0, width, height);
}

void N2D2::Xcell::addMultiscaleInput(HeteroEnvironment& env,
                                     unsigned int x0,
                                     unsigned int y0,
                                     unsigned int width,
                                     unsigned int height)
{
    for (unsigned int map = 0, size = env.size(); map < size; ++map) {
        const double scale
            = env[map]->getSizeX()
              / (double)env[0]
                    ->getSizeX(); // TODO: unsafe legacy code adaptation
        const unsigned int scaledX0 = (unsigned int)(scale * x0);
        const unsigned int scaledY0 = (unsigned int)(scale * y0);
        const unsigned int scaledWidth = (unsigned int)(scale * width);
        const unsigned int scaledHeight = (unsigned int)(scale * height);

        addInput(*(env[map]), scaledX0, scaledY0, scaledWidth, scaledHeight);
    }
}

void N2D2::Xcell::addBridge(Environment& env,
                            unsigned int channel,
                            unsigned int x0,
                            unsigned int length)
{
    addBridge(env, channel, x0, 0, length, 1);
}

void N2D2::Xcell::addBridge(Environment& env,
                            unsigned int channel,
                            unsigned int x0,
                            unsigned int y0,
                            unsigned int width,
                            unsigned int height)
{
    static std::vector<NodeNeuron*>::iterator it = mNeurons.begin();

    for (unsigned int x = x0; x < x0 + width; ++x) {
        for (unsigned int y = y0; y < y0 + height; ++y) {
            if (it == mNeurons.end())
                throw std::runtime_error(
                    "Not enough neurons in the cell to complete the bridge");

            if (mSyncClock > 0) {
                mSyncs.push_back(new NodeSync(mNet, *this));
                mSyncs.back()->addLink(env.getNode(channel, x, y));
                (*it)->addLink(mSyncs.back());
            } else
                (*it)->addLink(env.getNode(channel, x, y));

            ++it;
        }
    }
}

void N2D2::Xcell::addBridge(Environment& env,
                            unsigned int x0,
                            unsigned int y0,
                            unsigned int width,
                            unsigned int height)
{
    for (unsigned int channel = 0, nbChannels = env.getNbChannels();
         channel < nbChannels;
         ++channel)
        addBridge(env, channel, x0, y0, width, height);
}

void N2D2::Xcell::addMultiscaleBridge(HeteroEnvironment& env,
                                      unsigned int x0,
                                      unsigned int y0,
                                      unsigned int width,
                                      unsigned int height)
{
    for (unsigned int map = 0, size = env.size(); map < size; ++map) {
        const double scale
            = env[map]->getSizeX()
              / (double)env[0]
                    ->getSizeX(); // TODO: unsafe legacy code adaptation
        const unsigned int scaledX0 = (unsigned int)(scale * x0);
        const unsigned int scaledY0 = (unsigned int)(scale * y0);
        const unsigned int scaledWidth = (unsigned int)(scale * width);
        const unsigned int scaledHeight = (unsigned int)(scale * height);

        addBridge(*(env[map]), scaledX0, scaledY0, scaledWidth, scaledHeight);
    }
}

void N2D2::Xcell::notify(Time_T timestamp, NotifyType notify)
{
    if (notify == Reset) {
        if (mSyncClock > 0) {
            unsigned int i = 0;

            while (!mSyncFifo.empty()) {
                mNet.newEvent(
                    mSyncFifo.front(), NULL, timestamp + mSyncClock * i);
                mSyncFifo.pop();
                ++i;
            }
        }
    } else if (notify == Load)
        load(mNet.getLoadSavePath(), true);
    else if (notify == Save)
        save(mNet.getLoadSavePath(), true);
}

void N2D2::Xcell::readActivity(const std::string& fileName)
{
    std::ifstream activity(fileName.c_str());

    if (!activity.good())
        throw std::runtime_error("Could not open activity file: " + fileName);

    std::map<NodeId_T, std::vector<Time_T> > record;

    while (activity.good()) {
        NodeId_T nodeId;
        double time;

        activity >> nodeId;
        activity >> time;
        activity.ignore(std::numeric_limits<std::streamsize>::max(),
                        '\n'); // Go to next line

        record[nodeId].push_back((Time_T)(time * TimeS));
    }

    for (std::vector<NodeNeuron*>::const_iterator it = mNeurons.begin(),
                                                  itEnd = mNeurons.end();
         it != itEnd;
         ++it) {
        const NodeId_T id = (*it)->getId();

        if (record.find(id) != record.end())
            (*it)->readActivity(record[id]);
    }
}

void N2D2::Xcell::save(const std::string& dirName, bool cellStateOnly) const
{
    Utils::createDirectories(dirName);

    // Save parameters
    std::ostringstream fileName;
    fileName << dirName << "/xcell_" << mId << ".cfg";
    saveParameters(fileName.str());

    if (!cellStateOnly)
        std::for_each(
            mNeurons.begin(),
            mNeurons.end(),
            std::bind(&NodeNeuron::save, std::placeholders::_1, dirName));
}

void N2D2::Xcell::load(const std::string& dirName, bool cellStateOnly)
{
    // Load parameters
    std::ostringstream fileName;
    fileName << dirName << "/xcell_" << mId << ".cfg";
    loadParameters(fileName.str());

    if (!cellStateOnly)
        std::for_each(
            mNeurons.begin(),
            mNeurons.end(),
            std::bind(&NodeNeuron::load, std::placeholders::_1, dirName));
}

void
N2D2::Xcell::logState(const std::string& dirName, bool append, bool plot) const
{
    Utils::createDirectories(dirName);

    std::ostringstream fileName;

    for (std::vector<NodeNeuron*>::const_iterator it = mNeurons.begin(),
                                                  itEnd = mNeurons.end();
         it != itEnd;
         ++it) {
        fileName.str(std::string());
        fileName << dirName << "/l" << mId << "-[" << (*it)->getId() << "].dat";

        (*it)->logState(fileName.str(), append, plot);
    }
}

void N2D2::Xcell::logStats(std::ofstream& dataFile) const
{
    for (std::vector<NodeNeuron*>::const_iterator it = mNeurons.begin(),
                                                  itEnd = mNeurons.end();
         it != itEnd;
         ++it) {
        std::ostringstream id;
        id << (*it)->getId();
        (*it)->logStats(dataFile, id.str());
    }
}

void N2D2::Xcell::logStats(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create stats log file: "
                                 + fileName);

    logStats(data);
}

void N2D2::Xcell::clearStats()
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(&NodeNeuron::clearStats, std::placeholders::_1));
}

cv::Mat N2D2::Xcell::reconstructPattern(unsigned int neuron,
                                        bool normalize,
                                        bool multiLayer) const
{
    return mNeurons.at(neuron)->reconstructPattern(normalize, multiLayer);
}

void N2D2::Xcell::reconstructPatterns(const std::string& dirName,
                                      bool normalize,
                                      bool multiLayer) const
{
    Utils::createDirectories(dirName);

    for (std::vector<NodeNeuron*>::const_iterator it = mNeurons.begin(),
                                                  itEnd = mNeurons.end();
         it != itEnd;
         ++it) {
        std::ostringstream fileName;
        fileName << dirName << "/l" << mId << "-[" << (*it)->getId() << "].jpg";

        cv::Mat img((*it)->reconstructPattern(normalize, multiLayer));

        if (img.rows == 1) {
            cv::Mat img2D;
            cv::resize(img,
                       img2D,
                       cv::Size(img.cols, 100 * img.rows),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
            img = img2D;
        }

        cv::Mat imgResized;
        cv::resize(
            img, imgResized, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST);

        if (!cv::imwrite(fileName.str(), imgResized))
            throw std::runtime_error("Unable to write image: "
                                     + fileName.str());
    }
}

void N2D2::Xcell::setActivityRecording(bool activityRecording)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(&NodeNeuron::setActivityRecording,
                            std::placeholders::_1,
                            activityRecording));
}

std::string N2D2::Xcell::getNeuronsParameter(const std::string& name) const
{
    const std::string value = mNeurons.front()->getParameter(name);

    if (std::find_if(
            mNeurons.begin(),
            mNeurons.end(),
            std::bind(
                std::not_equal_to<std::string>(),
                value,
                std::bind(
                    static_cast
                    <std::string (NodeNeuron::*)(const std::string&)const>(
                        &NodeNeuron::getParameter),
                    std::placeholders::_1,
                    name))) != mNeurons.end()) {
        throw std::runtime_error(
            "Different values within the cell for neurons parameter: " + name);
    }

    return value;
}

void N2D2::Xcell::setNeuronsParameters(const std::map
                                       <std::string, std::string>& params,
                                       bool ignoreUnknown)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(&NodeNeuron::setParameters,
                            std::placeholders::_1,
                            params,
                            ignoreUnknown));
}

void N2D2::Xcell::loadNeuronsParameters(const std::string& fileName,
                                        bool ignoreNotExists,
                                        bool ignoreUnknown)
{
    if (!std::ifstream(fileName.c_str()).good()) {
        if (ignoreNotExists)
            std::cout << "Notice: Could not open configuration file: "
                      << fileName << std::endl;
        else
            throw std::runtime_error("Could not open configuration file: "
                                     + fileName);
    } else {
        // ignoreNotExists is set to false in the following calls, because the
        // file is supposed to exist and be readable.
        // Otherwise, something is really wrong and it's probably better to
        // throw an exception!
        std::for_each(mNeurons.begin(),
                      mNeurons.end(),
                      std::bind(&NodeNeuron::loadParameters,
                                std::placeholders::_1,
                                fileName,
                                false,
                                ignoreUnknown));
    }
}

void N2D2::Xcell::saveNeuronsParameters(const std::string& fileName) const
{
    mNeurons.front()->saveParameters(fileName);
}

void N2D2::Xcell::copyNeuronsParameters(const Xcell& from)
{
    std::vector<NodeNeuron*>::const_iterator itFrom = from.mNeurons.begin();
    const std::vector<NodeNeuron*>::const_iterator itFromEnd
        = from.mNeurons.end();

    if (itFrom == itFromEnd)
        throw std::runtime_error(
            "No neuron to copy parameters from in the cell!");

    if (from.mNeurons.size() < mNeurons.size()) {
        std::cout << "Warning: the number of neurons to copy parameters from "
                     "is lower than the number of neurons in the cell"
                  << std::endl;
    }

    for (std::vector<NodeNeuron*>::const_iterator it = mNeurons.begin(),
                                                  itEnd = mNeurons.end();
         it != itEnd;
         ++it) {
        (*it)->copyParameters(*(*itFrom));

        if (itFromEnd - itFrom > 1)
            ++itFrom;
    }
}

void N2D2::Xcell::copyNeuronsParameters(const NodeNeuron& from)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(&NodeNeuron::copyParameters,
                            std::placeholders::_1,
                            std::ref(from)));
}

N2D2::Xcell::~Xcell()
{
    // dtor
    std::for_each(mNeurons.begin(), mNeurons.end(), Utils::Delete());
    std::for_each(mSyncs.begin(), mSyncs.end(), Utils::Delete());
}
