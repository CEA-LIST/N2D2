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

#include "NodeNeuron.hpp"

#include "NodeNeuron_Reflective.hpp"

N2D2::NodeNeuron::NodeNeuron(Network& net)
    : Node(net),
      // Internal variables
      mInitializedState(false),
      mStateLogPlot(false),
      mCacheValid(false)
{
    // ctor
}

void N2D2::NodeNeuron::addLink(Node* origin)
{
    if (mLinks.find(origin) != mLinks.end())
        throw std::logic_error("Synaptic link already exists!");

    // Internal parameters for image reconstruction and data representation
    if (mLayer <= origin->getLayer())
        mLayer = origin->getLayer() + 1;

    if (mLinks.empty())
        mArea = origin->getArea();
    else {
        const Area& area = origin->getArea();

        if (area.x < mArea.x)
            mArea.x = area.x;
        if (area.y < mArea.y)
            mArea.y = area.y;
        if (area.x + area.width > mArea.x + mArea.width)
            mArea.width = area.x + area.width - mArea.x;
        if (area.y + area.height > mArea.y + mArea.height)
            mArea.height = area.y + area.height - mArea.y;
    }

    // Add the connexion
    mLinks.insert(std::make_pair(origin, newSynapse()));
    origin->addBranch(this);
}

void N2D2::NodeNeuron::addLateralBranch(NodeNeuron* lateralBranch)
{
    if (std::find(mLateralBranches.begin(),
                  mLateralBranches.end(),
                  lateralBranch) != mLateralBranches.end())
        throw std::logic_error("Lateral inhibition already exists!");

    mLateralBranches.push_back(lateralBranch);
}

void N2D2::NodeNeuron::notify(Time_T timestamp, NotifyType notify)
{
    Node::notify(timestamp, notify);

    if (notify == Initialize)
        initialize();
    else if (notify == Finalize) {
        finalize();

        if (mStateLog.is_open() && mStateLogPlot) {
            mStateLog.flush();
            logStatePlot();
        }

        mCacheValid = false; // Cache invalided at the end of a new run
    } else if (notify == Reset) {
        reset(timestamp);

        if (mStateLog.is_open())
            mStateLog.close();
    } else if (notify == Load)
        load(mNet.getLoadSavePath());
    else if (notify == Save)
        save(mNet.getLoadSavePath());
}

void N2D2::NodeNeuron::readActivity(const std::vector<Time_T>& activity)
{
    // /!\ C'est bien Node::emitSpike() qui doit être appelé ici et pas
    // NodeNeuron::emitSpike()
    // Ca ne marche pas avec un bind() car c'est alors NodeNeuron::emitSpike()
    // qui est appelé !
    for (std::vector<Time_T>::const_iterator it = activity.begin(),
                                             itEnd = activity.end();
         it != itEnd;
         ++it)
        mNet.newEvent(this, NULL, (*it));
}

void N2D2::NodeNeuron::save(const std::string& dirName) const
{
    // Save parameters
    std::ostringstream fileName;
    fileName << dirName << "/neuron_" << mId << ".cfg";
    saveParameters(fileName.str());

    // Save synapses
    fileName.str(std::string());
    fileName << dirName << "/neuron_" << mId << ".syn";
    std::ofstream syn(fileName.str().c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName.str());

    std::map<NodeId_T, Synapse*> orderedLinks;

    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it)
        orderedLinks.insert(std::make_pair((*it).first->getId(), (*it).second));

    for (std::map<NodeId_T, Synapse*>::const_iterator it = orderedLinks.begin(),
                                                      itEnd
                                                      = orderedLinks.end();
         it != itEnd;
         ++it)
        (*it).second->saveInternal(syn);

    // Save internal state
    fileName.str(std::string());
    fileName << dirName << "/neuron_" << mId << ".var";
    std::ofstream data(fileName.str().c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not create internal state file (.VAR): "
                                 + fileName.str());

    saveInternal(data);
}

void N2D2::NodeNeuron::load(const std::string& dirName)
{
    // Load parameters
    std::ostringstream fileName;
    fileName << dirName << "/neuron_" << mId << ".cfg";
    loadParameters(fileName.str());

    // Load synapses
    fileName.str(std::string());
    fileName << dirName << "/neuron_" << mId << ".syn";
    std::ifstream syn(fileName.str().c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not open synaptic file (.SYN): "
                                 + fileName.str());

    std::map<NodeId_T, Synapse*> orderedLinks;

    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it)
        orderedLinks.insert(std::make_pair((*it).first->getId(), (*it).second));

    for (std::map<NodeId_T, Synapse*>::const_iterator it = orderedLinks.begin(),
                                                      itEnd
                                                      = orderedLinks.end();
         it != itEnd;
         ++it)
        (*it).second->loadInternal(syn);

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName.str());
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName.str());
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: "
            + fileName.str());

    // Load internal state
    fileName.str(std::string());
    fileName << dirName << "/neuron_" << mId << ".var";
    std::ifstream data(fileName.str().c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not open internal state file (.VAR): "
                                 + fileName.str());

    loadInternal(data);

    mInitializedState = true;
    mCacheValid = false;
}

void N2D2::NodeNeuron::logWeights(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create weights log file: "
                                 + fileName);

    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it) {
        const Area& area = (*it).first->getArea();

        // MAP X Y WEIGHT TIMING
        data << (*it).first->getScale() << " " << (*it).first->getOrientation()
             << " " << area.x << " " << area.y << " "
             << (*it).second->getRelativeWeight() << " "
             << (*it).first->getLastActivationTime() << "\n";
    }
}

void
N2D2::NodeNeuron::logState(const std::string& fileName, bool append, bool plot)
{
    if (mStateLog.is_open())
        mStateLog.close();
    else if (!append)
        mStateLogFile.clear();

    if (!fileName.empty())
        mStateLogFile = fileName;

    if (!fileName.empty() || (append && !mStateLogFile.empty())) {
        if (append)
            mStateLog.open(mStateLogFile.c_str(), std::ofstream::app);
        else
            mStateLog.open(mStateLogFile.c_str());

        if (!mStateLog.good())
            throw std::runtime_error("Could not create state log file: "
                                     + mStateLogFile);

        // Use the full double precision to keep accuracy even on small scales
        mStateLog.precision(std::numeric_limits<double>::digits10 + 1);

        mStateLogPlot = plot;
    }
}

void N2D2::NodeNeuron::logStats(std::ofstream& dataFile,
                                const std::string& suffix) const
{
    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it) {
        std::ostringstream suffixStr;
        suffixStr << (*it).first->getId() << " " << suffix;
        (*it).second->logStats(dataFile, suffixStr.str());
    }
}

void N2D2::NodeNeuron::clearStats()
{
    for (std::unordered_map<Node*, Synapse*>::iterator it = mLinks.begin(),
                                                       itEnd = mLinks.end();
         it != itEnd;
         ++it)
        (*it).second->clearStats();
}

cv::Mat N2D2::NodeNeuron::reconstructPattern(bool normalize, bool multiLayer)
{
    Utils::createDirectories("_cache");

    std::ostringstream fileName;
    fileName << "_cache/node-" << mId << ".ppm";

    cv::Mat img;

    if (mCacheValid) {
        img = cv::imread(fileName.str());

        if (!img.data)
            throw std::runtime_error("NodeNeuron::reconstructPattern(): Could "
                                     "not open or find image: "
                                     + fileName.str());
    } else {
        img = cv::Mat(cv::Size(mArea.width, mArea.height), CV_32FC3, 0.0);

        NodeNeuron* nodePtr = dynamic_cast<NodeNeuron*>(mLinks.begin()->first);
        NodeNeuron_ReflectiveBridge* nodeBridgePtr = dynamic_cast
            <NodeNeuron_ReflectiveBridge*>(mLinks.begin()->first);
        NodeSync* nodeSyncPtr = dynamic_cast<NodeSync*>(mLinks.begin()->first);

        // Pour la reconstruction des poids sur plusieurs niveaux :
        // - On vérifie que l'on est bien sur une couche qui n'est pas connecté
        // sur des NodeEnv (mLayer > 1).
        if ((mLayer > 1 && nodePtr != NULL && nodeBridgePtr == NULL)
            || (mLayer > 2 && nodeSyncPtr != NULL)) {
            std::vector<std::vector<unsigned int> > mask(
                mArea.width, std::vector<unsigned int>(mArea.height, 0));
            unsigned int max = 0;
            double wSum = 0.0;

            for (std::unordered_map<Node*, Synapse*>::const_iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                const Area& area = (*it).first->getArea();
                wSum += (*it).second->getRelativeWeight();

                for (unsigned int x = area.x; x < area.x + area.width; ++x) {
                    for (unsigned int y = area.y; y < area.y + area.height;
                         ++y) {
                        ++mask[x][y];

                        if (mask[x][y] > max)
                            max = mask[x][y];
                    }
                }
            }

            cv::Mat imgMask(cv::Size(mArea.width, mArea.height), CV_32FC1, 0.0);

            for (unsigned int x = 0; x < mArea.width; ++x) {
                for (unsigned int y = 0; y < mArea.height; ++y)
                    imgMask.at<float>(y, x) = 1.0 / mask[x][y];
            }
            /*
                    // DEBUG
                    cv::Mat imgMaskDebug, imgMaskDebugResized;
                    cv::normalize(imgMask.reshape(1), imgMaskDebug, 0.0, 255.0,
               cv::NORM_MINMAX);
                    cv::resize(imgMaskDebug.reshape(3), imgMaskDebugResized,
               cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST);
                    cv::imwrite("mask.jpg", imgMaskDebugResized);
            */

            for (std::unordered_map<Node*, Synapse*>::const_iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                const Area& area = (*it).first->getArea();
                const double w = (*it).second->getRelativeWeight();
                cv::Mat subImg;

                if (multiLayer) {
                    NodeNeuron* parent = NULL;

                    if (nodePtr != NULL)
                        parent = dynamic_cast<NodeNeuron*>((*it).first);
                    else if (nodeSyncPtr != NULL) {
                        // It has to be a NodeSync, so we take its link
                        NodeSync* parentSync = dynamic_cast
                            <NodeSync*>((*it).first);

                        if (parentSync != NULL)
                            parent = dynamic_cast
                                <NodeNeuron*>(parentSync->getParent());
                    }

                    if (parent == NULL)
                        throw std::runtime_error(
                            "Unknown node type for parent node.");

                    parent->reconstructPattern(true)
                        .convertTo(subImg, CV_32F, 1.0 / 255.0);
                    subImg *= w / max;
                } else
                    subImg = cv::Mat(
                        cv::Size(area.width, area.height), CV_32FC3, w / max);

                img(cv::Rect(area.x, area.y, area.width, area.height))
                    += subImg;
            }
            /*
                    // DEBUG
                    cv::Mat imgDebug, imgDebugResized;
                    cv::normalize(img.reshape(1), imgDebug, 0.0, 255.0,
               cv::NORM_MINMAX);
                    cv::resize(imgDebug.reshape(3), imgDebugResized,
               cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST);
                    cv::imwrite("img.jpg", imgDebugResized);
            */
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            channels[0] = channels[0].mul(imgMask);
            channels[1] = channels[1].mul(imgMask);
            channels[2] = channels[2].mul(imgMask);
            cv::merge(channels, img);

            if (max / wSum > 1.0)
                img *= max * (max / wSum);
            else
                img *= max;
        } else {
            // Construct pyramidal image representation
            std::map<double, std::map<double, cv::Mat> > orientedMaps;
            bool singleOrientation = true;

            for (std::unordered_map<Node*, Synapse*>::const_iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                const double scale = (*it).first->getScale();
                const double orientation = (*it).first->getOrientation();
                const Area& area = (*it).first->getArea();

                if (orientedMaps.find(scale) == orientedMaps.end()
                    || orientedMaps[scale].find(orientation)
                       == orientedMaps[scale].end()) {
                    orientedMaps[scale].insert(std::make_pair(
                        orientation,
                        cv::Mat(cv::Size((unsigned int)(scale * mArea.width),
                                         (unsigned int)(scale * mArea.height)),
                                CV_32FC3,
                                0.0)));

                    if (orientedMaps[scale].size() > 1)
                        singleOrientation = false;
                }

                orientedMaps[scale][orientation].at
                    <cv::Vec3f>(area.y - (unsigned int)(scale * mArea.y),
                                area.x - (unsigned int)(scale* mArea.x))
                    = cv::Vec3f(360.0 * orientation,
                                1.0,
                                (*it).second->getRelativeWeight());
            }

            // DEBUG
            /*
                    for (std::map<double, std::map<double, cv::Mat> >::iterator
               it = orientedMaps.begin(),
                        itEnd = orientedMaps.end(); it != itEnd; ++it)
                    {
                        for (std::map<double, cv::Mat>::iterator itOrientation =
               (*it).second.begin(),
                            itOrientationEnd = (*it).second.end(); itOrientation
               != itOrientationEnd; ++itOrientation)
                        {
                            std::ostringstream buf;
                            buf << "filter_" << (*itOrientation).first <<
               ".png";
                            cv::Mat i, iResized;
                            cv::cvtColor((*itOrientation).second, i,
               cv::COLOR_HSV2BGR);
                            cv::resize(i, iResized, cv::Size(512, 512), 0.0,
               0.0, cv::INTER_NEAREST);
                            cv::imwrite(buf.str(), iResized);
                        }
                    }
            */
            // Fusion filters to a single image per scale
            std::map<double, cv::Mat> maps;

            for (std::map<double, std::map<double, cv::Mat> >::iterator it
                 = orientedMaps.begin(),
                 itEnd = orientedMaps.end();
                 it != itEnd;
                 ++it) {
                for (std::map<double, cv::Mat>::iterator itOrientation
                     = (*it).second.begin(),
                     itOrientationEnd = (*it).second.end();
                     itOrientation != itOrientationEnd;
                     ++itOrientation) {
                    const double scale = (*it).first;

                    if (maps.find(scale) == maps.end()) {
                        maps.insert(std::make_pair(
                            scale,
                            cv::Mat(
                                cv::Size((unsigned int)(scale * mArea.width),
                                         (unsigned int)(scale * mArea.height)),
                                CV_32FC3,
                                0.0)));
                    }

                    cv::Mat filterHsv;

                    if (singleOrientation) {
                        // Set saturation to 0
                        std::vector<cv::Mat> channels;
                        cv::split((*itOrientation).second, channels);
                        channels[1] *= 0.0;
                        cv::merge(channels, filterHsv);
                    } else
                        filterHsv = (*itOrientation).second;

                    cv::Mat filter;
                    cv::cvtColor(filterHsv, filter, CV_HSV2BGR);
                    filter *= 2.0 / (*it).second.size();
                    maps[scale] += filter;
                }
            }

            // Fusion pyramidal image reprensentation to a single image
            if (maps.size() > 1) {
                for (std::map<double, cv::Mat>::iterator it = maps.begin(),
                                                         itEnd = maps.end();
                     it != itEnd;
                     ++it) {
                    const double scale = (*it).first;
                    cv::Mat resized;

                    if (scale < 1.0)
                        cv::resize((*it).second,
                                   resized,
                                   cv::Size(mArea.width, mArea.height));
                    else
                        resized = (*it).second;

                    cv::Mat blured;
                    cv::GaussianBlur(resized,
                                     blured,
                                     cv::Size(1.0 / scale, 1.0 / scale),
                                     0.5 / scale);
                    blured /= maps.size();
                    img += blured;
                }
            } else
                img = maps.begin()->second;

            if (singleOrientation) {
                cv::Mat imgGray;
                cv::cvtColor(img, imgGray, CV_RGB2GRAY);
                img = imgGray;
            }
        }

        cv::Mat img8U;
        img.convertTo(img8U, CV_8U, 255.0);
        img = img8U;

        if (!cv::imwrite(fileName.str(), img))
            throw std::runtime_error("Unable to write image: "
                                     + fileName.str());

        mCacheValid = true;
    }

    if (normalize) {
        cv::Mat imgNorm;
        cv::normalize(img.reshape(1), imgNorm, 0.0, 255.0, cv::NORM_MINMAX);
        img = imgNorm.reshape(3);
    }

    return img;
}

cv::Mat N2D2::NodeNeuron::reconstructActivity(
    Time_T start, Time_T stop, EventType_T type, bool order, bool normalize)
{
    double normFactor = 0;
    double value;
    bool validTime = false;

    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it) {
        if (order)
            std::tie(value, validTime)
                = (*it).first->getFirstActivationTime(start, stop, type);
        else
            value = (*it).first->getActivity(start, stop, type);

        if (value > normFactor && (!order || (order && validTime)))
            normFactor = value;
    }

    cv::Mat img(cv::Size(mArea.width, mArea.height), CV_32FC3, 0.0);

    if (normFactor == 0)
        return img;

    NodeNeuron* nodePtr = dynamic_cast<NodeNeuron*>(mLinks.begin()->first);
    NodeNeuron_ReflectiveBridge* nodeBridgePtr = dynamic_cast
        <NodeNeuron_ReflectiveBridge*>(mLinks.begin()->first);
    NodeSync* nodeSyncPtr = dynamic_cast<NodeSync*>(mLinks.begin()->first);

    if ((mLayer > 1 && nodePtr != NULL && nodeBridgePtr == NULL)
        || (mLayer > 2 && nodeSyncPtr != NULL)) {
        std::vector<std::vector<unsigned int> > mask(
            mArea.width, std::vector<unsigned int>(mArea.height, 0));
        unsigned int max = 0;

        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            const Area& area = (*it).first->getArea();

            for (unsigned int x = area.x; x < area.x + area.width; ++x) {
                for (unsigned int y = area.y; y < area.y + area.height; ++y) {
                    ++mask[x][y];

                    if (mask[x][y] > max)
                        max = mask[x][y];
                }
            }
        }

        cv::Mat imgMask(cv::Size(mArea.width, mArea.height), CV_32FC1, 0.0);

        for (unsigned int x = 0; x < mArea.width; ++x) {
            for (unsigned int y = 0; y < mArea.height; ++y)
                imgMask.at<float>(y, x) = 1.0 / mask[x][y];
        }

        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            const Area& area = (*it).first->getArea();

            if (order) {
                std::tie(value, validTime)
                    = (*it).first->getFirstActivationTime(start, stop, type);

                if (validTime)
                    value = (normFactor - value) / normFactor;
            } else
                value = (*it).first->getActivity(start, stop, type)
                        / normFactor;
            /*
                        NodeNeuron* parent = NULL;

                        if (nodePtr != NULL)
                            parent = dynamic_cast<NodeNeuron*> ((*it).first);
                        else if (nodeSyncPtr != NULL) {
                            // It has to be a NodeSync, so we take its link
                            NodeSync* parentSync = dynamic_cast<NodeSync*>
               ((*it).first);

                            if (parentSync != NULL)
                                parent = dynamic_cast<NodeNeuron*>
               (parentSync->getParent());
                        }

                        if (parent == NULL)
                            throw std::runtime_error("Unknown node type for
               parent node.");

                        cv::Mat subImg;
                        parent->reconstructActivity(start, stop, type,
               order).convertTo(subImg, CV_32F, 1.0/255.0);
                        subImg*= value/max;
            */

            cv::Mat subImg = cv::Mat(
                cv::Size(area.width, area.height), CV_32FC3, value / max);
            img(cv::Rect(area.x, area.y, area.width, area.height)) += subImg;
        }

        img *= imgMask;
        img *= max;
    } else {
        // Construct pyramidal image representation
        std::map<double, std::map<double, cv::Mat> > orientedMaps;
        bool singleOrientation = true;

        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            const double scale = (*it).first->getScale();
            const double orientation = (*it).first->getOrientation();
            const Area& area = (*it).first->getArea();

            if (orientedMaps.find(scale) == orientedMaps.end()
                || orientedMaps[scale].find(orientation)
                   == orientedMaps[scale].end()) {
                orientedMaps[scale].insert(std::make_pair(
                    orientation,
                    cv::Mat(cv::Size((unsigned int)(scale * mArea.width),
                                     (unsigned int)(scale * mArea.height)),
                            CV_32FC3,
                            0.0)));

                if (orientedMaps[scale].size() > 1)
                    singleOrientation = false;
            }

            if (order) {
                std::tie(value, validTime)
                    = (*it).first->getFirstActivationTime(start, stop, type);

                if (validTime)
                    value = 1.0 - value / normFactor;
            } else
                value = (*it).first->getActivity(start, stop, type)
                        / normFactor;

            orientedMaps[scale][orientation].at
                <cv::Vec3f>(area.x - (unsigned int)(scale * mArea.x),
                            area.y - (unsigned int)(scale* mArea.y))
                = cv::Vec3f(360.0 * orientation, 0.5, value);
        }

        // Fusion filters to a single image per scale
        std::map<double, cv::Mat> maps;

        for (std::map<double, std::map<double, cv::Mat> >::iterator it
             = orientedMaps.begin(),
             itEnd = orientedMaps.end();
             it != itEnd;
             ++it) {
            for (std::map<double, cv::Mat>::iterator itOrientation
                 = (*it).second.begin(),
                 itOrientationEnd = (*it).second.end();
                 itOrientation != itOrientationEnd;
                 ++itOrientation) {
                const double scale = (*it).first;

                if (maps.find(scale) == maps.end()) {
                    maps.insert(std::make_pair(
                        scale,
                        cv::Mat(cv::Size((unsigned int)(scale * mArea.width),
                                         (unsigned int)(scale * mArea.height)),
                                CV_32FC3,
                                0.0)));
                }

                cv::Mat filter;
                cv::cvtColor((*itOrientation).second, filter, CV_HSV2BGR);
                filter /= (*it).second.size();
                maps[scale] += filter;
            }
        }

        // Fusion pyramidal image reprensentation to a single image
        if (maps.size() > 1) {
            for (std::map<double, cv::Mat>::iterator it = maps.begin(),
                                                     itEnd = maps.end();
                 it != itEnd;
                 ++it) {
                const double scale = (*it).first;
                cv::Mat resized;

                if (scale < 1.0)
                    cv::resize((*it).second,
                               resized,
                               cv::Size(mArea.width, mArea.height));
                else
                    resized = (*it).second;

                cv::Mat blured;
                cv::GaussianBlur(resized,
                                 blured,
                                 cv::Size(1.0 / scale, 1.0 / scale),
                                 0.5 / scale);
                blured /= maps.size();
                img += blured;
            }
        } else
            img = maps.begin()->second;

        if (singleOrientation) {
            cv::Mat imgGray;
            cv::cvtColor(img, imgGray, CV_RGB2GRAY);
            img = imgGray;
        }
    }

    if (normalize) {
        cv::Mat imgNorm;
        cv::normalize(img.reshape(1), imgNorm, 0.0, 1.0, cv::NORM_MINMAX);
        img = imgNorm.reshape(3);
    }

    cv::Mat img8U;
    img.convertTo(img8U, CV_8U, 255.0);
    return img8U;
}

N2D2::NodeNeuron::~NodeNeuron()
{
    // dtor
    for (std::unordered_map<Node*, Synapse*>::iterator it = mLinks.begin(),
                                                       itEnd = mLinks.end();
         it != itEnd;
         ++it) {
        // Avoid segmentation fault if shared synapses are used...
        if ((*it).second != NULL) {
            delete (*it).second;
            (*it).second = NULL;
        }
    }
}
