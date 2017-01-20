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

#include "Monitor.hpp"

N2D2::Monitor::Monitor(Network& net)
    : mNet(net),
      mEarlierId(0),
      mMostActiveId(0),
      mMostActiveRate(0),
      mTotalActivity(0),
      mLastEvent(0),
      mFirstEvent(0),
      mValidFirstEvent(false)
{
    // ctor
}

void N2D2::Monitor::add(Node& node)
{
    mNodes.push_back(&node);
    node.setActivityRecording(true);
}

void N2D2::Monitor::add(Xcell& cell)
{
    const std::vector<NodeNeuron*>& neurons = cell.getNeurons();
    mNodes.insert(mNodes.end(), neurons.begin(), neurons.end());
    cell.setActivityRecording(true);
}

void N2D2::Monitor::add(Layer& layer)
{
    const std::vector<Xcell*>& cells = layer.getCells();

    for (std::vector<Xcell*>::const_iterator it = cells.begin(),
                                             itEnd = cells.end();
         it != itEnd;
         ++it)
        add(*(*it));
}

void N2D2::Monitor::update(bool recordActivity)
{
    mEarlierId = 0;
    mMostActiveId = 0;
    mMostActiveRate = 0;
    mTotalActivity = 0;
    mLastEvent = mNet.getLastEvent();

    if (!mValidFirstEvent) {
        mFirstEvent = mNet.getFirstEvent();
        mValidFirstEvent = true;
    }

    Time_T first = 0;

    for (std::vector<Node*>::const_iterator it = mNodes.begin(),
                                            itEnd = mNodes.end();
         it != itEnd;
         ++it) {
        const NodeId_T nodeId = (*it)->getId();
        const NodeEvents_T& record = mNet.getSpikeRecording(nodeId);

        NodeEvents_T filteredRecord;

        // mFiringRate peut se déduire très facilement de mActivity, seulement
        // ça permet d'éviter d'avoir à garder en mémoire de
        // trop grosses quantités de données lorsque l'on a pas besoin de
        // mActivity.
        std::map<NodeId_T, std::map<EventType_T, unsigned int> >::iterator
        itFiringRate;
        std::tie(itFiringRate, std::ignore) = mFiringRate.insert(
            std::make_pair(nodeId, std::map<EventType_T, unsigned int>()));

        for (NodeEvents_T::const_iterator it = record.begin(),
                                          itEnd = record.end();
             it != itEnd;
             ++it) {
            if (mRecordEventTypes.empty()
                || mRecordEventTypes.find((*it).second)
                   != mRecordEventTypes.end()) {
                filteredRecord.push_back(*it);
                mEventTypes.insert((*it).second);
                std::map<EventType_T, unsigned int>::iterator itNodeFiringRate
                    = (*itFiringRate).second.find((*it).second);

                if (itNodeFiringRate != (*itFiringRate).second.end())
                    ++(*itNodeFiringRate).second;
                else
                    (*itFiringRate)
                        .second.insert(std::make_pair((*it).second, 1));
            }
        }

        const unsigned int activity = filteredRecord.size();

        mTotalActivity += activity;

        if (mMostActiveRate < activity) {
            mMostActiveRate = activity;
            mMostActiveId = nodeId;
        }

        if (activity > 0
            && (mEarlierId == 0 || filteredRecord[0].first < first)) {
            mEarlierId = nodeId;
            first = filteredRecord[0].first;
        }

        if (recordActivity && activity > 0)
            mActivity[nodeId].insert(mActivity[nodeId].end(),
                                     filteredRecord.begin(),
                                     filteredRecord.end());
    }

    // If no neuron fired more than once, take the first to have fired (for
    // frame-based learning)
    if (!(mMostActiveRate > 1))
        mMostActiveId = mEarlierId;
}

bool N2D2::Monitor::checkLearning(unsigned int cls,
                                  NodeId_T targetId,
                                  bool winnerIsEarlier,
                                  bool update)
{
    return checkLearningResponse(
        cls, std::vector<NodeId_T>(1, targetId), winnerIsEarlier, update);
}

bool N2D2::Monitor::checkLearningResponse(unsigned int cls,
                                          NodeId_T targetId,
                                          NodeId_T responseId,
                                          bool update)
{
    return checkLearningResponse(
        cls, std::vector<NodeId_T>(1, targetId), responseId, update);
}

bool N2D2::Monitor::checkLearning(unsigned int cls,
                                  const std::vector<NodeId_T>& targetIds,
                                  bool winnerIsEarlier,
                                  bool update)
{
    return checkLearningResponse(
        cls, targetIds, (winnerIsEarlier) ? mEarlierId : mMostActiveId, update);
}

bool N2D2::Monitor::checkLearningResponse(unsigned int cls,
                                          const std::vector
                                          <NodeId_T>& targetIds,
                                          NodeId_T responseId,
                                          bool update)
{
    bool success = false;

    if (mMostActiveRate > 0) {
        if (update) {
            std::map<NodeId_T, std::map<unsigned int, unsigned int> >::iterator
            itStats;
            std::tie(itStats, std::ignore) = mStats.insert(std::make_pair(
                responseId, std::map<unsigned int, unsigned int>()));

            std::map<unsigned int, unsigned int>::iterator itStatsNode
                = (*itStats).second.find(cls);

            if (itStatsNode != (*itStats).second.end())
                ++(*itStatsNode).second;
            else
                (*itStats).second.insert(std::make_pair(cls, 1));
        }

        success = (std::find(targetIds.begin(), targetIds.end(), responseId)
                   != targetIds.end());
    }

    mSuccess.push_back(success);
    return success;
}

bool N2D2::Monitor::checkLearning(unsigned int cls,
                                  bool winnerIsEarlier,
                                  bool update)
{
    return checkLearningResponse(
        cls, (winnerIsEarlier) ? mEarlierId : mMostActiveId, update);
}

bool N2D2::Monitor::checkLearningResponse(unsigned int cls,
                                          NodeId_T responseId,
                                          bool update)
{
    bool success = false;

    if (mMostActiveRate > 0) {
        std::map
            <NodeId_T, std::map<unsigned int, unsigned int> >::iterator itStats;
        std::tie(itStats, std::ignore) = mStats.insert(
            std::make_pair(responseId, std::map<unsigned int, unsigned int>()));

        std::map<unsigned int, unsigned int>::iterator itStatsNode
            = (*itStats).second.find(cls);

        if (update) {
            if (itStatsNode != (*itStats).second.end())
                ++(*itStatsNode).second;
            else
                (*itStats).second.insert(std::make_pair(cls, 1));
        }

        if (itStatsNode != (*itStats).second.end()) {
            success = true;

            for (std::map<unsigned int, unsigned int>::const_iterator it
                 = (*itStats).second.begin(),
                 itEnd = (*itStats).second.end();
                 it != itEnd;
                 ++it) {
                // >= plutôt que > me parait plus strict et rigoureux
                if ((*it).second >= (*itStatsNode).second && (*it).first
                                                             != cls) {
                    success = false;
                    break;
                }
            }
        }
    }

    mSuccess.push_back(success);
    return success;
}

unsigned int N2D2::Monitor::getFiringRate(NodeId_T nodeId) const
{
    std::map<NodeId_T, std::map<EventType_T, unsigned int> >::const_iterator
    itFiringRate = mFiringRate.find(nodeId);

    unsigned int firingRate = 0;

    for (std::map<EventType_T, unsigned int>::const_iterator it
         = (*itFiringRate).second.begin(),
         itEnd = (*itFiringRate).second.end();
         it != itEnd;
         ++it) {
        firingRate += (*it).second;
    }

    return firingRate;
}

unsigned int N2D2::Monitor::getTotalFiringRate() const
{
    unsigned int firingRate = 0;

    for (std::map
         <NodeId_T, std::map<EventType_T, unsigned int> >::const_iterator
             itFiringRate = mFiringRate.begin(),
             itFiringRateEnd = mFiringRate.end();
         itFiringRate != itFiringRateEnd;
         ++itFiringRate) {
        for (std::map<EventType_T, unsigned int>::const_iterator it
             = (*itFiringRate).second.begin(),
             itEnd = (*itFiringRate).second.end();
             it != itEnd;
             ++it) {
            firingRate += (*it).second;
        }
    }

    return firingRate;
}

unsigned int N2D2::Monitor::getTotalFiringRate(EventType_T type) const
{
    unsigned int firingRate = 0;

    for (std::map
         <NodeId_T, std::map<EventType_T, unsigned int> >::const_iterator
             itFiringRate = mFiringRate.begin(),
             itFiringRateEnd = mFiringRate.end();
         itFiringRate != itFiringRateEnd;
         ++itFiringRate) {
        std::map<EventType_T, unsigned int>::const_iterator it
            = (*itFiringRate).second.find(type);

        if (it != (*itFiringRate).second.end())
            firingRate += (*it).second;
    }

    return firingRate;
}

double N2D2::Monitor::getSuccessRate(unsigned int avgWindow) const
{
    const unsigned int size = mSuccess.size();

    if (size > 0) {
        return (avgWindow > 0 && size > avgWindow)
                   ? std::accumulate(mSuccess.end() - avgWindow,
                                     mSuccess.end(),
                                     0.0) / avgWindow
                   : std::accumulate(mSuccess.begin(), mSuccess.end(), 0.0)
                     / size;
    } else
        return 0.0;
}

void N2D2::Monitor::logSuccessRate(const std::string& fileName,
                                   unsigned int avgWindow,
                                   bool plot) const
{
    logDataRate(mSuccess, fileName, avgWindow, plot);
}

void N2D2::Monitor::logFiringRate(const std::string& fileName, bool plot) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create firing rate log file: "
                                 + fileName);

    unsigned int totalActivity = 0;

    for (std::map
         <NodeId_T, std::map<EventType_T, unsigned int> >::const_iterator it
         = mFiringRate.begin(),
         itEnd = mFiringRate.end();
         it != itEnd;
         ++it) {
        data << (*it).first;

        for (std::set<EventType_T>::const_iterator itType = mEventTypes.begin(),
                                                   itTypeEnd
                                                   = mEventTypes.end();
             itType != itTypeEnd;
             ++itType) {
            std::map<EventType_T, unsigned int>::const_iterator itNodeType
                = (*it).second.find(*itType);

            if (itNodeType != (*it).second.end()) {
                totalActivity += (*itNodeType).second;
                data << " " << (*itNodeType).second;
            } else
                data << " 0";
        }

        data << "\n";
    }

    data.close();

    if (mFiringRate.empty())
        std::cout << "Notice: no firing rate recorded." << std::endl;
    else if (plot) {
        NodeId_T xmin = mNodes[0]->getId();
        NodeId_T xmax = mNodes[0]->getId();

        for (std::vector<Node*>::const_iterator it = mNodes.begin(),
                                                itEnd = mNodes.end();
             it != itEnd;
             ++it) {
            xmin = std::min(xmin, (*it)->getId());
            xmax = std::max(xmax, (*it)->getId());
        }

        std::ostringstream label;
        label << "\"Total: " << totalActivity << "\"";
        label << " at graph 0.5, graph 0.95 front";

        const unsigned int nbEventTypes = mEventTypes.size();

        Gnuplot gnuplot;
        std::stringstream cmdStr;
        cmdStr << "n = " << (double)nbEventTypes;
        gnuplot << cmdStr.str();
        gnuplot << "box_width = 0.75";
        gnuplot << "gap_width = 0.1";
        gnuplot << "total_width = (gap_width + box_width)";
        gnuplot << "d_width = total_width/n";
        gnuplot << "offset = -total_width/2.0 + d_width/2.0";
        gnuplot.set("style data boxes").set("style fill solid noborder");
        gnuplot.set("boxwidth", "box_width/n relative");
        gnuplot.setXrange(xmin - 0.5, xmax + 0.5);
        gnuplot.set("yrange [0:]");
        gnuplot.setYlabel("Number of activations");
        gnuplot.setXlabel("Node ID");

        if (mFiringRate.size() < 100) {
            gnuplot.set("grid");
            gnuplot.set("xtics", "1 rotate by 90");
        }

        gnuplot.set("label", label.str());
        gnuplot.saveToFile(fileName);

        if (nbEventTypes > 1) {
            unsigned int i = 1;
            std::ostringstream plotCmd;

            for (std::set<EventType_T>::const_iterator it = mEventTypes.begin(),
                                                       itEnd
                                                       = mEventTypes.end();
                 it != itEnd;
                 ++it) {
                plotCmd << "using ($1+offset+" << (i - 1)
                        << "*d_width):" << (1 + i) << " title \"" << (*it)
                        << "\"";

                if ((i++) < nbEventTypes)
                    plotCmd << ", \"\" ";
            }

            gnuplot.plot(fileName, plotCmd.str());
        } else
            gnuplot.plot(fileName, "using 1:2 notitle");
    }
}

void N2D2::Monitor::logActivity(const std::string& fileName,
                                bool plot,
                                Time_T start,
                                Time_T stop) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create activity log file: "
                                 + fileName);

    // Use the full double precision to keep accuracy even on small scales
    data.precision(std::numeric_limits<double>::digits10 + 1);

    for (std::map<NodeId_T, NodeEvents_T>::const_iterator it
         = mActivity.begin(),
         itEnd = mActivity.end();
         it != itEnd;
         ++it) {
        for (NodeEvents_T::const_iterator itTime = (*it).second.begin(),
                                          itTimeEnd = (*it).second.end();
             itTime != itTimeEnd;
             ++itTime) {
            data << (*it).first << " " << (*itTime).first / ((double)TimeS)
                 << " " << (*itTime).second << "\n";
        }

        data << "\n\n";
    }

    data.close();

    if (mActivity.empty())
        std::cout << "Notice: no activity recorded." << std::endl;
    else if (plot) {
        const double xrange = (mLastEvent - mFirstEvent) / ((double)TimeS);
        NodeId_T ymin = mNodes[0]->getId();
        NodeId_T ymax = mNodes[0]->getId();

        for (std::vector<Node*>::const_iterator it = mNodes.begin(),
                                                itEnd = mNodes.end();
             it != itEnd;
             ++it) {
            ymin = std::min(ymin, (*it)->getId());
            ymax = std::max(ymax, (*it)->getId());
        }

        const double xmin
            = (start > 0)
                  ? start / ((double)TimeS)
                  : std::max(0.0,
                             (mFirstEvent / ((double)TimeS) - 0.1 * xrange));
        const double xmax = (stop > 0)
                                ? stop / ((double)TimeS)
                                : mLastEvent / ((double)TimeS) + 0.1 * xrange;
        const unsigned int nbEventTypes = mEventTypes.size();

        Gnuplot gnuplot;
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.setXrange(xmin, xmax);
        gnuplot.setYrange(ymin, ymax + 1);
        gnuplot.setXlabel("Time (s)");
        gnuplot.saveToFile(fileName);

        if (ymax - ymin < 100) {
            gnuplot.set("grid");
            gnuplot.set("ytics", 1);
        }

        if (nbEventTypes > 1) {
            unsigned int i = 1;
            std::ostringstream plotCmd;

            for (std::set<EventType_T>::const_iterator it = mEventTypes.begin(),
                                                       itEnd
                                                       = mEventTypes.end();
                 it != itEnd;
                 ++it) {
                plotCmd << "using 2:($3==" << (*it) << " ? $1 : 1/0):($1+"
                        << (0.8 * (nbEventTypes - i + 1) / nbEventTypes)
                        << "):($1)"
                        << " title \"" << (*it) << "\" with yerrorbars lt "
                        << i;

                if ((i++) < nbEventTypes)
                    plotCmd << ", \"\" ";
            }

            gnuplot.plot(fileName, plotCmd.str());
        } else
            gnuplot.plot(fileName,
                         "using 2:1:($1+0.8):($1) notitle with yerrorbars");
    }
}

void N2D2::Monitor::clearAll()
{
    mActivity.clear();
    mFirstEvent = 0;
    mValidFirstEvent = false;
    mFiringRate.clear();
    mSuccess.clear();
    mEventTypes.clear();
}

void N2D2::Monitor::clearActivity()
{
    mActivity.clear();
    mFirstEvent = 0;
    mValidFirstEvent = false;
}

void N2D2::Monitor::clearFiringRate()
{
    mFiringRate.clear();
}

void N2D2::Monitor::clearSuccess()
{
    mSuccess.clear();
}
