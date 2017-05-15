/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "DeepNet.hpp"

N2D2::DeepNet::RangeStats::RangeStats()
    : minVal(0.0), maxVal(0.0), moments(3, 0.0)
{
    // ctor
}

double N2D2::DeepNet::RangeStats::mean() const
{
    return (moments[1] / moments[0]);
}

double N2D2::DeepNet::RangeStats::stdDev() const
{
    assert(moments.size() > 2);
    assert(moments[0] > 0.0);

    const double mean = (moments[1] / moments[0]);
    const double meanSquare = (moments[2] / moments[0]);
    return std::sqrt(meanSquare - mean * mean);
}

void N2D2::DeepNet::RangeStats::operator()(double value)
{
    if (moments[0] > 0) {
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    } else {
        minVal = value;
        maxVal = value;
    }

    double inc = 1.0;

    for (std::vector<double>::iterator it = moments.begin(),
                                       itEnd = moments.end();
         it != itEnd;
         ++it) {
        (*it) += inc;
        inc *= value;
    }
}

N2D2::DeepNet::DeepNet(Network& net)
    : mNet(net),
      mLayers(1, std::vector<std::string>(1, "env")),
      mSignalsDiscretization(0),
      mFreeParametersDiscretization(0),
      mFreeParametersDiscretized(false),
      mStreamIdx(0),
      mStreamTestIdx(0)
{
    // ctor
}

void N2D2::DeepNet::addCell(const std::shared_ptr<Cell>& cell,
                            const std::vector<std::shared_ptr<Cell> >& parents)
{
    unsigned int cellOrder = 0;
    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itBegin = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::shared_ptr<Cell> >::const_iterator itParent
             = parents.begin(),
             itParentEnd = parents.end();
             itParent != itParentEnd;
             ++itParent) {
            if (*itParent) {
                std::vector<std::string>::const_iterator itCell = std::find(
                    (*it).begin(), (*it).end(), (*itParent)->getName());

                if (itCell != (*it).end())
                    cellOrder
                        = std::max(cellOrder, (unsigned int)(it - itBegin));
            }
        }
    }

    if (cellOrder + 1 >= mLayers.size())
        mLayers.resize(cellOrder + 2);

    mLayers[cellOrder + 1].push_back(cell->getName());

    for (std::vector<std::shared_ptr<Cell> >::const_iterator itParent
         = parents.begin(),
         itParentEnd = parents.end();
         itParent != itParentEnd;
         ++itParent) {
        if (*itParent)
            mParentLayers.insert(
                std::make_pair(cell->getName(), (*itParent)->getName()));
        else
            mParentLayers.insert(std::make_pair(cell->getName(), "env"));
    }

    mCells.insert(std::make_pair(cell->getName(), cell));
}

void N2D2::DeepNet::addTarget(const std::shared_ptr<Target>& target)
{
    std::shared_ptr<Cell> cell = target->getCell();

    // Check for wrong parameters
    if (cell->getType() == SoftmaxCell::Type
        && (target->getDefaultValue() != 0.0 || target->getTargetValue()
                                                != 1.0)) {
        std::cout << Utils::cwarning
                  << "Warning: default and target values should be 0 and 1 "
                     "respectively "
                     "with Softmax output layer: " << cell->getName() << "!"
                  << Utils::cdef << std::endl;
    } else if (cell->getType() == FcCell::Type
               && cell->isParameter("DropConnect")
               && cell->getParameter<double>("DropConnect") < 1) {
        std::cout << Utils::cwarning
                  << "Warning: using DropConnect on target layer "
                  << cell->getName() << "!" << Utils::cdef << std::endl;
    }

    std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast
        <Cell_Frame_Top>(cell);

    if (cellFrame && cellFrame->getActivation()
        && cellFrame->getActivation()->getType() == std::string("Rectifier")
        && target->getDefaultValue() < 0.0) {
        std::cout << Utils::cwarning << "Warning: using negative default value "
                                        "with rectifier target cell "
                  << cell->getName() << "!" << Utils::cdef << std::endl;
    }

    mTargets.push_back(target);
}

void N2D2::DeepNet::addMonitor(const std::string& name,
                               const std::shared_ptr<Monitor>& monitor)
{
    const std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
        = mMonitors.find(name);

    if (it != mMonitors.end())
        throw std::runtime_error("Monitor for layer " + name
                                 + " already exists");

    mMonitors.insert(std::make_pair(name, monitor));
}

std::vector<std::pair<std::string, unsigned int> >
N2D2::DeepNet::update(bool log, Time_T start, Time_T stop, bool update)
{
    std::vector<std::pair<std::string, unsigned int> > activity;

    // Update monitors
    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            std::map<std::string, std::shared_ptr<Monitor> >::const_iterator
            itMonitor = mMonitors.find(*itCell);

            if (itMonitor == mMonitors.end())
                continue;

            (*itMonitor).second->update(update);
            activity.push_back(std::make_pair(
                (*itCell), (*itMonitor).second->getTotalActivity()));
        }
    }

    if (log) {
        for (std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
             = mMonitors.begin(),
             itEnd = mMonitors.end();
             it != itEnd;
             ++it) {
            (*it).second->logActivity(
                "activity_" + (*it).first + ".dat", true, start, stop);
            (*it).second->logFiringRate("firing_rate_" + (*it).first + ".dat",
                                        true);

            (*it).second->clearActivity();
        }

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
             = mCells.begin(),
             itEnd = mCells.end();
             it != itEnd;
             ++it) {
            (*it).second->logFreeParameters((*it).first);
            if ((*it).second->getType() == ConvCell::Type) {
                std::dynamic_pointer_cast<ConvCell_Spike>((*it).second)
                    ->reconstructActivities((*it).first, start, stop);
            }
            /*
            else if ((*it).second->getType() == LcCell::Type) {std::dynamic_pointer_cast<LcCell_Spike>((*it).second)->reconstructActivities((*it).first,
            start, stop);
            }
            */
        }
    }

    return activity;
}

void N2D2::DeepNet::saveNetworkParameters() const
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->saveParameters((*it).first + ".cfg");
}

void N2D2::DeepNet::loadNetworkParameters()
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->loadParameters((*it).first + ".cfg");
}

void N2D2::DeepNet::exportNetworkFreeParameters(const std::string
                                                & dirName) const
{
    Utils::createDirectories(dirName);

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        (*it).second->exportFreeParameters(dirName + "/" + (*it).first
                                           + ".syntxt");
        (*it).second->logFreeParametersDistrib(dirName + "/" + (*it).first
                                               + ".distrib.dat");
    }
}

void N2D2::DeepNet::exportNetworkSolverParameters(const std::string
                                                  & dirName) const
{
    Utils::createDirectories(dirName);

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        (*it).second->exportFreeParameters(dirName + "/" + (*it).first
                                           + ".syntxt");
        //(*it).second->logFreeParametersDistrib(dirName + "/" + (*it).first +
        //".distrib.dat");
    }
}

void N2D2::DeepNet::importNetworkFreeParameters(const std::string& dirName,
                                                bool ignoreNotExists)
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->importFreeParameters(dirName + "/" + (*it).first
                                           + ".syntxt", ignoreNotExists);
}

std::shared_ptr<N2D2::Monitor> N2D2::DeepNet::getMonitor(const std::string
                                                         & name) const
{
    const std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
        = mMonitors.find(name);

    if (it == mMonitors.end())
        throw std::runtime_error("Monitor for layer " + name
                                 + " does not exist");

    return (*it).second;
}

std::vector<std::shared_ptr<N2D2::Cell> >
N2D2::DeepNet::getParentCells(const std::string& name) const
{
    std::vector<std::shared_ptr<Cell> > parentCells;
    std::pair<std::multimap<std::string, std::string>::const_iterator,
              std::multimap<std::string, std::string>::const_iterator> parents
        = mParentLayers.equal_range(name);

    for (std::multimap<std::string, std::string>::const_iterator itParent
         = parents.first;
         itParent != parents.second;
         ++itParent) {
        if ((*itParent).second == "env")
            parentCells.push_back(std::shared_ptr<Cell>());
        else
            parentCells.push_back((*mCells.find((*itParent).second)).second);
    }

    return parentCells;
}

void N2D2::DeepNet::getStats(Cell::Stats& stats) const
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->getStats(stats);
}

void N2D2::DeepNet::clearAll()
{
    for (std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
         = mMonitors.begin(),
         itEnd = mMonitors.end();
         it != itEnd;
         ++it) {
        (*it).second->clearAll();
    }
}

void N2D2::DeepNet::clearActivity()
{
    for (std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
         = mMonitors.begin(),
         itEnd = mMonitors.end();
         it != itEnd;
         ++it) {
        (*it).second->clearActivity();
    }
}

void N2D2::DeepNet::clearFiringRate()
{
    for (std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
         = mMonitors.begin(),
         itEnd = mMonitors.end();
         it != itEnd;
         ++it) {
        (*it).second->clearFiringRate();
    }
}

void N2D2::DeepNet::clearSuccess()
{
    for (std::map<std::string, std::shared_ptr<Monitor> >::const_iterator it
         = mMonitors.begin(),
         itEnd = mMonitors.end();
         it != itEnd;
         ++it) {
        (*it).second->clearSuccess();
    }
}

void N2D2::DeepNet::checkGradient(double epsilon, double maxError)
{
    for (unsigned int l = 1, nbLayers = mLayers.size(); l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            std::dynamic_pointer_cast<Cell_Frame_Top>(mCells[(*itCell)])
                ->checkGradient(epsilon, maxError);
        }
    }
}

void N2D2::DeepNet::initialize()
{
    for (unsigned int l = 1, nbLayers = mLayers.size(); l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            mCells[(*itCell)]->initialize();
        }
    }
}

void N2D2::DeepNet::spikeCodingCompare(const std::string& dirName,
                                       unsigned int idx) const
{
    std::shared_ptr<Environment> env = std::dynamic_pointer_cast
        <Environment>(mStimuliProvider);

    if (!env)
        throw std::runtime_error(
            "DeepNet::spikeCodingCompare(): require an Environment.");

    Utils::createDirectories(dirName);

    // Environment
    const std::string fileName = dirName + "/env.dat";
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "Could not save spike coding compare data file.");

    const unsigned int envSizeX = env->getSizeX();
    const unsigned int envSizeY = env->getSizeY();

    env->readStimulus(Database::Test, idx);
    const Tensor2d<Float_T> frame = env->getData(0);

    Float_T minVal = frame(0);
    Float_T maxVal = frame(0);

    Float_T avgSignal = 0.0;
    int avgActivity = 0;

    for (unsigned int y = 0; y < envSizeY; ++y) {
        for (unsigned int x = 0; x < envSizeX; ++x) {
            const int activity
                = (int)env->getNode(0, x, y)->getActivity(0, 0, 0)
                  - (int)env->getNode(0, x, y)->getActivity(0, 0, 1);

            minVal = std::min(minVal, frame(x, y));
            maxVal = std::max(maxVal, frame(x, y));

            avgSignal += frame(x, y);
            avgActivity += activity;

            data << x << " " << y << " " << frame(x, y) << " " << activity
                 << "\n";
        }
    }

    data.close();

    const double scalingRatio = avgActivity / avgSignal;

    // Plot results
    Gnuplot gnuplot;

    std::stringstream scalingStr;
    scalingStr << "scalingRatio=" << scalingRatio;

    gnuplot << scalingStr.str();
    gnuplot.set("key off");
    gnuplot.setXrange(-0.5, envSizeX - 0.5);
    gnuplot.setYrange(envSizeY - 0.5, -0.5);

    std::stringstream cbRangeStr, paletteStr;
    cbRangeStr << "cbrange [";
    paletteStr << "palette defined (";

    if (minVal < -1.0) {
        cbRangeStr << minVal;
        paletteStr << minVal << " \"blue\", -1 \"cyan\", ";
    } else if (minVal < 0.0) {
        cbRangeStr << -1.0;
        paletteStr << "-1 \"cyan\", ";
    } else
        cbRangeStr << 0.0;

    cbRangeStr << ":";
    paletteStr << "0 \"black\"";

    if (maxVal > 1.0) {
        cbRangeStr << maxVal;
        paletteStr << ", 1 \"white\", " << maxVal << " \"red\"";
    } else if (maxVal > 0.0 || !(minVal < 0)) {
        cbRangeStr << 1.0;
        paletteStr << ", 1 \"white\"";
    } else
        cbRangeStr << 0.0;

    cbRangeStr << "]";
    paletteStr << ")";

    gnuplot.set(paletteStr.str());
    gnuplot.set(cbRangeStr.str());

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
                 "using 1:2:3 with image, \"\" using 1:2:(abs($4) < "
                 "1 ? \"\" : sprintf(\"%d\",$4)) with labels");

    gnuplot.saveToFile(fileName, "-diff");
    gnuplot.plot(fileName,
                 "using 1:2:3 with image,"
                 " \"\" using 1:2:(($3*scalingRatio-$4) < 1 ? \"\" : "
                 "sprintf(\"%d\",$3*scalingRatio-$4)) with labels");

    // Layers
    for (unsigned int l = 1, nbLayers = mLayers.size(); l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            const std::map
                <std::string, std::shared_ptr<Cell> >::const_iterator it
                = mCells.find(*itCell);
            std::shared_ptr<Cell_Spike> cellSpike = std::dynamic_pointer_cast
                <Cell_Spike>((*it).second);

            if (!cellSpike)
                throw std::runtime_error("DeepNet::spikeCodingCompare(): only "
                                         "works with Spike models");

            cellSpike->spikeCodingCompare(dirName + "/" + (*it).first + ".dat");
        }
    }
}

void
N2D2::DeepNet::normalizeOutputsRange(const std::map
                                     <std::string, RangeStats>& outputsRange,
                                     double normFactor,
                                     double useMean)
{
    // const std::map<std::string, RangeStats>::const_iterator itEnvRange =
    // outputsRange.find(*(*mLayers.begin()).begin());

    double prevScalingFactor = 1.0;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1,
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        bool nextIsPool = false;

        if (it + 1 != itEnd) {
            std::shared_ptr<PoolCell> poolCell = std::dynamic_pointer_cast
                <PoolCell>((*mCells.find(*(*(it + 1)).begin())).second);

            if (poolCell) {
                std::cout << "Poolcell following" << std::endl;
                nextIsPool = true;
            }
        }

        double scalingFactor = 0.0;

        if (useMean) {
            double nbElements = 0.0;

            for (std::vector<std::string>::const_iterator itCell
                 = (nextIsPool) ? (*(it + 1)).begin() : (*it).begin(),
                 itCellEnd = (nextIsPool) ? (*(it + 1)).end() : (*it).end();
                 itCell != itCellEnd;
                 ++itCell) {
                const std::map<std::string, RangeStats>::const_iterator itRange
                    = outputsRange.find(*itCell);
                scalingFactor += (*itRange).second.moments[0]
                                 * (*itRange).second.mean();
                nbElements += (*itRange).second.moments[0];
            }

            scalingFactor /= nbElements;
        } else {
            for (std::vector<std::string>::const_iterator itCell
                 = (nextIsPool) ? (*(it + 1)).begin() : (*it).begin(),
                 itCellEnd = (nextIsPool) ? (*(it + 1)).end() : (*it).end();
                 itCell != itCellEnd;
                 ++itCell) {
                const std::map<std::string, RangeStats>::const_iterator itRange
                    = outputsRange.find(*itCell);
                scalingFactor
                    = std::max(scalingFactor, (*itRange).second.maxVal);
            }
        }

        scalingFactor /= normFactor;

        const double appliedFactor = scalingFactor / prevScalingFactor;
        bool applied = false;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame && cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                   == std::string("Rectifier")) {
                cell->processFreeParameters(std::bind(std::divides<double>(),
                                                      std::placeholders::_1,
                                                      appliedFactor));
                applied = true;
            }
        }

        if (applied)
            prevScalingFactor = scalingFactor;
    }
}

void N2D2::DeepNet::logOutputs(const std::string& dirName,
                               unsigned int batchPos) const
{
    Utils::createDirectories(dirName);

    const unsigned int nbLayers = mLayers.size();

    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            const Tensor3d<Float_T> outputs
                = std::dynamic_pointer_cast
                  <Cell_Frame_Top>(cell)->getOutputs()[batchPos];

            StimuliProvider::logData(dirName + "/" + (*itCell) + ".dat",
                                     outputs);
        }
    }
}

void N2D2::DeepNet::logDiffInputs(const std::string& dirName,
                                  unsigned int batchPos) const
{
    Utils::createDirectories(dirName);

    const unsigned int nbLayers = mLayers.size();

    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            const Tensor3d<Float_T> diffInputs
                = std::dynamic_pointer_cast
                  <Cell_Frame_Top>(cell)->getDiffInputs()[batchPos];

            StimuliProvider::logData(dirName + "/" + (*itCell) + ".dat",
                                     diffInputs);
        }
    }
}

void N2D2::DeepNet::logFreeParameters(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        (*it).second->logFreeParameters(dirName + "/" + (*it).first);
    }
}

void N2D2::DeepNet::logStats(const std::string& dirName) const
{
    Utils::createDirectories(dirName);
    const std::string statsFileName = dirName + "/stats.dat";
    const std::string logFileName = dirName + "/stats.log";

    // Cells stats
    std::ofstream statsData(statsFileName.c_str());

    if (!statsData.good())
        throw std::runtime_error("Could not open stats file: " + statsFileName);

    statsData << "Cell Memory Computing\n";

    Cell::Stats globalStats;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1,
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;

            Cell::Stats cellStats;
            cell->getStats(cellStats);
            cell->getStats(globalStats);

            statsData << (*itCell) << " " << cellStats.nbSynapses << " "
                      << cellStats.nbConnections << "\n";
        }
    }

    statsData.close();

    Gnuplot gnuplot;
    gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
               "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    gnuplot.set("style histogram cluster gap 1");
    gnuplot.set("style data histograms");
    gnuplot.set("style fill pattern 1.00 border");
    gnuplot.set("ytics nomirror");
    gnuplot.set("logscale y");
    gnuplot.set("y2range [0:*]");
    gnuplot.set("y2tics");
    gnuplot.setYlabel("Memory [int-8 bits] (bytes)");
    gnuplot.setY2label("Computing (MACs)");
    gnuplot.set("grid");
    gnuplot.set("ytics textcolor lt 1");
    gnuplot.set("ylabel textcolor lt 1");
    gnuplot.set("y2tics textcolor lt 2");
    gnuplot.set("y2label textcolor lt 2");
    gnuplot.set("format y \"%.0s%c\"");
    gnuplot.set("format y2 \"%.0s%c\"");
    gnuplot.unset("key");
    gnuplot.set("bmargin 4");
    gnuplot.saveToFile(statsFileName);
    gnuplot.plot(
        statsFileName,
        "using ($2):xticlabels(wrap(stringcolumn(1),5)) ti col,"
        " '' using 0:($2):(gprintf(\"%.2s%c\",$2)) ti col with labels offset "
        "char -3,1 textcolor lt 1,"
        " '' using ($3):xticlabels(wrap(stringcolumn(1),5)) ti col axes x1y2,"
        " '' using ($0+0.2):($3):(gprintf(\"%.2s%c\",$3)) ti col axes x1y2 "
        "with labels offset char 1,1 textcolor lt 2");

    // Global stats
    std::ofstream logData(logFileName.c_str());

    if (!logData.good())
        throw std::runtime_error("Could not open log stats file: "
                                 + logFileName);

    logData << "[Global stats]\n"
               "Total number of neurons: " << globalStats.nbNeurons
            << "\n"
               "Total number of nodes: " << globalStats.nbNodes
            << "\n"
               "Total number of synapses: " << globalStats.nbSynapses
            << "\n"
               "Total number of virtual synapses: "
            << globalStats.nbVirtualSynapses
            << "\n"
               "Total number of connections: " << globalStats.nbConnections
            << "\n\n";

    const unsigned int inputData = mStimuliProvider->getSizeX()
                                   * mStimuliProvider->getSizeY()
                                   * mStimuliProvider->getNbChannels();
    const unsigned int freeParameters = globalStats.nbSynapses;

    logData << "[Memory]\n"
               "Input data (int-8 bits): " << inputData / 1024.0
            << " kBytes\n"
               "Input data (float-16 bits): " << 2.0 * inputData / 1024.0
            << " kBytes\n"
               "Input data (float-32 bits): " << 4.0 * inputData / 1024.0
            << " kBytes\n"
               "Free parameters (int-8 bits): " << freeParameters / 1024.0
            << " kBytes\n"
               "Free parameters (float-16 bits): " << 2.0 * freeParameters
                                                      / 1024.0
            << " kBytes\n"
               "Free parameters (float-32 bits): " << 4.0 * freeParameters
                                                      / 1024.0 << " kBytes\n\n";

    logData << "[Computing]\n"
               "MACS / input data: " << globalStats.nbConnections / 1.0e6
            << "M\n";
}

void N2D2::DeepNet::logSpikeStats(const std::string& dirName,
                                  unsigned int nbPatterns) const
{
    std::shared_ptr<Environment> env = std::dynamic_pointer_cast
        <Environment>(mStimuliProvider);

    if (!env)
        throw std::runtime_error(
            "DeepNet::logSpikeStats(): require an Environment.");

    Synapse::Stats globalStats;

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        const Synapse::Stats cellStats = std::dynamic_pointer_cast<Cell_Spike>(
            (*it).second)->logStats(dirName + "/" + (*it).first);

        globalStats.nbSynapses += cellStats.nbSynapses;
        globalStats.readEvents += cellStats.readEvents;
        globalStats.maxReadEvents
            = std::max(globalStats.maxReadEvents, cellStats.maxReadEvents);
    }

    std::ofstream globalData((dirName + ".log").c_str());

    if (!globalData.good())
        throw std::runtime_error("Could not create stats log file: "
                                 + (dirName + ".log"));

    globalData.imbue(Utils::locale);

    const unsigned int nbNodes = env->getNbNodes();

    Cell::Stats stats;
    getStats(stats);

    globalData << "[Global stats]\n"
                  "Patterns: " << nbPatterns << "\n"
                                                "Patterns channels: " << nbNodes
               << "\n"
                  "Neurons: " << stats.nbNeurons << "\n"
                                                    "Nodes: " << stats.nbNodes
               << "\n"
                  "Synapses: " << globalStats.nbSynapses
               << "\n"
                  "Virtual synapses: " << stats.nbVirtualSynapses
               << "\n"
                  "Connections: " << stats.nbConnections << "\n";

    if (mFreeParametersDiscretization > 0 && mFreeParametersDiscretized)
        globalData << "Free parameters discretization: "
                   << mFreeParametersDiscretization << " levels (+ sign bit)\n";

    globalData << "\n"
                  "[Neuron stats]\n";

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itBegin = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            const std::map
                <std::string, std::shared_ptr<Monitor> >::const_iterator
            itMonitor = mMonitors.find(*itCell);

            if (itMonitor == mMonitors.end())
                continue;

            const unsigned int firingRate
                = (*itMonitor).second->getTotalFiringRate();

            globalData << (*itCell) << " output events: " << firingRate << "\n"
                       << (*itCell) << " output events per pattern (average): "
                       << firingRate / (double)nbPatterns << "\n" << (*itCell)
                       << " output events per pattern per neuron (average): "
                       << firingRate / (double)nbPatterns
                          / (double)(*itMonitor).second->getNbNodes() << "\n";

            if (it
                != itBegin) { // Check that current cell is not the environment
                const std::shared_ptr<Cell> cell
                    = (*mCells.find(*itCell)).second;

                if (cell->isParameter("TerminateDelta"))
                    globalData << (*itCell) << " terminate delta: "
                               << cell->getParameter("TerminateDelta") << "\n";
            }
        }
    }

    globalData << "\n"
                  "[Synapse stats]\n"
                  "Read events: " << globalStats.readEvents
               << "\n"
                  "Read events per pattern (average): "
               << globalStats.readEvents / (double)nbPatterns
               << "\n"
                  "Read events per pattern channel (average): "
               << globalStats.readEvents / (double)nbPatterns / (double)nbNodes
               << "\n"
                  "Read events per synapse (average): "
               << globalStats.readEvents / (double)globalStats.nbSynapses
               << "\n"
                  "Max. read events per synapse: " << globalStats.maxReadEvents
               << "\n"
                  "Read events per virtual synapse (average): "
               << globalStats.readEvents / (double)stats.nbVirtualSynapses
               << "\n"
                  "Read events per synapse per pattern (average): "
               << globalStats.readEvents / (double)globalStats.nbSynapses
                  / (double)nbPatterns
               << "\n"
                  "Read events per virtual synapse per pattern (average): "
               << globalStats.readEvents / (double)stats.nbVirtualSynapses
                  / (double)nbPatterns
               << "\n"
                  "Max. read events per synapse per pattern (average): "
               << globalStats.maxReadEvents / (double)nbPatterns << std::endl;

    std::cout << "Network activity (events/connection/pattern): "
              << globalStats.readEvents / (double)stats.nbVirtualSynapses
                 / (double)nbPatterns << std::endl;
}

void N2D2::DeepNet::learn(std::vector<std::pair<std::string, double> >* timings)
{
    const unsigned int nbLayers = mLayers.size();

    std::chrono::high_resolution_clock::time_point time1, time2;

    if (timings != NULL)
        (*timings).clear();

    // Signal propagation
    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells[(*itCell)]);

            if (!cellFrame)
                throw std::runtime_error(
                    "DeepNet::learn(): learning requires Cell_Frame_Top cells");

            if (mSignalsDiscretization > 0)
                cellFrame->discretizeSignals(mSignalsDiscretization);

            //std::cout << "propagate " << mCells[(*itCell)]->getName()
            //    << std::endl;
            time1 = std::chrono::high_resolution_clock::now();
            cellFrame->propagate();

            if (timings != NULL) {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
                time2 = std::chrono::high_resolution_clock::now();
                (*timings).push_back(std::make_pair(
                    (*itCell) + "[prop]",
                    std::chrono::duration_cast
                    <std::chrono::duration<double> >(time2 - time1).count()));
            }
        }
    }

    // Set targets
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets)
    {
        //std::cout << "process " << (*itTargets)->getName() << std::endl;
        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->process(Database::Learn);

        if (timings != NULL) {
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
            time2 = std::chrono::high_resolution_clock::now();
            (*timings).push_back(std::make_pair(
                (*itTargets)->getCell()->getName() + "."
                + (*itTargets)->getType(),
                std::chrono::duration_cast
                <std::chrono::duration<double> >(time2 - time1).count()));
        }
    }

    // Error back-propagation
    for (unsigned int l = nbLayers - 1; l > 0; --l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell)
        {
            //std::cout << "back-propagate " << mCells[(*itCell)]->getName()
            //    << std::endl;
            time1 = std::chrono::high_resolution_clock::now();
            std::dynamic_pointer_cast
                <Cell_Frame_Top>(mCells[(*itCell)])->backPropagate();

            if (timings != NULL) {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
                time2 = std::chrono::high_resolution_clock::now();
                (*timings).push_back(std::make_pair(
                    (*itCell) + "[back-prop]",
                    std::chrono::duration_cast
                    <std::chrono::duration<double> >(time2 - time1).count()));
            }
        }
    }

    // Weights update
    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell)
        {
            //std::cout << "update " << mCells[(*itCell)]->getName()
            //    << std::endl;
            time1 = std::chrono::high_resolution_clock::now();
            std::dynamic_pointer_cast
                <Cell_Frame_Top>(mCells[(*itCell)])->update();

            if (timings != NULL) {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
                time2 = std::chrono::high_resolution_clock::now();
                (*timings).push_back(std::make_pair(
                    (*itCell) + "[update]",
                    std::chrono::duration_cast
                    <std::chrono::duration<double> >(time2 - time1).count()));
            }
        }
    }
}

void N2D2::DeepNet::test(Database::StimuliSet set,
                         std::vector<std::pair<std::string, double> >* timings)
{
    const unsigned int nbLayers = mLayers.size();

    if (mFreeParametersDiscretization > 0 && !mFreeParametersDiscretized) {
        const std::string dirName = "free_parameters_discretized";
        Utils::createDirectories(dirName);

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
             = mCells.begin(),
             itEnd = mCells.end();
             it != itEnd;
             ++it) {
            (*it).second->discretizeFreeParameters(
                mFreeParametersDiscretization);
            (*it).second->exportFreeParameters(dirName + "/" + (*it).first
                                               + ".syntxt");
            (*it).second->logFreeParametersDistrib(dirName + "/" + (*it).first
                                                   + ".distrib.dat");
        }

        mFreeParametersDiscretized = true;
    }

    std::chrono::high_resolution_clock::time_point time1, time2;

    if (timings != NULL)
        (*timings).clear();

    // Signal propagation
    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell) {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells[(*itCell)]);

            if (!cellFrame)
                throw std::runtime_error(
                    "DeepNet::test(): testing requires Cell_Frame_Top cells");

            if (mSignalsDiscretization > 0)
                cellFrame->discretizeSignals(mSignalsDiscretization);

            time1 = std::chrono::high_resolution_clock::now();
            cellFrame->propagate(true);

            if (timings != NULL) {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
                time2 = std::chrono::high_resolution_clock::now();
                (*timings).push_back(std::make_pair(
                    *itCell,
                    std::chrono::duration_cast
                    <std::chrono::duration<double> >(time2 - time1).count()));
            }
        }
    }

    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->process(set);

        if (timings != NULL) {
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
            time2 = std::chrono::high_resolution_clock::now();
            (*timings).push_back(std::make_pair(
                (*itTargets)->getCell()->getName() + "."
                + (*itTargets)->getType(),
                std::chrono::duration_cast
                <std::chrono::duration<double> >(time2 - time1).count()));
        }
    }
}

void N2D2::DeepNet::cTicks(Time_T start, Time_T stop, Time_T timestep)
{
    const unsigned int nbLayers = mLayers.size();

    std::shared_ptr<CEnvironment> cEnv = std::dynamic_pointer_cast
        <CEnvironment>(mStimuliProvider);

    if (!cEnv)
        throw std::runtime_error("DeepNet::cTicks(): require a CEnvironment.");

    for (Time_T t = start; t <= stop; t += timestep) {
        cEnv->tick(t, start, stop);

        for (unsigned int l = 1; l < nbLayers; ++l) {
            for (std::vector<std::string>::const_iterator itCell
                 = mLayers[l].begin(),
                 itCellEnd = mLayers[l].end();
                 itCell != itCellEnd;
                 ++itCell) {
                std::shared_ptr<Cell_CSpike_Top> cellCSpike
                    = std::dynamic_pointer_cast
                    <Cell_CSpike_Top>(mCells[(*itCell)]);

                if (!cellCSpike)
                    throw std::runtime_error(
                        "DeepNet::cTicks(): requires Cell_CSpike cells");

                if (cellCSpike->tick(t))
                    return;
            }
        }
    }
}

void N2D2::DeepNet::cTargetsProcess(Database::StimuliSet set)
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->process(set);
    }
}

void N2D2::DeepNet::cReset(Time_T timestamp)
{
    std::shared_ptr<CEnvironment> cEnv = std::dynamic_pointer_cast
        <CEnvironment>(mStimuliProvider);

    if (!cEnv)
        throw std::runtime_error("DeepNet::cReset(): require a CEnvironment.");

    cEnv->reset(timestamp);

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        std::shared_ptr<Cell_CSpike_Top> cellCSpike = std::dynamic_pointer_cast
            <Cell_CSpike_Top>((*it).second);

        if (!cellCSpike)
            throw std::runtime_error(
                "DeepNet::cReset(): requires Cell_CSpike cells");

        cellCSpike->reset(timestamp);
    }
}

void N2D2::DeepNet::log(const std::string& baseName,
                        Database::StimuliSet set) const
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->log(baseName, set);
    }
}

void N2D2::DeepNet::logLabelsMapping(const std::string& fileName) const
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->logLabelsMapping(fileName);
    }
}

void N2D2::DeepNet::logEstimatedLabels(const std::string& dirName) const
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->logEstimatedLabels(dirName);
    }
}

void N2D2::DeepNet::logLabelsLegend(const std::string& dirName) const
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->logLabelsLegend(dirName);
    }
}

void N2D2::DeepNet::logTimings(const std::string& fileName,
                               const std::vector
                               <std::pair<std::string, double> >& timings) const
{
    const double totalTime = std::accumulate(
        timings.begin(),
        timings.end(),
        std::pair<std::string, double>("", 0.0),
        Utils::PairOp
        <std::string, double, Utils::Left<std::string>, std::plus<double> >())
                                 .second;

    std::ofstream timingsData(fileName.c_str());

    if (!timingsData.good())
        throw std::runtime_error("Could not open timings file: " + fileName);

    for (std::vector<std::pair<std::string, double> >::const_iterator it
         = timings.begin(),
         itEnd = timings.end();
         it != itEnd;
         ++it) {
        timingsData << (*it).first << " " << (*it).second << " "
                    << (100.0 * (*it).second / totalTime) << "\n";
    }

    timingsData.close();

    std::stringstream outputStr;
    outputStr << "size " << (timings.size() * 50) << ",600 enhanced large";

    Gnuplot::setDefaultOutput("png", outputStr.str(), "png");

    Gnuplot gnuplot;
    gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
               "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    gnuplot.set("style histogram cluster gap 1");
    gnuplot.set("style data histograms");
    gnuplot.set("style fill pattern 1.00 border");
    gnuplot.set("ytics nomirror");
    gnuplot.set("logscale y");
    gnuplot.set("y2range [0:*]");
    gnuplot.set("y2tics");
    gnuplot.setYlabel("Timing (s)");
    gnuplot.setY2label("Relative timing (%)");
    gnuplot.set("grid");
    gnuplot.set("ytics textcolor lt 3");
    gnuplot.set("ylabel textcolor lt 3");
    gnuplot.set("y2tics textcolor lt 2");
    gnuplot.set("y2label textcolor lt 2");
    gnuplot.set("format y \"%.0s%c\"");
    gnuplot.set("bmargin 4");
    gnuplot.unset("key");
    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
                 "using ($2):xticlabels(wrap(stringcolumn(1),5)) lt 3,"
                 " '' using 0:($2):(gprintf(\"%.2s%c\",$2)) with labels offset "
                 "char -3,1 textcolor lt 3,"
                 " '' using ($3):xticlabels(wrap(stringcolumn(1),5)) axes x1y2,"
                 " '' using ($0+0.2):($3):(gprintf(\"%.2f\",$3)) axes x1y2 "
                 "with labels offset char 1,1 textcolor lt 2");

    Gnuplot::setDefaultOutput();
}

void
N2D2::DeepNet::reportOutputsRange(std::map
                                  <std::string, RangeStats>& outputsRange) const
{
    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            const Tensor4d<Float_T> outputs
                = (mCells.find(*itCell) != mCells.end())
                      ? std::dynamic_pointer_cast<Cell_Frame_Top>(
                            (*mCells.find(*itCell)).second)->getOutputs()
                      : mStimuliProvider->getData();

            bool newInsert;
            std::map<std::string, RangeStats>::iterator itRange;
            std::tie(itRange, newInsert)
                = outputsRange.insert(std::make_pair(*itCell, RangeStats()));

            (*itRange).second = std::for_each(
                outputs.begin(), outputs.end(), (*itRange).second);
        }
    }
}

void
N2D2::DeepNet::logOutputsRange(const std::string& fileName,
                               const std::map
                               <std::string, RangeStats>& outputsRange) const
{
    std::ofstream rangeData(fileName.c_str());

    if (!rangeData.good())
        throw std::runtime_error("Could not open range file: " + fileName);

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {
            const std::map<std::string, RangeStats>::const_iterator itRange
                = outputsRange.find(*itCell);

            rangeData << (*itRange).first << " " << (*itRange).second.minVal
                      << " " << (*itRange).second.maxVal << " "
                      << (*itRange).second.mean() << " "
                      << (*itRange).second.stdDev() << "\n";
        }
    }

    rangeData.close();

    Gnuplot gnuplot;
    gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
               "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    gnuplot.setYlabel("Range (min, mean, max) with std. dev.");
    gnuplot.setXrange(-0.5, outputsRange.size() - 0.5);
    gnuplot.set("grid");
    gnuplot.set("bmargin 4");
    gnuplot.set("boxwidth 0.2");
    gnuplot << "min(x,y) = (x < y) ? x : y";
    gnuplot << "max(x,y) = (x > y) ? x : y";
    gnuplot.unset("key");
    gnuplot.saveToFile(fileName);
    gnuplot.plot(
        fileName,
        "using ($4):xticlabels(wrap(stringcolumn(1),5)) lt 1,"
        " '' using 0:(max($2,$4-$5)):2:3:(min($3,$4+$5)) with candlesticks lt "
        "1 lw 2 whiskerbars,"
        " '' using 0:($3):($3) with labels offset char 0,1 textcolor lt 1,"
        " '' using 0:($2):($2) with labels offset char 0,-1 textcolor lt 3,"
        " '' using 0:($4):($4) with labels offset char 7,0 textcolor lt -1,"
        " '' using 0:4:4:4:4 with candlesticks lt -1 lw 2 notitle");
}

void N2D2::DeepNet::clear(Database::StimuliSet set)
{
    for (std::vector<std::shared_ptr<Target> >::iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->clear(set);
    }
}
