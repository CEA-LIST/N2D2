/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#include "CEnvironment.hpp"
#include "CMonitor.hpp"
#include "DeepNet.hpp"
#include "Environment.hpp"
#include "Monitor.hpp"
#include "NodeEnv.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/DeconvCell.hpp"
#include "Cell/ConvCell_Spike.hpp"
#include "Cell/DropoutCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "utils/Utils.hpp"
#include "Solver/Solver.hpp"

N2D2::DeepNet::DeepNet(Network& net)
    : mName(this, "Name", ""),
      mSignalsDiscretization(this, "SignalsDiscretization", 0U),
      mFreeParametersDiscretization(this, "FreeParametersDiscretization", 0U),
      mNet(net),
      mLayers(1, std::vector<std::string>(1, "env")),
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

void N2D2::DeepNet::removeCell(const std::shared_ptr<Cell>& cell,
                               bool reconnect)
{
    const std::string name = cell->getName();

    std::vector<std::string> parents;
    std::vector<std::string> childs;

    for (std::multimap<std::string, std::string>::iterator
         itParentLayers = mParentLayers.begin();
         itParentLayers != mParentLayers.end(); )
    {
        if ((*itParentLayers).first == name
            || (*itParentLayers).second == name)
        {
            if ((*itParentLayers).first == name)
                parents.push_back((*itParentLayers).second);
            else
                childs.push_back((*itParentLayers).first);

            // Remove the cell from mParentLayers
            itParentLayers
                = mParentLayers.erase(itParentLayers);
        }
        else
            ++itParentLayers;
    }

    if (reconnect) {
        // Connect directly childs to parents

        /* Example with single branch:
           A -> X -> B                =>    A -> B
           mParentLayers:
           ("X", "A"), ("B", "X")     =>     ("B", "A")

           Examples with multiples branches:
           (1)  A \                                         A \
                   -> X -> C                      =>           -> C
                B /                                         B /
           mParentLayers:
           ("X", "A"), ("X", "B"), ("C", "X")     =>    ("C", "A"), ("C", "B")

           (2)  A \       / C                               A \-/ C
                   -> X ->                        =>           X
                B /       \ D                               B /-\ D
           mParentLayers:
           ("X", "A"), ("X", "B")                 =>    ("C", "A"), ("C", "B")
           ("C", "X"), ("D", "X")                       ("D", "A"), ("D", "B")
        */

        std::shared_ptr<Cell_Frame_Top> cellTop =
            std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

        for (std::vector<std::string>::const_iterator itChild = childs.begin(),
             itChildEnd = childs.end(); itChild != itChildEnd; ++itChild)
        {
            std::shared_ptr<Cell_Frame_Top> childCellTop =
                std::dynamic_pointer_cast<Cell_Frame_Top>(mCells[*itChild]);

            for (std::vector<std::string>::const_iterator itParent
                 = parents.begin(), itParentEnd = parents.end();
                 itParent != itParentEnd; ++itParent)
            {
                std::shared_ptr<Cell_Frame_Top> parentCellTop =
                   std::dynamic_pointer_cast<Cell_Frame_Top>(mCells[*itParent]);

                if (cellTop && childCellTop && parentCellTop) {
                    childCellTop->replaceInput(
                        cellTop->getOutputs(),
                        parentCellTop->getOutputs(),
                        parentCellTop->getDiffInputs());
                }

                mParentLayers.insert(std::make_pair(*itChild, *itParent));
            }
        }
    }

    mCells.erase(name);

    for (unsigned int l = 1; l < mLayers.size(); ) {
        mLayers[l].erase(std::remove(mLayers[l].begin(),
                                     mLayers[l].end(),
                                     name),
                         mLayers[l].end());

        if (mLayers[l].empty())
            mLayers.erase(mLayers.begin() + l);
        else
            ++l;
    }
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

void N2D2::DeepNet::addCMonitor(const std::string& name,
                                const std::shared_ptr<CMonitor>& monitor)
{
    const std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
        = mCMonitors.find(name);

    if (it != mCMonitors.end())
        throw std::runtime_error("CMonitor for layer " + name
                                 + " already exists");

    mCMonitors.insert(std::make_pair(name, monitor));
}

std::vector<std::pair<std::string, long long int> >
N2D2::DeepNet::update(bool log, Time_T start, Time_T stop, bool update)
{
    std::vector<std::pair<std::string, long long int> > activity;

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
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell) {

            std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator
            itMonitor = mCMonitors.find(*itCell);

            if (itMonitor ==  mCMonitors.end())
                continue;

            (*itMonitor).second->update(start, stop);

            //activity.push_back(std::make_pair(
            //    (*itCell), (*itMonitor).second->getTotalBatchExampleFiringRate()));
            activity.push_back(std::make_pair(
                (*itCell), (*itMonitor).second->getTotalBatchOutputsActivity()));

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

        for (std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
             = mCMonitors.begin(),
             itEnd = mCMonitors.end();
             it != itEnd;
             ++it) {
            (*it).second->logActivity("activity_batchElem_0_" + (*it).first + ".dat", 0, true, start, stop);
            //(*it).second->logFiringRate("firing_rate_" + (*it).first + ".dat",
            //                            true, start, stop);
            (*it).second->logFiringRate("firing_rate_" + (*it).first + ".dat",
                                        true, 0, 0);

        }

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
             = mCells.begin(),
             itEnd = mCells.end();
             it != itEnd;
             ++it) {

            //(*it).second->logFreeParameters((*it).first);

            if ((*it).second->getType() == ConvCell::Type) {
                std::shared_ptr<ConvCell_Spike> cellSpike =
                    std::dynamic_pointer_cast<ConvCell_Spike>((*it).second);
                    if (cellSpike){
                        cellSpike->reconstructActivities((*it).first, start, stop);
                    }
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

void N2D2::DeepNet::save(const std::string& dirName) const
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->save(dirName + "/" + (*it).first);
}

void N2D2::DeepNet::load(const std::string& dirName)
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it)
        (*it).second->load(dirName + "/" + (*it).first);
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
    std::cout << "Importing weights from directory '" << dirName << "'." << std::endl;
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it){
        (*it).second->importFreeParameters(dirName + "/" + (*it).first
                                           + ".syntxt", ignoreNotExists);
    }
}


/* This implementation loads only those weights from
the directory which are explicitly specified */
void N2D2::DeepNet::importNetworkFreeParameters(const std::string& dirName,
                                                const std::string& weightName) {
    bool weightsFound = false;
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
    it = mCells.begin(), itEnd = mCells.end(); it != itEnd; ++it) {
        if (weightName.compare((*it).first) == 0) {
            (*it).second->importFreeParameters(dirName + "/"
                                                + (*it).first + ".syntxt");
            std::cout << "Weight " << (*it).first
            << " successfully imported" << std::endl;
            weightsFound = true;
        }
    }
    if (!weightsFound)
        std::cout << "Warning: weight file " << weightName
        << " was not found!" << std::endl;
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

std::shared_ptr<N2D2::CMonitor> N2D2::DeepNet::getCMonitor(const std::string
                                                         & name) const
{
    const std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
        = mCMonitors.find(name);

    if (it == mCMonitors.end())
        throw std::runtime_error("CMonitor for layer " + name
                                 + " does not exist");

    return (*it).second;
}

std::vector<std::shared_ptr<N2D2::Cell> >
N2D2::DeepNet::getChildCells(const std::string& name) const
{
    std::vector<std::shared_ptr<Cell> > childCells;

    for (std::multimap<std::string, std::string>::const_iterator it
         = mParentLayers.begin(), itEnd = mParentLayers.end(); it != itEnd;
         ++it)
    {
        if ((*it).second == name)
            childCells.push_back((*mCells.find((*it).first)).second);
    }

    return childCells;
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

std::vector<unsigned int> N2D2::DeepNet::getReceptiveField(
    const std::string& name,
    const std::vector<unsigned int>& outputField) const
{
    const std::map<std::string, std::shared_ptr<Cell> >::const_iterator itCell
        = mCells.find(name);
    const std::vector<unsigned int> cellReceptiveField
        = (*itCell).second->getReceptiveField(outputField);

    const std::vector<std::shared_ptr<Cell> > parents = getParentCells(name);
    std::vector<unsigned int> maxReceptiveField(
                                (*itCell).second->getOutputsDims().size(), 0);
    bool hasParent = false;

    for (std::vector<std::shared_ptr<Cell> >::const_iterator
         it = parents.begin(), itEnd = parents.end(); it != itEnd; ++it)
    {
        if (*it) {
            const std::vector<unsigned int> receptiveField
                = getReceptiveField((*it)->getName(), cellReceptiveField);

            std::transform(receptiveField.begin(), receptiveField.end(),
                           maxReceptiveField.begin(), maxReceptiveField.begin(),
                           Utils::max<unsigned int>());

            hasParent = true;
        }
    }

    return (hasParent) ? maxReceptiveField : cellReceptiveField;
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
    for (std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
         = mCMonitors.begin(),
         itEnd = mCMonitors.end();
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
    for (std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
         = mCMonitors.begin(),
         itEnd = mCMonitors.end();
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
    for (std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator it
         = mCMonitors.begin(),
         itEnd = mCMonitors.end();
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
     std::shared_ptr<CEnvironment> cenv = std::dynamic_pointer_cast
        <CEnvironment>(mStimuliProvider);

    if (cenv) {
        cenv->initialize();
    }

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
    const Tensor<Float_T> frame = env->getData(0);

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

void N2D2::DeepNet::rescaleAdditiveParameters(Float_T rescaleFactor) {
    for (auto it = mLayers.begin() + 1; it != mLayers.end(); ++it) {
        for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
            auto& cell = (*mCells.find(*itCell)).second;
            cell->processFreeParameters([&](double v) { return v/rescaleFactor; }, 
                                        Cell::Additive);
        }
    }
}

void N2D2::DeepNet::normalizeFreeParameters(double normFactor)
{
    Float_T bNorm = 1.0;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1, itEnd = mLayers.end(); it != itEnd; ++it)
    {
        Float_T wNorm = 0.0;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
            itCellEnd = (*it).end(); itCell != itCellEnd; ++itCell)
        {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;

            if (bNorm != 1.0) {
                cell->processFreeParameters(std::bind(std::divides<double>(),
                                                      std::placeholders::_1,
                                                      bNorm), Cell::Additive);
            }

            Float_T wMin, wMax;
            std::tie(wMin, wMax) = cell->getFreeParametersRange(false);

            const Float_T wMaxAbs = std::max(std::abs(wMin), std::abs(wMax));
            wNorm = std::max(wMaxAbs, wNorm);
        }

        // Don't normalize the layer if all free parameters are equal to 0
        if(wNorm == 0.0) {
            continue;
        }

        wNorm /= normFactor;
        bNorm *= wNorm;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
            itCellEnd = (*it).end(); itCell != itCellEnd; ++itCell)
        {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;

            cell->processFreeParameters(std::bind(std::divides<double>(),
                                                  std::placeholders::_1,
                                                  wNorm));
        }
    }
}

void
N2D2::DeepNet::normalizeOutputsRange(const std::map
                                     <std::string, RangeStats>& outputsRange,
                                     double normFactor,
                                     double useMean,
                                     double stdDevOffset)
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
            std::string layerName;
            double nbElements = 0.0;
            double sum = 0.0;
            double sumSquare = 0.0;

            for (std::vector<std::string>::const_iterator itCell
                 = (nextIsPool) ? (*(it + 1)).begin() : (*it).begin(),
                 itCellEnd = (nextIsPool) ? (*(it + 1)).end() : (*it).end();
                 itCell != itCellEnd;
                 ++itCell)
            {
                if (!layerName.empty())
                    layerName += "_";

                layerName += (*itCell);

                const std::map<std::string, RangeStats>::const_iterator itRange
                    = outputsRange.find(*itCell);

                if (itRange != outputsRange.end()) {
                    nbElements += (*itRange).second.moments()[0];
                    sum += (*itRange).second.moments()[1];
                    sumSquare += (*itRange).second.moments()[2];
                }
                else {
                    throw std::runtime_error("Missing range stats for cell: "
                                             + (*itCell));
                }
            }

            const double mean = sum / nbElements;
            const double meanSquare = sumSquare / nbElements;
            const double stdDev = std::sqrt(meanSquare - mean * mean);

            scalingFactor = mean + stdDevOffset * stdDev;

            std::cout << "Scaling factor " << layerName << " = "
                << scalingFactor << " (mean: " << mean << " | stddev: "
                << stdDev << ")" << std::endl;
        }
        else {
            for (std::vector<std::string>::const_iterator itCell
                 = (nextIsPool) ? (*(it + 1)).begin() : (*it).begin(),
                 itCellEnd = (nextIsPool) ? (*(it + 1)).end() : (*it).end();
                 itCell != itCellEnd;
                 ++itCell)
            {
                const std::map<std::string, RangeStats>::const_iterator itRange
                    = outputsRange.find(*itCell);

                if (itRange != outputsRange.end()) {
                    scalingFactor
                        = std::max(scalingFactor, (*itRange).second.maxVal());
                }
                else {
                    throw std::runtime_error("Missing range stats for cell: "
                                             + (*itCell));
                }
            }
        }

        scalingFactor /= normFactor;

        const double appliedFactor = scalingFactor / prevScalingFactor;
        bool applied = false;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd;
             ++itCell)
        {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cellFrame && cellFrame->getActivation()
                && cellFrame->getActivation()->getType()
                   == std::string("Rectifier"))
            {
/*
                const int shifting = (appliedFactor > 1.0)
                    ? Utils::round(log2(appliedFactor))
                    : -Utils::round(log2(1.0 / appliedFactor));

                cellFrame->getActivation()->setParameter<int>("Shifting",
                    shifting);

                std::cout << Utils::cnotice << "Shifting " << (*itCell)
                    << " = " << shifting << Utils::cdef << std::endl;
*/

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

void
N2D2::DeepNet::normalizeOutputsRange(const std::map
                                     <std::string, Histogram>& outputsHistogram,
                                     const std::map
                                     <std::string, RangeStats>& outputsRange,
                                     unsigned int nbLevels,
                                     unsigned int nbPasses)
{
    double prevScalingFactor = 1.0;
    bool nextIsMaxPool = false;
    bool nextIsAvgPool = false;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1, itEnd = mLayers.end(); it != itEnd; ++it)
    {
        if (nextIsMaxPool || nextIsAvgPool) {
            nextIsMaxPool = false;
            nextIsAvgPool = false;
            continue;
        }

        if (it + 1 != itEnd) {
            std::shared_ptr<PoolCell> poolCell = std::dynamic_pointer_cast
                <PoolCell>((*mCells.find(*(*(it + 1)).begin())).second);

            if (poolCell && poolCell->getPooling() == PoolCell::Max) {
                std::cout << "MAX pool following: " << *(*(it + 1)).begin()
                    << std::endl;
                nextIsMaxPool = true;
            }

            if (poolCell && poolCell->getPooling() == PoolCell::Average) {
                std::cout << "AVG pool following: " << *(*(it + 1)).begin()
                    << std::endl;
                nextIsAvgPool = true;
            }
        }

        double scalingFactor = 0.0;
        double appliedFactor = 0.0;
        bool applied = false;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
             itCellEnd = (*it).end(); itCell != itCellEnd; ++itCell)
        {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (!cellFrame)
                continue;

            const std::shared_ptr<Activation> activation
                = cellFrame->getActivation();
            const std::string activationType = (activation)
                ? activation->getType() : "Linear";

            std::map<std::string, Histogram>::const_iterator itHistogram;
            std::map<std::string, RangeStats>::const_iterator itRange;

            const std::vector<std::string>::const_iterator itCellStats
                = (nextIsMaxPool) ? (*(it + 1)).begin() : itCell;
            itHistogram = outputsHistogram.find(*itCellStats);
            itRange = outputsRange.find(*itCellStats);

            if (itHistogram == outputsHistogram.end()) {
                throw std::runtime_error("Missing histogram for cell: "
                                         + (*itCellStats));
            }

            if (itRange == outputsRange.end()) {
                throw std::runtime_error("Missing range stats for cell: "
                                         + (*itCellStats));
            }

            if (activationType == "Rectifier"
                || (*itRange).second.minVal() >= 0.0
                    // e.g. average pooling following a Rectifier
                || (activationType == "Linear" && cell->getNbOutputs() > 2))
                    // Here we assume that this layer is the preceding layer of
                    // a softmax with more than 2 channels.
                    // In this case, the full range is required as several
                    // values can be very high.
            {
                double threshold = std::max(std::abs((*itRange).second.minVal()), 
                                            std::abs((*itRange).second.maxVal()));

                if (nbPasses > 0) {
                    Histogram hist = (*itHistogram).second;

                    // First pass
                    threshold = hist.calibrateKL(nbLevels);

                    // More passes
                    for (unsigned int p = 1; p < nbPasses; ++p) {
                        hist.truncate(threshold);
                        threshold = hist.calibrateKL(nbLevels);
                    }
                }

                scalingFactor = std::max(scalingFactor, threshold);
            }
            else {
                // Here we assume that this layer has a logistic activation
                // (or is preceding a 2 channels softmax, which is equivalent)
                // The loss function minimization tends to push
                // the output values to either -inf (wrong class) or +inf
                // (correct class)
                // Therefore, high precision is required towards 0 in order to
                // be able to distinguish the two.
                scalingFactor = std::max(scalingFactor,
                            std::abs((*itRange).second.stdDev()) / nbLevels);
                        // mean() cannot be used because it can be 0.0 or close
            }
/*
            else {
                // Here we assume that this layer is the preceding layer of a
                // softmax with more than 2 channels.
                // In this case, the full range is required as several values
                // can be very high.
                scalingFactor = std::max(scalingFactor,
                                         (*itRange).second.maxVal);
            }
*/
        }

        const double targetFactor = scalingFactor / prevScalingFactor;

        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
             itCellEnd = (*it).end(); itCell != itCellEnd; ++itCell)
        {
            std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (!cellFrame)
                continue;

            const std::shared_ptr<Activation>& activation
                = cellFrame->getActivation();
            const std::string activationType = (activation)
                ? activation->getType() : "Linear";

            if (activationType == "Rectifier"
                || activationType == "Logistic"
                || activationType == "LogisticWithLoss"
                || activationType == "Linear")
            {
                const int shifting = (targetFactor > 1.0)
                    ? Utils::round(log2(targetFactor))
                    : -Utils::round(log2(1.0 / targetFactor));
                const double shiftedFactor = (shifting >= 0)
                    ? 1.0 / (1 << shifting)
                    : (1 << (-shifting));
                // max(1, ...) is used to avoid weights truncation
                const double remainingFactor = std::max(1.0,
                                                targetFactor * shiftedFactor);
                appliedFactor = prevScalingFactor
                                            * (remainingFactor / shiftedFactor);

                if (activation)
                    activation->setParameter<int>("Shifting", shifting);
                else if (shifting != 0) {
                    std::cout << Utils::cwarning
                        << "DeepNet::normalizeOutputsRange(): no activation "
                        "for cell " << (*itCell) << ", unable to add Shifting"
                        << Utils::cdef << std::endl;
                }

                if (activationType == "Rectifier" || activationType == "Linear")
                {
                    if (activation)
                        activation->setParameter<double>("Clipping", 1.0);
                    else {
                        std::cout << Utils::cwarning
                            << "DeepNet::normalizeOutputsRange(): no "
                            "activation for cell " << (*itCell) << ", unable "
                            "to add Clipping" << Utils::cdef << std::endl;
                    }
                }

                cell->processFreeParameters(std::bind(std::divides<double>(),
                                                      std::placeholders::_1,
                                                      remainingFactor), 
                                                      Cell::Multiplicative);
                
                cell->processFreeParameters(std::bind(std::divides<double>(),
                                                      std::placeholders::_1,
                                                      prevScalingFactor * remainingFactor), 
                                                      Cell::Additive);

                std::cout << std::setprecision(4) << (*itCell) << ": "
                    "scaling = " << scalingFactor << "   "
                    "previous scaling = " << prevScalingFactor << "   "
                    "target = " << targetFactor << "   "
                    "applied = " << appliedFactor << "   "
                    "shifting = " << shifting << "    "
                    "shifting factor = " << shiftedFactor << "    "
                    "remaining = " << remainingFactor << "    " << std::endl;

                applied = true;
            }
        }

        if (applied)
            prevScalingFactor = appliedFactor;
    }
}

/**
 * Ref: https://tkv.io/posts/fusing-batchnorm-and-conv/
*/
void N2D2::DeepNet::fuseBatchNormWithConv() {
    std::cout << "Fuse BatchNorm with Conv..." << std::endl;

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(); it != mCells.end(); )
    {
        const std::shared_ptr<Cell>& cell = (*it).second;
        ++it; // increase it before being potentially invalided by removeCell()

        // check every BatchNorm cell
        if (cell->getType() == BatchNormCell::Type) {
            // check if a Conv is preceding
            const std::vector<std::shared_ptr<Cell> > bnParents
                = getParentCells(cell->getName());

            if (bnParents.size() == 1
                && bnParents[0]->getType() == ConvCell::Type)
            {
                // only a single Conv is preceding
                // check if BatchNorm is the only child
                const std::vector<std::shared_ptr<Cell> > convChilds
                    = getChildCells(bnParents[0]->getName());

                if (convChilds.size() == 1) {
                    assert(convChilds[0] == cell);

                    // OK, Conv's only child is BatchNorm, fuse them...
                    std::cout << "  fuse BatchNorm \"" << cell->getName()
                        << "\" with Conv \"" << bnParents[0]->getName() << "\""
                        << std::endl;

                    std::shared_ptr<ConvCell> convCell =
                        std::dynamic_pointer_cast<ConvCell>(bnParents[0]);
                    const bool noBias = convCell->getParameter<bool>("NoBias");

                    std::shared_ptr<BatchNormCell> bnCell =
                        std::dynamic_pointer_cast<BatchNormCell>(cell);
                    const Tensor<double>& bnScales
                        = tensor_cast<double>(*(bnCell->getScales()));
                    const Tensor<double>& bnBiases
                        = tensor_cast<double>(*(bnCell->getBiases()));
                    const Tensor<double>& bnMeans
                        = tensor_cast<double>(*(bnCell->getMeans()));
                    const Tensor<double>& bnVariances
                        = tensor_cast<double>(*(bnCell->getVariances()));
                    const double eps = bnCell->getParameter<double>("Epsilon");

                    assert(bnScales.size() == convCell->getNbOutputs());
                    assert(bnBiases.size() == convCell->getNbOutputs());
                    assert(bnMeans.size() == convCell->getNbOutputs());
                    assert(bnVariances.size() == convCell->getNbOutputs());
                    assert(eps > 0.0);

                    std::shared_ptr<Cell_Frame_Top> convCellTop =
                        std::dynamic_pointer_cast<Cell_Frame_Top>(bnParents[0]);
                    std::shared_ptr<Cell_Frame_Top> bnCellTop =
                        std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

                    // Fuse only if  the convolution has a linear activation
                    if (convCellTop->getActivation()
                        && std::string(convCellTop->getActivation()
                                        ->getType()) != "Linear")
                    {
                        std::cout << Utils::cwarning << "  -> non-linear "
                            "activation before BatchNorm prevents fuse!"
                            << Utils::cdef << std::endl;
                        continue;
                    }

                    convCellTop->setActivation(bnCellTop->getActivation());

                    if (noBias)
                        convCell->setParameter<bool>("NoBias", false);

                    for (unsigned int output = 0;
                        output < convCell->getNbOutputs(); ++output)
                    {
                        const double factor = bnScales(output)
                                        / std::sqrt(eps + bnVariances(output));

                        // Weights adjustments
                        for (unsigned int channel = 0;
                            channel < convCell->getNbChannels(); ++channel)
                        {
                            Tensor<double> kernel;
                            convCell->getWeight(output, channel, kernel);

                            for (unsigned int index = 0, size = kernel.size();
                                index < size; ++index)
                            {
                                kernel(index) *= factor;
                            }

                            convCell->setWeight(output, channel, kernel);
                        }

                        // Biases adjustments
                        Tensor<double> bias;

                        if (noBias)
                            bias.resize({1}, 0.0);
                        else
                            convCell->getBias(output, bias);

                        bias(0) = bnBiases(output) + (bias(0) - bnMeans(output)) * factor;
                        convCell->setBias(output, bias);
                    }

                    // Replace BatchNorm by Conv for BatchNorm childs
                    // and BatchNorm cell removal from DeepNet
                    removeCell(cell, true);
                }
                else {
                    std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                        << cell->getName() << "\" because parent Conv "
                        "(\"" << bnParents[0]->getName() << "\") has multiple "
                        "childs" << Utils::cdef << std::endl;
                }
            }
            else if (bnParents.size() == 1) {
                std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                    << cell->getName() << "\" because parent cell (\""
                    << bnParents[0]->getName() << "\") is not a Conv"
                    << Utils::cdef << std::endl;
            }
            else {
                std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                    << cell->getName() << "\" because it has multiple "
                    "parents (not supported)" << Utils::cdef << std::endl;
            }
        }
    }
}

void N2D2::DeepNet::removeDropout() {
    std::cout << "Remove Dropout..." << std::endl;

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(); it != mCells.end(); )
    {
        const std::shared_ptr<Cell>& cell = (*it).second;
        ++it; // increase it before being potentially invalided by removeCell()

        // check every Dropout cell
        if (cell->getType() == DropoutCell::Type) {
            // remove them
            std::cout << "  remove Dropout \"" << cell->getName()
                << "\"" << std::endl;

            removeCell(cell, true);
        }
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
            const std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            cellFrame->getOutputs().synchronizeDToH();
            const Tensor<Float_T> outputs
                = tensor_cast<Float_T>(cellFrame->getOutputs())[batchPos];

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
            const std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            cellFrame->getDiffInputs().synchronizeDToH();
            const Tensor<Float_T> diffInputs
                = tensor_cast<Float_T>(cellFrame->getDiffInputs())[batchPos];

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

void N2D2::DeepNet::logSchedule(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    std::vector<std::pair<std::string, std::shared_ptr<Solver> > > solvers;

    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(), itEnd = mCells.end(); it != itEnd; ++it)
    {
        const std::shared_ptr<Cell> cell = (*it).second;

        // BatchNorm
        const std::shared_ptr<BatchNormCell> cellBatchNorm
            = std::dynamic_pointer_cast<BatchNormCell>(cell);

        if (cellBatchNorm) {
            solvers.push_back(std::make_pair((*it).first + "_scale",
                                             cellBatchNorm->getScaleSolver()));
            solvers.push_back(std::make_pair((*it).first + "_bias",
                                             cellBatchNorm->getBiasSolver()));
        }

        // Conv
        const std::shared_ptr<ConvCell> cellConv
            = std::dynamic_pointer_cast<ConvCell>(cell);

        if (cellConv) {
            solvers.push_back(std::make_pair((*it).first + "_bias",
                                             cellConv->getBiasSolver()));
            solvers.push_back(std::make_pair((*it).first + "_weights",
                                             cellConv->getWeightsSolver()));
        }

        // Deconv
        const std::shared_ptr<DeconvCell> cellDeconv
            = std::dynamic_pointer_cast<DeconvCell>(cell);

        if (cellDeconv) {
            solvers.push_back(std::make_pair((*it).first + "_bias",
                                             cellDeconv->getBiasSolver()));
            solvers.push_back(std::make_pair((*it).first + "_weights",
                                             cellDeconv->getWeightsSolver()));
        }

        // Fc
        const std::shared_ptr<FcCell> cellFc
            = std::dynamic_pointer_cast<FcCell>(cell);

        if (cellFc) {
            solvers.push_back(std::make_pair((*it).first + "_bias",
                                             cellFc->getBiasSolver()));
            solvers.push_back(std::make_pair((*it).first + "_weights",
                                             cellFc->getWeightsSolver()));
        }
    }

#pragma omp parallel for
    for (int i = 0; i < (int)solvers.size(); ++i) {
        if (solvers[i].second) {
            solvers[i].second->logSchedule(dirName + "/"
                    + solvers[i].first + "_schedule.log",
                mStimuliProvider->getBatchSize(),
                mDatabase->getNbStimuli(Database::Learn));
        }
    }
}

void N2D2::DeepNet::logStats(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    // Global stats
    const std::string logFileName = dirName + "/stats.log";

    Cell::Stats globalStats;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1, itEnd = mLayers.end(); it != itEnd; ++it)
    {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd; ++itCell)
        {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            cell->getStats(globalStats);
        }
    }

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
    const unsigned int hiddenData = globalStats.nbNodes;

    logData << "[Memory]\n"
               "Input data (int-8 bits): " << inputData / 1000.0
            << " kB   (" << inputData / 1024.0 << " KiB)\n"
               "Input data (float-16 bits): " << 2.0 * inputData / 1000.0
            << " kB   (" << 2.0 * inputData / 1024.0 << " KiB)\n"
               "Input data (float-32 bits): " << 4.0 * inputData / 1000.0
            << " kB   (" << 4.0 * inputData / 1024.0 << " KiB)\n"
               "Free parameters (int-8 bits): " << freeParameters / 1000.0
            << " kB   (" << freeParameters / 1024.0 << " KiB)\n"
               "Free parameters (float-16 bits): " << 2.0 * freeParameters
                                                      / 1000.0
            << " kB   (" << 2.0 * freeParameters / 1024.0 << " KiB)\n"
               "Free parameters (float-32 bits): " << 4.0 * freeParameters
                                                      / 1000.0
            << " kB   (" << 4.0 * freeParameters / 1024.0 << " KiB)\n"
               "Layers data (int-8 bits): " << hiddenData / 1000.0
            << " kB   (" << hiddenData / 1024.0 << " KiB)\n"
               "Layers data (float-16 bits): " << 2.0 * hiddenData / 1000.0
            << " kB   (" << 2.0 * hiddenData / 1024.0 << " KiB)\n"
               "Layers data (float-32 bits): " << 4.0 * hiddenData / 1000.0
            << " kB   (" << 4.0 * hiddenData / 1024.0 << " KiB)\n\n";

    logData << "[Computing]\n"
               "MACS / input data: " << globalStats.nbConnections / 1.0e6
            << "M\n";

    // Cells stats
    const std::string statsFileName = dirName + "/stats.dat";
    const std::string relStatsFileName = dirName + "/stats_relative.dat";

    std::ofstream stats(statsFileName.c_str());

    if (!stats.good())
        throw std::runtime_error("Could not open stats file: " + statsFileName);

    stats << "Cell ParamMemory Computing DataMemory\n";

    std::ofstream relStats(relStatsFileName.c_str());

    if (!relStats.good())
        throw std::runtime_error("Could not open stats file: "
                                 + relStatsFileName);

    relStats << "Cell ParamMemory(%) Computing(%) DataMemory(%)\n";

    unsigned int maxStringSizeCellName = 1;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1, itEnd = mLayers.end(); it != itEnd; ++it)
    {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd; ++itCell)
        {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;

            Cell::Stats cellStats;
            cell->getStats(cellStats);

            maxStringSizeCellName = (maxStringSizeCellName >
                                        cell->getName().size()) ?
                                        maxStringSizeCellName :
                                            cell->getName().size() ;

            stats << (*itCell) << " "
                       << cellStats.nbSynapses << " "
                       << cellStats.nbConnections << " "
                       << cellStats.nbNodes << "\n";

            relStats << (*itCell) << " " << std::fixed << std::setprecision(2)
                    << std::setfill('0')
                    << ((float) cellStats.nbSynapses /
                                (float) globalStats.nbSynapses)*100.0
                    << " "
                    << ((float) cellStats.nbConnections /
                                (float) globalStats.nbConnections)*100.0
                    << " "
                    << ((float) cellStats.nbNodes /
                                (float) globalStats.nbNodes)*100.0
                    << "\n";
        }
    }

    stats.close();
    relStats.close();

    const std::string logStatsDataFileName = dirName
                                        + "/stats_data_logarithmic.dat";
    const std::string relStatsDataFileName = dirName
                                        + "/stats_data_relative.dat";
    const std::string logStatsParamFileName = dirName
                                        + "/stats_parameters_logarithmic.dat";
    const std::string relStatsParamFileName = dirName
                                        + "/stats_parameters_relative.dat";
    const std::string logStatsComputingFileName = dirName
                                        + "/stats_computing_logarithmic.dat";
    const std::string relStatsComputingFileName = dirName
                                        + "/stats_computing_relative.dat";

    std::string paramTitle = "Param Memory";
    std::string dataTitle = "Data Memory";
    std::string computingTitle = "Computing";

    std::stringstream multiTermStr;
    multiTermStr << "set term png size "
            << (mLayers.size() * 50)*2
            << ",1800 enhanced large";

    Gnuplot multiplot;
    multiplot.saveToFile(statsFileName);
    multiplot << multiTermStr.str();
    multiplot.setMultiplot(3, 2);
    multiplot.set("origin 0,0.66");
    multiplot.set("grid");
    drawHistogram(paramTitle + "[int-8 bits] (bytes)",
                  statsFileName,
                  2U,
                  maxStringSizeCellName,
                  true,
                  multiplot);

    multiplot.set("origin 0.5,0.66");
    multiplot.set("grid");
    drawHistogram(paramTitle + " (%)",
                  relStatsFileName,
                  2U,
                  maxStringSizeCellName,
                  false,
                  multiplot);


    multiplot.set("origin 0.0,0.33");
    multiplot.set("grid");
    drawHistogram(dataTitle + "[int-8 bits] (bytes)",
                  statsFileName,
                  4U,
                  maxStringSizeCellName,
                  true,
                  multiplot);

    multiplot.set("origin 0.5,0.33");
    multiplot.set("grid");
    drawHistogram(dataTitle + " (%)",
                  relStatsFileName,
                  4U,
                  maxStringSizeCellName,
                  false,
                  multiplot);

    multiplot.set("origin 0.0,0.0");
    multiplot.set("grid");
    drawHistogram(computingTitle + "(MACs)",
                  statsFileName,
                  3U,
                  maxStringSizeCellName,
                  true,
                  multiplot);

    multiplot.set("origin 0.5,0.0");
    multiplot.set("grid");
    drawHistogram(computingTitle + " (%)",
                  relStatsFileName,
                  3U,
                  maxStringSizeCellName,
                  false,
                  multiplot);

    multiplot.unsetMultiplot();

    std::stringstream termStr;
    termStr << "set term png size "
            << (mLayers.size() * 50)
            << ",600 enhanced large";

    Gnuplot paramPlot;
    paramPlot.saveToFile(logStatsParamFileName);
    paramPlot << termStr.str();
    drawHistogram(paramTitle + "[int-8 bits] (bytes)",
                  statsFileName,
                  2U,
                  maxStringSizeCellName,
                  true,
                  paramPlot);

    Gnuplot relParamPlot;
    relParamPlot.saveToFile(relStatsParamFileName);
    relParamPlot << termStr.str();
    drawHistogram(paramTitle + "[int-8 bits] (%)",
                  relStatsFileName,
                  2U,
                  maxStringSizeCellName,
                  false,
                  relParamPlot);


    Gnuplot computingPlot;
    computingPlot.saveToFile(logStatsComputingFileName);
    computingPlot << termStr.str();
    drawHistogram(computingTitle + "(MACs)",
                  statsFileName,
                  3U,
                  maxStringSizeCellName,
                  true,
                  computingPlot);

    Gnuplot relComputingPlot;
    relComputingPlot.saveToFile(relStatsComputingFileName);
    relComputingPlot << termStr.str();
    drawHistogram(computingTitle + " (%)",
                  relStatsFileName,
                  3U,
                  maxStringSizeCellName,
                  false,
                  relComputingPlot);

    Gnuplot dataPlot;
    dataPlot.saveToFile(logStatsDataFileName);
    dataPlot << termStr.str();
    drawHistogram(dataTitle + "[int-8 bits] (bytes)",
                  statsFileName,
                  4U,
                  maxStringSizeCellName,
                  true,
                  dataPlot);
    Gnuplot relDataPlot;
    relDataPlot.saveToFile(relStatsDataFileName);
    relDataPlot << termStr.str();
    drawHistogram(dataTitle + "[int-8 bits] (%)",
                  relStatsFileName,
                  4U,
                  maxStringSizeCellName,
                  false,
                  relDataPlot);
}

void N2D2::DeepNet::drawHistogram(std::string title,
                                  const std::string& dataFileName,
                                  unsigned int fileRow,
                                  unsigned int maxLabelSize,
                                  bool isLog, Gnuplot& p)
{
    p << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
               "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    p << "unset colorbox";
    p.set("style histogram cluster gap 1");
    p.set("style data histograms");
    p.set("style fill pattern 1.00 border");
    p.set("ytics nomirror");
    if(isLog) p.set("logscale y");
    p.setYlabel(title);
    p.set("grid");
    p.set("ytics textcolor lt 2");
    p.set("ylabel textcolor lt 2");
    p.set("format y \"%.0s%c\"");
    p.unset("key");
    p.set("tmargin", 4);
    p.set("bmargin", (maxLabelSize/2)+2);
    p.set("xtics rotate");
    p.set("palette model RGB defined (1 \"blue\", 2 \"red\")");
    p.set("boxwidth 0.5");

    std::stringstream fileRowStr;
    fileRowStr << fileRow;
    std::stringstream maxLabelSizeStr;
    maxLabelSizeStr << maxLabelSize;

    if(isLog)
    {
        p.plot(
            dataFileName,
            "i 0 using ($"
            + fileRowStr.str()
            + "):xticlabels(wrap(stringcolumn(1),"
            + maxLabelSizeStr.str()
            + ")) ti col,"
            " '' i 0 using 0:($"
            + fileRowStr.str()
            + "):(gprintf(\"%.2s%c\",$"
            + fileRowStr.str()
            + ")) ti col with labels rotate"
            " offset char 0,2 textcolor lt 2,"
            + " '' i 0 using 0:($"
            + fileRowStr.str() + "):" + fileRowStr.str()
            + "ti col with boxes lc palette"
            );
            p << "unset logscale y";
    }
    else
    {
        p.plot(
            dataFileName,
            "i 0 using ($"
            + fileRowStr.str()
            + "):xticlabels(wrap(stringcolumn(1),"
            + maxLabelSizeStr.str()
            + ")) ti col,"
            " '' i 0 using 0:($"
            + fileRowStr.str() + "):($"
            + fileRowStr.str()
            + ") ti col with labels rotate"
            " offset char 0,2 textcolor lt 2,"
            + " '' i 0 using 0:($"
            + fileRowStr.str() + "):" + fileRowStr.str()
            + "ti col with boxes lc palette"
            );
    }

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
	    //TODO: Implement this for CMonitor

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
        if (mSignalsDiscretization > 0) {
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>((*itTargets)->
                                                            getCell());

            cellFrame->discretizeSignals(mSignalsDiscretization,
                                         Cell_Frame_Top::Out);
        }

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
        const std::string dirName = "weights_discretized";
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
                if(cellFrame->isCuda())
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
         ++itTargets)
    {
        std::shared_ptr<Cell_Frame_Top> cellFrame
            = std::dynamic_pointer_cast<Cell_Frame_Top>((*itTargets)->
                                                        getCell());

        if (mSignalsDiscretization > 0) {
            cellFrame->discretizeSignals(mSignalsDiscretization,
                                         Cell_Frame_Top::Out);
        }

        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->process(set);

        if (timings != NULL) {
#ifdef CUDA
            if (cellFrame->isCuda())
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

void N2D2::DeepNet::cTicks(Time_T start,
                           Time_T stop,
                           Time_T timestep,
                           bool record)
{
    const unsigned int nbLayers = mLayers.size();

    std::shared_ptr<CEnvironment> cEnv = std::dynamic_pointer_cast
        <CEnvironment>(mStimuliProvider);

    if (!cEnv)
        throw std::runtime_error("DeepNet::cTicks(): requires a CEnvironment.");

    if (stop < start){
        throw std::runtime_error("DeepNet::cTicks(): stop < start");
    }

    cEnv->initializeSpikeGenerator(start, stop);

    if (record) {
        for (std::vector<std::vector<std::string> >::const_iterator it
        = mLayers.begin(),
        itEnd = mLayers.end();
        it != itEnd;
        ++it) {
            for (std::vector<std::string>::const_iterator
                itCell = (*it).begin(),
                itCellEnd = (*it).end();
            itCell != itCellEnd;
            ++itCell) {
                const std::map
                    <std::string, std::shared_ptr<CMonitor> >::const_iterator
                itMonitor = mCMonitors.find(*itCell);

                if (itMonitor == mCMonitors.end()){

                    continue;
                }

                (*itMonitor).second->tick(start);
            }
        }
    }


    for (Time_T t = start+timestep; t <= stop; t += timestep) {
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
                //std::cout << mCells[(*itCell)]->getName() << std::endl;
                if (cellCSpike->tick(t))
                    return;
            }
        }

        if (record) {

            for (std::vector<std::vector<std::string> >::const_iterator it
            = mLayers.begin(),
            itEnd = mLayers.end();
            it != itEnd;
            ++it) {
                for (std::vector<std::string>::const_iterator
                    itCell = (*it).begin(),
                    itCellEnd = (*it).end();
                itCell != itCellEnd;
                ++itCell) {
                    const std::map
                        <std::string, std::shared_ptr<CMonitor> >::const_iterator
                    itMonitor = mCMonitors.find(*itCell);

                    if (itMonitor == mCMonitors.end()){

                        continue;
                    }

                    (*itMonitor).second->tick(t);
                }
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

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin(),
         itEnd = mLayers.end();
         it != itEnd;
         ++it) {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
         itCell != itCellEnd;
         ++itCell) {

            std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator
            itMonitor = mCMonitors.find(*itCell);

            if (itMonitor ==  mCMonitors.end())
                continue;

            //(*itMonitor).second->clearMostActive();

            (*itMonitor).second->reset(timestamp);

         }
    }
}

void N2D2::DeepNet::initializeCMonitors(unsigned int nbTimesteps)
{
    for (std::vector<std::vector<std::string> >::const_iterator it
    = mLayers.begin(),
    itEnd = mLayers.end();
    it != itEnd;
    ++it) {
        for (std::vector<std::string>::const_iterator
            itCell = (*it).begin(),
            itCellEnd = (*it).end();
        itCell != itCellEnd;
        ++itCell) {
            const std::map
                <std::string, std::shared_ptr<CMonitor> >::const_iterator
            itMonitor = mCMonitors.find(*itCell);

            if (itMonitor == mCMonitors.end()){
                continue;
            }

            (*itMonitor).second->initialize(nbTimesteps,
                    mStimuliProvider->getDatabase().getNbLabels());
        }
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

void N2D2::DeepNet::logEstimatedLabelsJSON(const std::string& dirName) const
{
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets) {
        (*itTargets)->logEstimatedLabelsJSON(dirName);
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
    const double totalFPS = 1.0/totalTime;
    std::stringstream totalFPSstr;
    totalFPSstr << totalFPS << " (FPS)";
    std::ofstream timingsData(fileName.c_str());
    unsigned int maxStringSizeCellName = 1;

    if (!timingsData.good())
        throw std::runtime_error("Could not open timings file: " + fileName);

    timingsData << "Cell Timing(s) Timing(%)\n";

    double propTime = 0.0;
    double backPropTime = 0.0;
    double updateTime = 0.0;

    for (std::vector<std::pair<std::string, double> >::const_iterator it
         = timings.begin(),
         itEnd = timings.end();
         it != itEnd;
         ++it)
    {
        const std::string backPropStr = "[back-prop]";
        const std::string updateStr = "[update]";

        if (std::equal(backPropStr.rbegin(), backPropStr.rend(),
                            (*it).first.rbegin()))
        {
            backPropTime += (*it).second;
        }
        else if (std::equal(updateStr.rbegin(), updateStr.rend(),
                            (*it).first.rbegin()))
        {
            updateTime += (*it).second;
        }
        else
            propTime += (*it).second;

        timingsData << (*it).first << " " << (*it).second << " "
                    << (100.0 * (*it).second / totalTime) << "\n";

        maxStringSizeCellName = (maxStringSizeCellName > ((*it).first).size()) ?
                                    maxStringSizeCellName : ((*it).first).size() ;

    }

    timingsData << "\n\n";
    timingsData << "Total[prop] " << propTime << " "
                << (100.0 * propTime / totalTime) << "\n";
    timingsData << "Total[back-prop] " << backPropTime << " "
                << (100.0 * backPropTime / totalTime) << "\n";
    timingsData << "Total[update] " << updateTime << " "
                << (100.0 * updateTime / totalTime) << "\n";
    timingsData << "Total " << totalTime << " "
                << (100.0 * totalTime / totalTime) << "\n";

    timingsData.close();

    std::stringstream outputStr;
    outputStr << "set term png size "
            << (timings.size() * 50)
            << ",1600 enhanced large";

    Gnuplot multiplot;
    multiplot.saveToFile(fileName);
    multiplot << outputStr.str();
    multiplot.setMultiplot(2, 1);
    multiplot.set("origin 0.0,0.0");
    multiplot.set("grid");
    drawHistogram("Timing (s)" + totalFPSstr.str(),
                  fileName,
                  2U,
                  maxStringSizeCellName,
                  true,
                  multiplot);

    multiplot.set("origin 0.0,0.5");
    multiplot.set("grid");
    drawHistogram("Relative Timing (%)" + totalFPSstr.str(),
                  fileName,
                  3U,
                  maxStringSizeCellName,
                  false,
                  multiplot);

    multiplot.unsetMultiplot();
}

void N2D2::DeepNet::logReceptiveFields(const std::string& fileName) const
{
    std::ofstream receptiveFields(fileName.c_str());

    if (!receptiveFields.good())
        throw std::runtime_error("Could not open receptive field file: "
                                 + fileName);

    receptiveFields << "Name R.F. R.F./env\n";

    Gnuplot gnuplot;
    unsigned int objCount = 0;
    unsigned int objMaxSize = 0;
    const unsigned int maxObj = 10000;
    const unsigned int objOffset = 5;

    for (std::vector<std::vector<std::string> >::const_iterator it
         = mLayers.begin() + 1, itEnd = mLayers.end(); it != itEnd; ++it)
    {
        for (std::vector<std::string>::const_iterator itCell = (*it).begin(),
                                                      itCellEnd = (*it).end();
             itCell != itCellEnd; ++itCell)
        {
            const std::shared_ptr<Cell> cell = (*mCells.find(*itCell)).second;
            const std::vector<unsigned int> receptiveField
                = getReceptiveField(*itCell);

            receptiveFields << (*itCell)
                << " " << cell->getReceptiveField()
                << " " << receptiveField << "\n";

            if (receptiveField.size() >= 2) {
                std::stringstream objStr;
                objStr << "object " << (maxObj - objCount) << " rect at "
                    << (objOffset * objCount) << "," << (objOffset * objCount)
                    << " size " << receptiveField[0] << "," << receptiveField[1]
                    << " fc lt " << objCount << " lw 0"
                    " fill transparent solid 0.33 behind";
                gnuplot.set(objStr.str());

                objStr.str(std::string());
                objStr << "label \"" << (*itCell) << "\" at "
                    << (objOffset * objCount + receptiveField[0] / 2.0 + 1)
                    << "," << (objOffset * objCount);
                gnuplot.set(objStr.str());

                objStr.str(std::string());
                objStr << "label \"" <<  receptiveField[0] << "x"
                    << receptiveField[1] << "\" at "
                    << (objOffset * objCount - receptiveField[0] / 2.0 + 1)
                    << "," << (objOffset * objCount
                               + receptiveField[1] / 2.0 - 3);
                gnuplot.set(objStr.str());

                const unsigned int maxSize = std::max(receptiveField[0],
                                                      receptiveField[1]);
                objMaxSize = std::max(objMaxSize,
                                      objOffset * objCount + maxSize / 2);
                ++objCount;
            }
        }
    }

    gnuplot.setXrange(0, 1.2 * objMaxSize);
    gnuplot.setYrange(0, 1.2 * objMaxSize);
    gnuplot.set("grid");
    gnuplot.set("size square");
    gnuplot.set("key off");
    gnuplot.saveToFile(fileName);
    gnuplot << "if (!exists(\"multiplot\")) "
                "set term png size 800,600 enhanced small";
    gnuplot << "plot 1/0";
}

void
N2D2::DeepNet::reportOutputsRange(std::map
                                  <std::string, RangeStats>& outputsRange) const
{
    if (outputsRange.empty()) {
        // Populate outputsRange first to avoid thread issues
        for (unsigned int i = 0; i < mLayers.size(); ++i) {
            for (std::vector<std::string>::const_iterator
                 itCell = mLayers[i].begin(), itCellEnd = mLayers[i].end();
                 itCell != itCellEnd;
                 ++itCell)
            {
                outputsRange.insert(std::make_pair(*itCell, RangeStats()));
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)mLayers.size(); ++i) {
        for (std::vector<std::string>::const_iterator
             itCell = mLayers[i].begin(), itCellEnd = mLayers[i].end();
             itCell != itCellEnd;
             ++itCell)
        {
            std::shared_ptr<Cell_Frame_Top> cellFrame;

            if (mCells.find(*itCell) != mCells.end()) {
                cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(
                                                (*mCells.find(*itCell)).second);
                cellFrame->getOutputs().synchronizeDToH();
            }

            const Tensor<Float_T>& outputs = (cellFrame)
                ? tensor_cast<Float_T>(cellFrame->getOutputs())
                : mStimuliProvider->getData();

            const std::map<std::string, RangeStats>::iterator itRange
                = outputsRange.find(*itCell);
            (*itRange).second = std::for_each(
                outputs.begin(), outputs.end(), (*itRange).second);
        }
    }
}

void
N2D2::DeepNet::reportOutputsHistogram(std::map
                            <std::string, Histogram>& outputsHistogram) const
{
    if (outputsHistogram.empty()) {
        // Populate outputsHistogram first to avoid thread issues
        for (unsigned int i = 0; i < mLayers.size(); ++i) {
            for (std::vector<std::string>::const_iterator
                 itCell = mLayers[i].begin(), itCellEnd = mLayers[i].end();
                 itCell != itCellEnd;
                 ++itCell)
            {
                std::shared_ptr<Cell_Frame_Top> cellFrame;

                if (mCells.find(*itCell) != mCells.end()) {
                    cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(
                                                (*mCells.find(*itCell)).second);
                    cellFrame->getOutputs().synchronizeDToH();
                }

                const Tensor<Float_T>& outputs = (cellFrame)
                    ? tensor_cast<Float_T>(cellFrame->getOutputs())
                    : mStimuliProvider->getData();

                const Float_T maxVal = std::abs(*Utils::max_abs_element(outputs.begin(), outputs.end()));
                outputsHistogram.insert(std::make_pair(*itCell,
                                                    Histogram(0.0, maxVal)));
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)mLayers.size(); ++i) {
        for (std::vector<std::string>::const_iterator
             itCell = mLayers[i].begin(), itCellEnd = mLayers[i].end();
             itCell != itCellEnd;
             ++itCell)
        {
            std::shared_ptr<Cell_Frame_Top> cellFrame;

            if (mCells.find(*itCell) != mCells.end()) {
                cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(
                                                (*mCells.find(*itCell)).second);
                cellFrame->getOutputs().synchronizeDToH();
            }

            const Tensor<Float_T>& outputs = (cellFrame)
                ? tensor_cast<Float_T>(cellFrame->getOutputs())
                : mStimuliProvider->getData();

            const Float_T maxVal = std::abs(*Utils::max_abs_element(outputs.begin(), outputs.end()));

            const std::map<std::string, Histogram>::iterator itHistogram
                = outputsHistogram.find(*itCell);
            (*itHistogram).second.enlarge(maxVal);
            for(Float_T val: outputs) {
                (*itHistogram).second(std::abs(val));
            }
        }
    }
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
