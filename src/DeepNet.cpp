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
                            const std::vector<std::shared_ptr<Cell>>& parents)
{
    unsigned int cellOrder = 0;
    for (auto it = mLayers.begin(); it != mLayers.end(); ++it) {
        for (auto itParent = parents.begin(); itParent != parents.end(); ++itParent) {
            if (*itParent) {
                auto itCell = std::find(it->begin(), it->end(), (*itParent)->getName());

                if (itCell != it->end()) {
                    cellOrder = std::max(cellOrder, (unsigned int)(it - mLayers.begin()));
                }
            }
        }
    }

    if (cellOrder + 1 >= mLayers.size()) {
        mLayers.resize(cellOrder + 2);
    }

    mLayers[cellOrder + 1].push_back(cell->getName());

    for (auto itParent = parents.begin(); itParent != parents.end(); ++itParent) {
        if (*itParent) {
            mParentLayers.insert(std::make_pair(cell->getName(), (*itParent)->getName()));
        }
        else {
            mParentLayers.insert(std::make_pair(cell->getName(), "env"));
        }
    }

    mCells.insert(std::make_pair(cell->getName(), cell));
}

void N2D2::DeepNet::addCellBetween(const std::shared_ptr<Cell>& newCell,
                                   const std::shared_ptr<Cell>& parent,
                                   const std::shared_ptr<Cell>& child)
{
    auto parentChildren = parent->getChildrenCells();
    if(std::find(parentChildren.begin(), parentChildren.end(), child) == parentChildren.end()) {
        throw std::runtime_error("The cell '" + parent->getName() + "' isn't a parent of the cell '" + 
                                  child->getName() + "'.");
    }

    auto childParents = child->getParentsCells();
    if(std::find(childParents.begin(), childParents.end(), parent) == childParents.end()) {
        throw std::runtime_error("The cell '" + child->getName() + "' isn't a child of the cell '" + 
                                  parent->getName() + "'.");
    }

    /**
     * mCells 
     */
    mCells.insert(std::make_pair(newCell->getName(), newCell));

    /**
     * mParentLayers
     */
    std::multimap<std::string, std::string>::iterator itChildPos = mParentLayers.end();

    // Remove child -> parent link
    auto itChildParents = mParentLayers.equal_range(child->getName());
    while(itChildParents.first != itChildParents.second) {
        if(itChildParents.first->second == parent->getName()) {
            itChildPos = mParentLayers.erase(itChildParents.first);
            break;
        }

        ++itChildParents.first;
    }

    // Add newCell -> parent link
    mParentLayers.insert(std::make_pair(newCell->getName(), parent->getName()));

    // Add child -> newCell link
    mParentLayers.insert(itChildPos, std::make_pair(child->getName(), newCell->getName()));

    /**
     * mLayers
     */
    auto itParentLayer = mLayers.end();
    for(auto itLayer = mLayers.begin(); itLayer != mLayers.end(); ++itLayer) {
        auto itCell = std::find(itLayer->begin(), itLayer->end(), parent->getName());
        if(itCell != itLayer->end()) {
            itParentLayer = itLayer;
            break;
        }
    }

    if(itParentLayer == mLayers.end()) {
        throw std::runtime_error("The cell '" + parent->getName() + "' was not found in the graph.");
    }

    mLayers.insert(itParentLayer + 1, std::vector<std::string>(1, newCell->getName()));

    /**
     * Cells inputs
     */
    auto newCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(newCell);
    auto parentCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(parent);
    auto childCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(child);

    newCell->addInput(parent.get());
    childCellTop->replaceInput(parentCellTop->getOutputs(), newCellTop->getOutputs(), 
                               newCellTop->getDiffInputs());
}

void N2D2::DeepNet::addCellAfter(const std::shared_ptr<Cell>& newCell,
                                 const std::shared_ptr<Cell>& parent)
{
    /**
     * mCells 
     */
    mCells.insert(std::make_pair(newCell->getName(), newCell));

    newCell->addInput(parent.get());
    
    /**
     * mParentLayers and cells inputs connections
     */
    auto parentCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(parent);
    auto newCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(newCell);

    // Change the parent of the childrenn of the cell 'parent' to 'newCell'
    for(const auto& child: getChildCells(parent->getName())) {
        std::multimap<std::string, std::string>::iterator itChildPos = mParentLayers.end();

        // Remove child->parent link which is the pair {child->getName(), parent->getName()}
        auto parents = mParentLayers.equal_range(child->getName());
        while(parents.first != parents.second) {
            if(parents.first->second == parent->getName()) {
                itChildPos = mParentLayers.erase(parents.first);
                break;
            }

            ++parents.first;
        }
        

        // Add new child->newCell link
        mParentLayers.insert(itChildPos, std::make_pair(child->getName(), newCell->getName()));

        auto childCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(child);
        childCellTop->replaceInput(parentCellTop->getOutputs(), newCellTop->getOutputs(), 
                                   newCellTop->getDiffInputs());
    }

    mParentLayers.insert(std::make_pair(newCell->getName(), parent->getName()));

    
    /**
     * mLayers
     */
    auto itParentLayer = mLayers.end();
    for(auto itLayer = mLayers.begin(); itLayer != mLayers.end(); ++itLayer) {
        auto itCell = std::find(itLayer->begin(), itLayer->end(), parent->getName());
        if(itCell != itLayer->end()) {
            itParentLayer = itLayer;
            break;
        }
    }

    if(itParentLayer == mLayers.end()) {
        throw std::runtime_error("The cell '" + parent->getName() + "' was not found in the graph.");
    }

    mLayers.insert(itParentLayer + 1, std::vector<std::string>(1, newCell->getName()));
}

void N2D2::DeepNet::addCellBefore(const std::shared_ptr<Cell>& newCell,
                                  const std::shared_ptr<Cell>& child)
{
    /**
     * mCells 
     */
    mCells.insert(std::make_pair(newCell->getName(), newCell));
    
    /**
     * mParentLayers and cells inputs connections
     */
    // Add parent links between 'newCell' and the parent of 'child'.
    for(auto& parent: child->getParentsCells()) {
        mParentLayers.emplace(newCell->getName(), parent->getName());
        newCell->addInput(parent.get());
    }

    // Set the parent link of 'child' to 'newCell'.
    mParentLayers.erase(child->getName());
    mParentLayers.emplace(child->getName(), newCell->getName());

    child->clearInputs();
    child->addInput(newCell.get());

    /**
     * mLayers
     */
    auto itChildLayer = mLayers.end();
    for(auto itLayer = mLayers.begin(); itLayer != mLayers.end(); ++itLayer) {
        auto itCell = std::find(itLayer->begin(), itLayer->end(), child->getName());
        if(itCell != itLayer->end()) {
            itChildLayer = itLayer;
            break;
        }
    }

    if(itChildLayer == mLayers.end()) {
        throw std::runtime_error("The cell '" + child->getName() + "' was not found in the graph.");
    }

    mLayers.insert(itChildLayer, std::vector<std::string>(1, newCell->getName()));
}

void N2D2::DeepNet::removeCell(const std::shared_ptr<Cell>& cell,
                               bool reconnect)
{
    // TODO Refactorize and simplify the method.
    const std::string cellName = cell->getName();

    std::multimap<std::string, std::string>::iterator itChildPos = mParentLayers.end();
    std::vector<std::string> parents;
    std::vector<std::string> children;

    for (auto itParentLayers = mParentLayers.begin(); itParentLayers != mParentLayers.end(); ) {
        if (itParentLayers->first == cellName) {
            parents.push_back(itParentLayers->second);

            // The next element after a child in the multimap might be a parent,
            // depending on the name of the cells.
            // In this case, itChildPos will be invalided. If this happens, we
            // provide an invalid hint to multimap::insert(), which causes
            // undefined behavior and corrupt the multimap.
            if (itChildPos == itParentLayers)
                itChildPos = mParentLayers.end();

            itParentLayers = mParentLayers.erase(itParentLayers);
        }
        else if(itParentLayers->second == cellName) {
            children.push_back(itParentLayers->first);
            itParentLayers = mParentLayers.erase(itParentLayers);
            itChildPos = itParentLayers;
        }
        else {
            ++itParentLayers;
        }
    }

    if (reconnect) {
        /**
         * Each child of 'cell' has only 'cell' as parent and is completely connected to its parent. 
         * Clear the input of each child and connect all the parents of 'cell' to each child.
         */
        if(cell->isFullMap() &&
           std::all_of(children.begin(), children.end(), 
                       [&](const std::string& childName) { 
                           return mParentLayers.count(childName) == 0 && 
                                  mCells.at(childName)->isFullMap(); 
                        }))
        {
            for(const std::string& childName: children) {
                auto child = mCells.at(childName);
                child->clearInputs();

                for(const std::string& parentName: parents) {
                    if (parentName == "env") {
                        child->addInput(*mStimuliProvider, 0, 0,
                            mStimuliProvider->getSizeX(),
                            mStimuliProvider->getSizeY());
                    }
                    else {
                        auto parent = mCells.at(parentName);

                        child->addInput(parent.get());
                    }

                    mParentLayers.emplace(childName, parentName);
                }
            }
        }
        else {
            auto cellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            for(const std::string& childName: children) {
                auto childCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells.at(childName));

                for(const std::string& parentName: parents) {
                    auto parentCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells.at(parentName));

                    if (cellTop && childCellTop && parentCellTop) {
                        childCellTop->replaceInput(
                            cellTop->getOutputs(),
                            parentCellTop->getOutputs(),
                            parentCellTop->getDiffInputs());
                    }

                    mParentLayers.insert(itChildPos, std::make_pair(childName, parentName));
                }
            }
        }
    }

    mCells.erase(cellName);

    for(std::size_t l = 1; l < mLayers.size(); ) {
        mLayers[l].erase(std::remove(mLayers[l].begin(), mLayers[l].end(), cellName),
                         mLayers[l].end());

        if (mLayers[l].empty()) {
            mLayers.erase(mLayers.begin() + l);
        }
        else {
            ++l;
        }
    }
}

std::string N2D2::DeepNet::generateNewCellName(const std::string& baseName) const {
    if(!hasCell(baseName)) {
        return baseName;
    }

    std::string newCellName;

    std::size_t i = 0;
    do {
        newCellName = baseName + "_" + std::to_string(i);
        i++;
    } while(hasCell(newCellName));

    return newCellName;
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
            + "_weights.distrib.dat", Cell::Multiplicative);
        (*it).second->logFreeParametersDistrib(dirName + "/" + (*it).first
            + "_biases.distrib.dat", Cell::Additive);
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

bool N2D2::DeepNet::hasCell(const std::string& name) const {
    return mCells.find(name) != mCells.end();
}

std::vector<std::shared_ptr<N2D2::Cell>> N2D2::DeepNet::getChildCells(const std::string& name) const
{
    std::vector<std::shared_ptr<Cell>> childCells;

    for(auto it = mParentLayers.begin(); it != mParentLayers.end(); ++it) {
        if (it->second == name) {
            childCells.push_back(mCells.at(it->first));
        }
    }

    return childCells;
}

std::vector<std::shared_ptr<N2D2::Cell>> N2D2::DeepNet::getParentCells(const std::string& name) const
{
    std::vector<std::shared_ptr<Cell>> parentCells;

    auto parents = mParentLayers.equal_range(name);
    for(auto itParent = parents.first; itParent != parents.second; ++itParent) {
        if (itParent->second == "env") {
            parentCells.push_back(std::shared_ptr<Cell>());
        }
        else {
            parentCells.push_back(mCells.at(itParent->second));
        }
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

/**
 * Ref: https://tkv.io/posts/fusing-batchnorm-and-conv/
*/
void N2D2::DeepNet::fuseBatchNormWithConv() {
    std::cout << "Fuse BatchNorm with Conv..." << std::endl;

    for (auto it = mCells.begin(); it != mCells.end(); ) {
        const std::shared_ptr<Cell>& cell = (*it).second;
        ++it; // increase it before being potentially invalided by removeCell()

        if (cell->getType() != BatchNormCell::Type) {
            continue;
        }

        // check if a Conv is preceding
        const std::vector<std::shared_ptr<Cell> > bnParents = getParentCells(cell->getName());

        if(bnParents.size() > 1) {
            std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                << cell->getName() << "\" because it has multiple "
                "parents (not supported)" << Utils::cdef << std::endl;
            
            continue;
        }
        
        if(bnParents[0]->getType() != ConvCell::Type) {
            std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                << cell->getName() << "\" because parent cell (\""
                << bnParents[0]->getName() << "\") is not a Conv"
                << Utils::cdef << std::endl;
            
            continue;
        }

        // only a single Conv is preceding
        // check if BatchNorm is the only child
        const std::vector<std::shared_ptr<Cell>> convChilds
            = getChildCells(bnParents[0]->getName());

        if (convChilds.size() != 1) {
            std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                << cell->getName() << "\" because parent Conv "
                "(\"" << bnParents[0]->getName() << "\") has multiple "
                "childs" << Utils::cdef << std::endl;
            
            continue;
        }

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
            && std::string(convCellTop->getActivation()->getType()) != "Linear")
        {
            std::cout << Utils::cwarning << "  -> non-linear "
                "activation before BatchNorm prevents fuse!"
                << Utils::cdef << std::endl;
            continue;
        }

        convCellTop->setActivation(bnCellTop->getActivation());

        if (noBias)
            convCell->setParameter<bool>("NoBias", false);

        double meanVariance = 0.0;
        unsigned int count = 0;

        for (std::size_t output = 0; output < convCell->getNbOutputs(); ++output) {
            if (bnVariances(output) > 1.0e-12) {
                meanVariance += bnVariances(output);
                ++count;
            }
            else {
                std::cout << "    zero-variance " << convCell->getName()
                    << "[" << output << "]" << std::endl;
            }
        }

        meanVariance /= count;

        for (std::size_t output = 0; output < convCell->getNbOutputs(); ++output) {
            // Corrected for zero-variance issue:
            // "A Quantization-Friendly Separable Convolution for MobileNets"
            // https://arxiv.org/pdf/1803.08607.pdf
            // to help post-training quantization
            const double factor = bnScales(output)
                / std::sqrt(eps + ((bnVariances(output) > 1.0e-12)
                            ? bnVariances(output) : meanVariance));

            // Weights adjustments
            for (std::size_t channel = 0; channel < convCell->getNbChannels(); ++channel) {
                Tensor<double> kernel;
                convCell->getWeight(output, channel, kernel);

                for (std::size_t index = 0; index < kernel.size(); ++index) {
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


    StimuliProvider::logData(dirName + "/env.dat",
                             mStimuliProvider->getData()[batchPos]);

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

    // Provide targets
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets)
    {
        //std::cout << "process " << (*itTargets)->getName() << std::endl;
        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->provideTargets(Database::Learn);

        if (timings != NULL) {
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
            time2 = std::chrono::high_resolution_clock::now();
            (*timings).push_back(std::make_pair(
                (*itTargets)->getCell()->getName() + "."
                + (*itTargets)->getType() + "[provide]",
                std::chrono::duration_cast
                <std::chrono::duration<double> >(time2 - time1).count()));
        }
    }

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

    // Targets processing
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
                + (*itTargets)->getType() + "[process]",
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

    // Provide targets
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets)
    {
        //std::cout << "process " << (*itTargets)->getName() << std::endl;
        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->provideTargets(set);

        if (timings != NULL) {
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
            time2 = std::chrono::high_resolution_clock::now();
            (*timings).push_back(std::make_pair(
                (*itTargets)->getCell()->getName() + "."
                + (*itTargets)->getType() + "[provide]",
                std::chrono::duration_cast
                <std::chrono::duration<double> >(time2 - time1).count()));
        }
    }

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


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_DeepNet(py::module &m) {
    py::class_<DeepNet, std::shared_ptr<DeepNet>>(m, "DeepNet", py::multiple_inheritance())
    .def(py::init<Network&>(), py::arg("net"))
    .def("addCell", &DeepNet::addCell, py::arg("cell"), py::arg("parents"))
    .def("removeCell", &DeepNet::removeCell, py::arg("cell"), py::arg("reconnect") = true)
    .def("addTarget", &DeepNet::addTarget, py::arg("cell"))
    .def("addMonitor", &DeepNet::addMonitor, py::arg("name"), py::arg("monitor"))
    .def("addCMonitor", &DeepNet::addCMonitor, py::arg("name"), py::arg("monitor"))
    .def("update", &DeepNet::update, py::arg("log"), py::arg("start"), py::arg("stop") = 0, py::arg("update") = true)
    .def("save", &DeepNet::save, py::arg("dirName"))
    .def("load", &DeepNet::load, py::arg("dirName"))
    .def("saveNetworkParameters", &DeepNet::saveNetworkParameters)
    .def("loadNetworkParameters", &DeepNet::loadNetworkParameters)
    .def("exportNetworkFreeParameters", &DeepNet::exportNetworkFreeParameters, py::arg("dirName"))
    .def("exportNetworkSolverParameters", &DeepNet::exportNetworkSolverParameters, py::arg("dirName"))
    .def("importNetworkFreeParameters", (void (DeepNet::*)(const std::string&, bool)) &DeepNet::importNetworkFreeParameters, py::arg("dirName"), py::arg("ignoreNotExists") = false)
    .def("importNetworkFreeParameters", (void (DeepNet::*)(const std::string&, const std::string&)) &DeepNet::importNetworkFreeParameters, py::arg("dirName"), py::arg("weightName"))
    //.def("importNetworkSolverParameters", &DeepNet::importNetworkSolverParameters, py::arg("dirName"))
    .def("checkGradient", &DeepNet::checkGradient, py::arg("epsilon") = 1.0e-4, py::arg("maxError") = 1.0e-6)
    .def("initialize", &DeepNet::initialize)
    .def("learn", &DeepNet::learn, py::arg("timings") = NULL)
    .def("test", &DeepNet::test, py::arg("set"), py::arg("timings") = NULL)
    .def("cTicks", &DeepNet::cTicks, py::arg("start"), py::arg("stop"), py::arg("timestep"), py::arg("record") = false)
    .def("cTargetsProcess", &DeepNet::cTargetsProcess, py::arg("set"))
    .def("cReset", &DeepNet::cReset, py::arg("timestamp") = 0)
    .def("initializeCMonitors", &DeepNet::initializeCMonitors, py::arg("nbTimesteps"))
    .def("spikeCodingCompare", &DeepNet::spikeCodingCompare, py::arg("dirName"), py::arg("idx"))
    .def("fuseBatchNormWithConv", &DeepNet::fuseBatchNormWithConv)
    .def("removeDropout", &DeepNet::removeDropout)
    .def("setDatabase", &DeepNet::setDatabase, py::arg("database"))
    .def("setStimuliProvider", &DeepNet::setStimuliProvider, py::arg("sp"))
    .def("getDatabase", &DeepNet::getDatabase)
    .def("getStimuliProvider", &DeepNet::getStimuliProvider)
    .def("getCells", &DeepNet::getCells)
    .def("getMonitor", &DeepNet::getMonitor, py::arg("name"))
    .def("getCMonitor", &DeepNet::getCMonitor, py::arg("name"))
    .def("getLayers", &DeepNet::getLayers)
    .def("getLayer", &DeepNet::getLayer, py::arg("layer"))
    .def("getChildCells", &DeepNet::getChildCells, py::arg("name"))
    .def("getParentCells", &DeepNet::getParentCells, py::arg("name"))
    .def("getTargets", &DeepNet::getTargets)
    .def("getStats", &DeepNet::getStats)
    //.def("getReceptiveField", &DeepNet::getReceptiveField, py::arg("name"), py::arg("outputField") = std::vector<unsigned int>())
    .def("clearAll", &DeepNet::clearAll)
    .def("clearActivity", &DeepNet::clearActivity)
    .def("clearFiringRate", &DeepNet::clearFiringRate)
    .def("clearSuccess", &DeepNet::clearSuccess)
    .def("clear", &DeepNet::clear, py::arg("set"))
    .def("logOutputs", &DeepNet::logOutputs, py::arg("dirName"), py::arg("batchPos") = 0)
    .def("logDiffInputs", &DeepNet::logDiffInputs, py::arg("dirName"), py::arg("batchPos") = 0)
    .def("logFreeParameters", &DeepNet::logFreeParameters, py::arg("dirName"))
    .def("logSchedule", &DeepNet::logSchedule, py::arg("dirName"))
    .def("logStats", &DeepNet::logStats, py::arg("dirName"))
    .def("logSpikeStats", &DeepNet::logSpikeStats, py::arg("dirName"), py::arg("nbPatterns"))
    .def("log", &DeepNet::log, py::arg("baseName"), py::arg("set"))
    .def("logLabelsMapping", &DeepNet::logLabelsMapping, py::arg("fileName"))
    .def("logEstimatedLabels", &DeepNet::logEstimatedLabels, py::arg("dirName"))
    .def("logEstimatedLabelsJSON", &DeepNet::logEstimatedLabelsJSON, py::arg("dirName"))
    .def("logLabelsLegend", &DeepNet::logLabelsLegend, py::arg("fileName"))
    .def("logTimings", &DeepNet::logTimings, py::arg("fileName"), py::arg("timings"))
    .def("logReceptiveFields", &DeepNet::logReceptiveFields, py::arg("fileName"));
}
}
#endif
