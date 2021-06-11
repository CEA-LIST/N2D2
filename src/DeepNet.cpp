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
#include "Xnet/Environment.hpp"
#include "Xnet/Monitor.hpp"
#include "Xnet/NodeEnv.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/DeconvCell.hpp"
#include "Cell/DistanceCell.hpp"
#include "Cell/ConvCell_Spike.hpp"
#include "Cell/DropoutCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#include "utils/Utils.hpp"
#include "Solver/Solver.hpp"

N2D2::DeepNet::DeepNet(Network& net)
    : mName(this, "Name", ""),
      mNet(net),
      mLayers(1, std::vector<std::string>(1, "env")),
      mStreamIdx(0),
      mStreamTestIdx(0)
{
    // ctor

#ifdef CUDA
    mLastPass = false;
    mBanAllowed = false;
    mNbPassBeforeBan = 10;
    mAveragePowerUsage = 0;

#ifdef NVML
    // Used to catch the power of every device during learning
    // See N2D2::DeepNet::learn for details
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result)
        std::cout << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
#endif

#endif
}

void N2D2::DeepNet::addCell(const std::shared_ptr<Cell>& cell,
                            const std::vector<std::shared_ptr<Cell>>& parents)
{
    // Check which parent has the largest order in mLayers 
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

    // Add additional layer for new cell if highest order parent is in final layer
    if (cellOrder + 1 >= mLayers.size()) {
        mLayers.resize(cellOrder + 2);
    }

    mLayers[cellOrder + 1].push_back(cell->getName());

    // Add link information to mParentLayers s
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

    // itChildsPos is needed to ensure that the order of the inputs is kept
    // in mParentLayers when first removing and then re-inserting elements.
    // C++11: the function (multimap::insert) optimizes its insertion time if 
    // position points to the element that will follow the inserted element 
    // (or to the end, if it would be the last).
    // WARNING: there is no strong guarantee that the hint, even pointing at the
    // intended place, will be respected, as it is only a "hint". This may be
    // implementation defined.
    // TODO: a refactoring is needed...
    std::vector<std::multimap<std::string, std::string>::iterator> itChildsPos;
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
            std::vector<std::multimap<std::string, std::string>::iterator>
                ::iterator itChild = std::find(itChildsPos.begin(),
                    itChildsPos.end(), itParentLayers);

            itParentLayers = mParentLayers.erase(itParentLayers);

            if (itChild != itChildsPos.end())
                (*itChild) = itParentLayers;
        }
        else if(itParentLayers->second == cellName) {
            children.push_back(itParentLayers->first);
            itParentLayers = mParentLayers.erase(itParentLayers);
            itChildsPos.push_back(itParentLayers);
        }
        else {
            ++itParentLayers;
        }
    }

    if (reconnect) {
        /**
         * Each child of 'cell' has only 'cell' as parent. 
         * Clear the input of each child and connect all the parents of 'cell' to each child.
         */
        if(cell->isFullMap() &&
           std::all_of(children.begin(), children.end(), 
                       [&](const std::string& childName) { 
                           return mParentLayers.count(childName) == 0; 
                        }))
        {
            for(const std::string& childName: children) {
                auto child = mCells.at(childName);
                const Tensor<bool> mapping = child->getMapping().clone();
                child->clearInputs();

                unsigned int nbChannels = 0;

                for (size_t k = 0; k < parents.size(); ++k) {
                    const std::string parentName = parents[k];

                    if (parentName == "env") {
                        const Tensor<bool> parentMapping = mapping.rows(
                            nbChannels, mStimuliProvider->getNbChannels());
                        nbChannels += mStimuliProvider->getNbChannels();

                        child->addInput(*mStimuliProvider, 0, 0,
                            mStimuliProvider->getSizeX(),
                            mStimuliProvider->getSizeY(), parentMapping);
                    }
                    else {
                        auto parentCell = mCells.at(parentName);
                        const Tensor<bool> parentMapping = mapping.rows(
                            nbChannels, parentCell->getNbOutputs());
                        nbChannels += parentCell->getNbOutputs();

                        child->addInput(parentCell.get(), parentMapping);
                    }

                    mParentLayers.emplace(childName, parentName);
                }
            }
        }
        else {
            auto cellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            for(size_t child = 0; child < children.size(); ++child) {
                const std::string childName = children[child];
                auto childCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells.at(childName));

                for(const std::string& parentName: parents) {
                    auto parentCellTop = std::dynamic_pointer_cast<Cell_Frame_Top>(mCells.at(parentName));

                    if (cellTop && childCellTop && parentCellTop) {
                        childCellTop->replaceInput(
                            cellTop->getOutputs(),
                            parentCellTop->getOutputs(),
                            parentCellTop->getDiffInputs());
                    }

                    auto itChildPos = itChildsPos[child];
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
N2D2::DeepNet::getCMonitorOutputsActivities()
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

            std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator
            itMonitor = mCMonitors.find(*itCell);

            if (itMonitor ==  mCMonitors.end())
                continue;

            activity.push_back(std::make_pair(
                  (*itCell), (*itMonitor).second->getIntegratedOutputsActivity()));
           
        }
    }

    return activity;
}



std::vector<std::pair<std::string, long long unsigned int> >
N2D2::DeepNet::getCMonitorFiringRates()
{
    std::vector<std::pair<std::string, long long unsigned int> > activity;

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

            std::map<std::string, std::shared_ptr<CMonitor> >::const_iterator
            itMonitor = mCMonitors.find(*itCell);

            if (itMonitor ==  mCMonitors.end())
                continue;

            activity.push_back(std::make_pair(
                (*itCell), (*itMonitor).second->getIntegratedFiringRate()));
           
        }
    }

    return activity;
}


std::vector<std::pair<std::string, long long unsigned int> >
N2D2::DeepNet::update(bool log, Time_T start, Time_T stop, bool update)
{
    std::vector<std::pair<std::string, long long unsigned int> > activity;

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

            activity.push_back(std::make_pair(
                (*itCell), (*itMonitor).second->getIntegratedFiringRate()));

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
            (*it).second->logActivity("activity_batchElem_0_" + (*it).first + ".dat", 0, true);
            (*it).second->logFiringRate("firing_rate_" + (*it).first + ".dat",true);

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

        (*it).second->exportQuantFreeParameters(dirName + "/" + (*it).first
                                           + "_quant.syntxt");
        (*it).second->logQuantFreeParametersDistrib(dirName + "/" + (*it).first
            + "_weights_quant.distrib.dat", Cell::Multiplicative);

        (*it).second->logFreeParametersDistrib(dirName + "/" + (*it).first
            + "_biases.distrib.dat", Cell::Additive);

        (*it).second->exportActivationParameters(dirName);
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
        (*it).second->importActivationParameters(dirName, ignoreNotExists);

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
        (*it).second->clearAccumulators();
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
        std::cout << "DeepNet::initialize(): " 
                  << " Initialize CEnv environment" << std::endl;
    }

    const std::vector<int> devices(mStimuliProvider->getDevices().begin(),
                                   mStimuliProvider->getDevices().end());

#ifdef CUDA
    int count = 1;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        count = 1;

    mStates.resize(count, N2D2::DeviceState::Excluded);
    mDevicesWarning.resize(count, 0);
    mDropDevices.resize(count, false);
    (mMultiDevicesInfo.finished).resize(count, false);

    int currentDev = 0;
    status = cudaGetDevice(&currentDev);
    if (status != cudaSuccess)
        currentDev = 0;

    mMasterDevice = currentDev;
#endif

    // Enable Peer-to-Peer communications between devices
#ifdef CUDA
    for (int i = 0; i < (int)devices.size(); ++i) {
        for (int j = 0; j < (int)devices.size(); ++j) {
            if (i != j) {
                int canAccessPeer = 0;
                CHECK_CUDA_STATUS(cudaDeviceCanAccessPeer(&canAccessPeer,
                                              devices[j], devices[i]));
                if (canAccessPeer) {
                    CHECK_CUDA_STATUS(cudaSetDevice(devices[j]));
                    CHECK_CUDA_STATUS(cudaDeviceEnablePeerAccess(devices[i], 0));
                }
            }
        }
    }
#endif

    // NOT parallelizable
    for (int dev = 0; dev < (int)devices.size(); ++dev) {
#ifdef CUDA
        mStates[devices[dev]] = N2D2::DeviceState::Connected;
        if (devices.size() > 1 || devices[dev] != currentDev) {
            CHECK_CUDA_STATUS(cudaSetDevice(devices[dev]));
        }
#endif
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

#ifdef CUDA
    if (devices.size() > 1 || (devices.size() > 0 && devices[0] != currentDev)) {
        CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
    }
#endif
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

        if(!bnParents[0] || bnParents[0]->getType() != ConvCell::Type) {
            std::cout << Utils::cnotice << "  cannot fuse BatchNorm \""
                << cell->getName() << "\" because parent cell (\""
                << ((bnParents[0]) ? bnParents[0]->getName() : "env")
                << "\") is not a Conv" << Utils::cdef << std::endl;

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

        if (count > 0)
            meanVariance /= count;
        else {
            std::cout << Utils::cwarning << "    variance < 1e-12 for all"
                " outputs! Is the network correctly trained?"
                << Utils::cdef << std::endl;
        }

        convCellTop->synchronizeToH(false);

        for (std::size_t output = 0; output < convCell->getNbOutputs(); ++output) {
            // Corrected for zero-variance issue:
            // "A Quantization-Friendly Separable Convolution for MobileNets"
            // https://arxiv.org/pdf/1803.08607.pdf
            // to help post-training quantization
            const double factor = bnScales(output)
                / std::sqrt(eps + ((bnVariances(output) > 1.0e-12 || count == 0)
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

        convCellTop->synchronizeToD(true);

        // Replace BatchNorm by Conv for BatchNorm childs
        // and BatchNorm cell removal from DeepNet
        removeCell(cell, true);
    }
}

void N2D2::DeepNet::insertBatchNormAfterConv(bool moveActivation) {
    std::cout << "Insert BatchNorm after Conv..." << std::endl;

    const std::map<std::string, std::shared_ptr<Cell> > cells(mCells);

    for (auto it = cells.begin(); it != cells.end(); ++it) {
        const std::shared_ptr<Cell>& cell = (*it).second;

        if (cell->getType() != ConvCell::Type) {
            continue;
        }

        // check if there is not already a BatchNorm following
        const std::vector<std::shared_ptr<Cell> > convChilds
            = getChildCells(cell->getName());

        bool isBatchNorm = false;

        for (std::vector<std::shared_ptr<Cell> >::const_iterator
            itChild = convChilds.begin(), itChildEnd = convChilds.end();
            itChild != itChildEnd; ++itChild)
        {
            if ((*itChild)->getType() == BatchNormCell::Type) {
                isBatchNorm = true;
                break;
            }
        }

        if (!isBatchNorm) {
            std::cout << "  insert BatchNorm after Conv \"" << cell->getName()
                << "\"" << std::endl;

            const std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
            std::shared_ptr<Activation> activation
                = (moveActivation) ? cellFrame->getActivation()
                                   : std::shared_ptr<Activation>();

            auto reg = Registrar<BatchNormCell>::create<Float_T>(getCellModelType(*cell));
            auto batchNormCell = reg(*this, 
                                    generateNewCellName(cell->getName() + "_batchnorm"), 
                                    cell->getNbOutputs(),
                                    activation);

            if (moveActivation)
                cellFrame->setActivation(std::shared_ptr<Activation>());

            addCellAfter(batchNormCell, cell);
        }
    }
}

void N2D2::DeepNet::fusePadding() {
    std::cout << "Fuse Padding..." << std::endl;

    for (auto it = mCells.begin(); it != mCells.end(); ) {
        const std::shared_ptr<Cell>& cell = (*it).second;
        ++it; // increase it before being potentially invalided by removeCell()

        if (cell->getType() != PaddingCell::Type) {
            continue;
        }

        // check if Conv/Pool are the only childs
        bool fuse = true;
        const std::vector<std::shared_ptr<Cell> > padChilds
            = getChildCells(cell->getName());

        for (auto itChild = padChilds.begin(); itChild != padChilds.end();
            ++itChild)
        {
            if ((*itChild)->getType() != ConvCell::Type
                && (*itChild)->getType() != PoolCell::Type)
            {
                std::cout << Utils::cnotice << "  cannot fuse Padding \""
                    << cell->getName() << "\" because child cells are not all "
                    "Conv/Pool"
                    << Utils::cdef << std::endl;

                fuse = false;
                break;
            }

            // check if childs Conv/Pool have other parents
            const std::vector<std::shared_ptr<Cell> > childParents
                = getParentCells((*itChild)->getName());

            if (childParents.size() > 1) {
                std::cout << Utils::cnotice << "  cannot fuse Padding \""
                    << cell->getName() << "\" because child Conv/Pool "
                    "(\"" << (*itChild)->getName() << "\") has multiple "
                    "parents" << Utils::cdef << std::endl;

                fuse = false;
                break;
            }
        }

        if (!fuse)
            continue;

        std::shared_ptr<PaddingCell> padCell =
            std::dynamic_pointer_cast<PaddingCell>(cell);
        const std::vector<int> paddingDims = { padCell->getLeftPad(),
                                               padCell->getTopPad(),
                                               padCell->getRightPad(),
                                               padCell->getBotPad() };

        // Remove padding cell before setExtendedPadding(), because 
        // setExtendedPadding() needs the correct cell's input dimensions, 
        // which are the dimensions before padding!
        removeCell(cell, true);

        // Here the input dims of childs are correct but their output dims
        // is automatically changed by the reconnection in removeCell() to 
        // take into account the padding removal.
        // At this point the graph is therefore corrupted, because the output
        // dims change is not automatically propagated through the graph.
        // The correct output dims will be recomputed by setExtendedPadding().

        for (auto itChild = padChilds.begin(); itChild != padChilds.end();
            ++itChild)
        {
            if ((*itChild)->getType() == ConvCell::Type) {
                std::shared_ptr<ConvCell> convCell =
                    std::dynamic_pointer_cast<ConvCell>(*itChild);

                std::cout << "  fuse Padding \"" << cell->getName()
                    << "\" with Conv \"" << convCell->getName() << "\""
                    << std::endl;

                convCell->setExtendedPadding(paddingDims);
            }
            else if ((*itChild)->getType() == PoolCell::Type) {
                std::shared_ptr<PoolCell> poolCell =
                    std::dynamic_pointer_cast<PoolCell>(*itChild);

                std::cout << "  fuse Padding \"" << cell->getName()
                    << "\" with Pool \"" << poolCell->getName() << "\""
                    << std::endl;

                poolCell->setExtendedPadding(paddingDims);
            }
        }

        // At this point the graph is correct again.
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
        // Distance
        const std::shared_ptr<DistanceCell> cellDistance
            = std::dynamic_pointer_cast<DistanceCell>(cell);

        if (cellDistance) {
            solvers.push_back(std::make_pair((*it).first + "_centroid",
                                             cellDistance->getWeightsSolver()));
        }
    }

#pragma omp parallel for if (solvers.size() > 1)
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
    if (timings != NULL)
        (*timings).clear();

#ifdef CUDA
    const std::vector<int> devices(mStimuliProvider->getDevices().begin(),
                                   mStimuliProvider->getDevices().end());

    if (devices.size() > 1)
        learn_multiDevices(timings);
    else
        learn_singleDevice(timings);

#else
    learn_singleDevice(timings);
#endif

}

void N2D2::DeepNet::learn_singleDevice(std::vector<std::pair<std::string, double> >* timings)
{
    propagate(Database::Learn, false, timings); 
    backPropagate(timings);
    update(timings);
}

#ifdef CUDA

#ifdef NVML
/** Give the power usage of the target device
 * 
 * @param dev   The identifier of the target device
 * @return      Power usage for the target device in milliwatts
 */
unsigned int devicePowerUsage(int dev)
{
    nvmlDevice_t device;
    unsigned int power = 0;
    nvmlDeviceGetHandleByIndex_v2(dev, &device);
    nvmlDeviceGetPowerUsage(device, &power);

    return power;
}
#endif

/** Launch the phases of propagate and backpropagate. 
 * 
 * The function is used with a thread in DeepNet::learn_multiDevices() to launch 
 * propagation and backpropagation on a target device. 
 * After the backpropagation, there is a check to know if the results 
 * can be saved for the update phase. The check depends on the state of the  
 * target device (more details in DeepNet::learn_multiDevices()) 
 * 
 * @param deepNet   A pointer to the deepNet
 * @param dev       The identifier of the device 
 *                  where the learning is executed
 * @param timings   Vector of time values for benchmarking
 */
void learnThread(const std::shared_ptr<N2D2::DeepNet>& deepNet, 
                 int dev,
                 std::vector<std::pair<std::string, double> >* timings)
{

    CHECK_CUDA_STATUS(cudaSetDevice(dev));

    deepNet->propagate(N2D2::Database::Learn, false, timings);
    deepNet->backPropagate(timings);

    if (!deepNet->isDeviceDropped(dev)) {
        N2D2::DeepNet::SharedValues& sharedVal = deepNet->getMultiDevicesInfo();
        sharedVal.finished[dev] = true;
#ifdef NVML
        sharedVal.power += devicePowerUsage(dev);
#endif
        sharedVal.nbFinished += 1;
        sharedVal.firstFinished = true;
    }
}

void N2D2::DeepNet::learn_multiDevices(std::vector<std::pair<std::string, double> >* timings)
{
    mMultiDevicesInfo.nbFinished = 0;
    mMultiDevicesInfo.power = 0;
    mMultiDevicesInfo.firstFinished = false;

    std::fill((mMultiDevicesInfo.finished).begin(),
              (mMultiDevicesInfo.finished).end(),
              false);

    const unsigned int nbThreads = std::count(mStates.begin(), 
                                            mStates.end(), 
                                            N2D2::DeviceState::Connected);

    const double coeffTime = (nbThreads > 1)
                ? (double)nbThreads / (nbThreads - 1)
                : 1.0;
#ifdef NVML
    const double coeffPower = 0.5;
    /// Limit of consecutive authorized slowdowns
    const unsigned int warningLimit = 3;
#endif

    const std::vector<int> devices(mStimuliProvider->getDevices().begin(),
                                   mStimuliProvider->getDevices().end());

    int currentDev = 0;
    CHECK_CUDA_STATUS(cudaGetDevice(&currentDev));

    std::chrono::high_resolution_clock::time_point startTime 
        = std::chrono::high_resolution_clock::now();

    // Launching threads and testing of banned devices
    for (int dev = 0; dev < (int)devices.size(); ++dev) {
        if (mStates[devices[dev]] == N2D2::DeviceState::Connected) {
            std::thread t(&learnThread, 
                          shared_from_this(),
                          devices[dev],
                          (dev == 0) ? timings : NULL);
            t.detach();
            mDropDevices[devices[dev]] = false;
        }
#ifdef NVML
        else if (mStates[devices[dev]] == N2D2::DeviceState::Banned) {
            if((double)devicePowerUsage(devices[dev]) < mAveragePowerUsage * coeffPower) 
            {
                std::cout << "\nDevice " << devices[dev] << " will be reconnected" << std::endl;
                mStates[devices[dev]] = N2D2::DeviceState::Debanned;
            }  
        }
        else if (mStates[devices[dev]] == N2D2::DeviceState::Debanned)
            mStates[devices[dev]] = N2D2::DeviceState::Ready;
#endif
    }

    CHECK_CUDA_STATUS(cudaSetDevice(currentDev));

    // Waiting for the first device to establish the reference time
    while (!mMultiDevicesInfo.firstFinished)
        std::this_thread::yield();
    
    double refTime = std::chrono::duration_cast <std::chrono::duration<double> > 
                    (std::chrono::high_resolution_clock::now() - startTime).count();
    double newTime = refTime;

    if (mNbPassBeforeBan > 0 || !mBanAllowed || mLastPass) {

        if (mNbPassBeforeBan > 0)
            --mNbPassBeforeBan;

        // Waiting for the end of the threads
        while (mMultiDevicesInfo.nbFinished < nbThreads)
            std::this_thread::yield();

        // Updating the average power usage
        mAveragePowerUsage = mMultiDevicesInfo.power / nbThreads;

    } else {

        // Waiting for the end of the threads or the time limit
        while (mMultiDevicesInfo.nbFinished < nbThreads && newTime < refTime * coeffTime) {
            newTime = std::chrono::duration_cast <std::chrono::duration<double> > 
                    (std::chrono::high_resolution_clock::now() - startTime).count();
            std::this_thread::yield();
        }

        // Time limit reached
        if (newTime >= refTime * coeffTime) {
            for (int dev = 0; dev < (int)devices.size(); ++dev) {
                if (!mMultiDevicesInfo.finished[devices[dev]] 
                    && mStates[devices[dev]] == N2D2::DeviceState::Connected) 
                {
#ifdef NVML
                    if (mDevicesWarning[devices[dev]] >= warningLimit) {
                        mDropDevices[devices[dev]] = true;
                        mStates[devices[dev]] = N2D2::DeviceState::Banned;
                        std::cout << "\nDevice " << devices[dev] << " has been banned" << std::endl;
                    } else {
#endif
                        mDevicesWarning[devices[dev]] += 1;
                        while (!mMultiDevicesInfo.finished[devices[dev]])
                            std::this_thread::yield();
#ifdef NVML
                    }
#endif
                }
            }
            // Changing the master device if this one is banned
            if (mStates[mMasterDevice] == N2D2::DeviceState::Banned) {
                for (int dev = 0; dev < (int)devices.size(); ++dev) {
                    if (mStates[devices[dev]] == N2D2::DeviceState::Connected) {
                        mMasterDevice = devices[dev];
                        break;
                    }
                }
            }
        } else {
            // Updating the average power usage
            mAveragePowerUsage = mMultiDevicesInfo.power / nbThreads;

            // Reset of all warning counters
            std::fill(mDevicesWarning.begin(),
                      mDevicesWarning.end(),
                      0U);
        }
    }

    // During last pass, all banned devices are settled to ready
    // in order to broadcast all changes to all devices
    if (mLastPass) {
        mLastPass = false;
        for (int dev = 0; dev < (int)devices.size(); ++dev) {
            if (mStates[devices[dev]] == N2D2::DeviceState::Banned
                || mStates[devices[dev]] == N2D2::DeviceState::Debanned) 
            {
                mStates[devices[dev]] = N2D2::DeviceState::Ready;
            }
        }
    }
            
    CHECK_CUDA_STATUS(cudaSetDevice(mMasterDevice));
    update(timings);

    // If a device is ready at this point, it must be reconnected
    std::replace(mStates.begin(),
                mStates.end(),
                N2D2::DeviceState::Ready,
                N2D2::DeviceState::Connected);

    CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
}
#endif

/*
void N2D2::DeepNet::learn(std::vector<std::pair<std::string, double> >* timings)
{
    if (timings != NULL)
        (*timings).clear();

    const std::vector<int> devices(mStimuliProvider->getDevices().begin(),
                                   mStimuliProvider->getDevices().end());

    int currentDev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&currentDev));
#endif

#pragma omp parallel for if (devices.size() > 1)
    for (int dev = 0; dev < (int)devices.size(); ++dev) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(devices[dev]));
#endif
        propagate(Database::Learn, false, (dev == 0) ? timings : NULL); 
        backPropagate((dev == 0) ? timings : NULL);
    }

#ifdef CUDA
    CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
#endif
    update(timings);
}
*/

void N2D2::DeepNet::test(
    Database::StimuliSet set,
    std::vector<std::pair<std::string, double> >* timings)
{
    if (timings != NULL)
        (*timings).clear();

    const std::vector<int> devices(mStimuliProvider->getDevices().begin(),
                                   mStimuliProvider->getDevices().end());

#ifdef CUDA
    int currentDev = 0;
    const cudaError_t status = cudaGetDevice(&currentDev);
    if (status != cudaSuccess)
        currentDev = 0;
#endif

#pragma omp parallel for if (devices.size() > 1)
    for (int dev = 0; dev < (int)devices.size(); ++dev) {
#ifdef CUDA
        if (mStates[devices[dev]] == N2D2::DeviceState::Connected) {
            if (devices.size() > 1 || devices[dev] != currentDev) {
                CHECK_CUDA_STATUS(cudaSetDevice(devices[dev]));
            }
            propagate(set, true, (dev == 0) ? timings : NULL);
        }
#else
        propagate(set, true, (dev == 0) ? timings : NULL);
#endif
    }

#ifdef CUDA
    if (devices.size() > 1 || (devices.size() > 0 && devices[0] != currentDev)) {
        CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
    }
#endif
}

void N2D2::DeepNet::propagate(
    Database::StimuliSet set,
    bool inference,
    std::vector<std::pair<std::string, double> >* timings)
{
    const unsigned int nbLayers = mLayers.size();
    std::chrono::high_resolution_clock::time_point time1, time2;

    // Provide targets
    for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
         = mTargets.begin(),
         itTargetsEnd = mTargets.end();
         itTargets != itTargetsEnd;
         ++itTargets)
    {
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
                    "DeepNet::learn(): learning requires Cell_Frame_Top cells");

            time1 = std::chrono::high_resolution_clock::now();
            cellFrame->propagate(inference);

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
        time1 = std::chrono::high_resolution_clock::now();
        (*itTargets)->process(set);

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
}


void N2D2::DeepNet::propagate(bool inference)
{
    const unsigned int nbLayers = mLayers.size();

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

            cellFrame->propagate(inference);
    
        }
    }
}

void N2D2::DeepNet::backPropagate(
    std::vector<std::pair<std::string, double> >* timings)
{
    const unsigned int nbLayers = mLayers.size();
    std::chrono::high_resolution_clock::time_point time1, time2;

    // Error back-propagation
    for (unsigned int l = nbLayers - 1; l > 0; --l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell)
        {

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
}

void N2D2::DeepNet::update(
    std::vector<std::pair<std::string, double> >* timings)
{
    const unsigned int nbLayers = mLayers.size();
    std::chrono::high_resolution_clock::time_point time1, time2;

#ifdef CUDA
    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    // Weights update
    for (unsigned int l = 1; l < nbLayers; ++l) {
        for (std::vector<std::string>::const_iterator itCell
             = mLayers[l].begin(),
             itCellEnd = mLayers[l].end();
             itCell != itCellEnd;
             ++itCell)
        {
      
            time1 = std::chrono::high_resolution_clock::now();
#ifdef CUDA
            //update states
            std::dynamic_pointer_cast
                <Cell_Frame_Top>(mCells[(*itCell)])->updateDeviceStates(mStates);
#endif
            std::dynamic_pointer_cast
                <Cell_Frame_Top>(mCells[(*itCell)])->update();

#ifdef CUDA
            // MultiGPU issue
            // After BatchNorm layer update, the master changes
            // Thus, this line is to fix this issue
            CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

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

            (*itMonitor).second->initialize(nbTimesteps);
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

std::string N2D2::DeepNet::getCellModelType(const Cell& cell) {
    const Cell_Frame_Top& cellFrameTop = dynamic_cast<const Cell_Frame_Top&>(cell);
    if(cellFrameTop.isCuda()) {
        return Cell_Frame_Top::FRAME_CUDA_TYPE;
    }
    else {
        return Cell_Frame_Top::FRAME_TYPE;
    }
}

N2D2::DeepNet::~DeepNet()
{
#ifdef CUDA

#ifdef NVML
    // Used to catch the power of every device during learning
    // See N2D2::DeepNet::learn_multiDevices for details
    nvmlReturn_t result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        std::cout << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
#endif

#endif
}


