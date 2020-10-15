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

#include "Generator/TargetGenerator.hpp"
#include "Target/TargetROIs.hpp"
#include "Target/TargetAggregate.hpp"

std::shared_ptr<N2D2::Target>
N2D2::TargetGenerator::generate(const std::shared_ptr<Cell>& cell,
                                const std::shared_ptr<DeepNet>& deepNet,
                                IniParser& iniConfig,
                                const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string type = iniConfig.getProperty
                                   <std::string>("Type", "TargetScore");
    const double targetValue = iniConfig.getProperty
                               <double>("TargetValue", 1.0);
    const double defaultValue = iniConfig.getProperty
                                <double>("DefaultValue", 0.0);
    const unsigned int targetTopN = iniConfig.getProperty
                                    <unsigned int>("TopN", 1U);
    const std::string labelsMapping = Utils::expandEnvVars(
        iniConfig.getProperty<std::string>("LabelsMapping", ""));
    const bool createMissingLabels
        = iniConfig.getProperty<bool>("CreateMissingLabels", false);

    // These options are processed afterwards in postGenerate()
    iniConfig.getProperty<std::string>("ROIsLabelTarget", "");
    iniConfig.getProperty<std::string>("MaskLabelTarget", "");

    std::cout << "Target: " << cell->getName()
              << " (target value: " << targetValue
              << " / default value: " << defaultValue
              << " / top-n value: " << targetTopN << ")" << std::endl;

    std::shared_ptr<Target> target = Registrar
        <Target>::create(type)(section,
                                     cell,
                                     deepNet->getStimuliProvider(),
                                     targetValue,
                                     defaultValue,
                                     targetTopN,
                                     labelsMapping,
                                     createMissingLabels);

    if (type == "TargetRP") {
        const TargetRP::TargetType targetType = iniConfig.getProperty
                                       <TargetRP::TargetType>("TargetType");
        const std::string RPCellName = iniConfig.getProperty
                                        <std::string>("RP");
        const std::string anchorCellName = iniConfig.getProperty
                                        <std::string>("Anchor");

        std::shared_ptr<RPCell> RPCellPtr;
        std::shared_ptr<AnchorCell> anchorCellPtr;

        for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator
             itCells = deepNet->getCells().begin(),
             itCellsEnd = deepNet->getCells().end();
             itCells != itCellsEnd;
             ++itCells)
        {
            if ((*itCells).first == RPCellName) {
                RPCellPtr
                    = std::dynamic_pointer_cast<RPCell>((*itCells).second);
            }
            else if ((*itCells).first == anchorCellName) {
                anchorCellPtr
                    = std::dynamic_pointer_cast<AnchorCell>((*itCells).second);
            }
        }

        if (!RPCellPtr) {
            throw std::runtime_error("RPCell name \"" + RPCellName
                                     + "\" not found for TargetRP ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }

        if (!anchorCellPtr) {
            throw std::runtime_error("AnchorCell name \"" + anchorCellName
                                     + "\" not found for TargetRP ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }

        std::static_pointer_cast<TargetRP>(target)->initialize(targetType,
                                                               RPCellPtr,
                                                               anchorCellPtr);
    }
    else if (type == "TargetBBox")
    {

        const bool generateAnchors = iniConfig.getProperty<bool>("GenerateAnchors", false);
        const unsigned int nbAnchors = iniConfig.getProperty <unsigned int>("NbAnchors", 6);
        const long unsigned int nbMaxIter = iniConfig.getProperty <long unsigned int>("NbIter", 1000000);
        const unsigned int nbClass = iniConfig.getProperty <unsigned int>("NbClass", 1);

        std::static_pointer_cast<TargetBBox>(target)->initialize(generateAnchors, 
                                                                 nbAnchors, 
                                                                 nbMaxIter,
                                                                 nbClass);
    }

    target->setParameters(iniConfig.getSection(section, true));
    return target;
}

void N2D2::TargetGenerator::postGenerate(const std::shared_ptr<Target>& target,
                                         const std::shared_ptr
                                         <DeepNet>& deepNet,
                                         IniParser& iniConfig,
                                         const std::string& section)
{
    if (!iniConfig.currentSection(section))
        return;

    if ((target->getType() == std::string("TargetROIs")
        || target->getType() == std::string("TargetAggregate"))
        && iniConfig.isProperty("ROIsLabelTarget"))
    {
        std::shared_ptr<TargetROIs> targetROIs
            = std::dynamic_pointer_cast<TargetROIs>(target);
        std::shared_ptr<TargetAggregate> targetAggregate
            = std::dynamic_pointer_cast<TargetAggregate>(target);

        const std::string labelTarget = iniConfig.getProperty
                                        <std::string>("ROIsLabelTarget");
        bool found = false;

        for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
             = deepNet->getTargets().begin(),
             itTargetsEnd = deepNet->getTargets().end();
             itTargets != itTargetsEnd;
             ++itTargets)
        {
            if ((*itTargets)->getName() == labelTarget) {
                if (targetROIs)
                    targetROIs->setROIsLabelTarget(*itTargets);
                else if (targetAggregate)
                    targetAggregate->setROIsLabelTarget(*itTargets);

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Target name \"" + labelTarget
                                     + "\" not found for ROIsLabelTarget ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }

    if (iniConfig.isProperty("MaskLabelTarget")) {
        const std::string labelTarget = iniConfig.getProperty
                                        <std::string>("MaskLabelTarget");
        bool found = false;

        for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
             = deepNet->getTargets().begin(),
             itTargetsEnd = deepNet->getTargets().end();
             itTargets != itTargetsEnd;
             ++itTargets) {
            if ((*itTargets)->getName() == labelTarget) {
                target->setMaskLabelTarget(*itTargets);
                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Target name \"" + labelTarget
                                     + "\" not found for MaskLabelTarget ["
                                     + section
                                     + "] in network configuration file: "
                                     + iniConfig.getFileName());
        }
    }
}