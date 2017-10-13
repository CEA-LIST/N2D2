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

#include "Generator/AnchorCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::AnchorCellGenerator::mRegistrar(AnchorCell::Type,
                                       N2D2::AnchorCellGenerator::generate);

std::shared_ptr<N2D2::AnchorCell>
N2D2::AnchorCellGenerator::generate(Network& /*network*/,
                                     StimuliProvider& sp,
                                     const std::vector
                                     <std::shared_ptr<Cell> >& parents,
                                     IniParser& iniConfig,
                                     const std::string& section)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string model = iniConfig.getProperty<std::string>(
        "Model", CellGenerator::mDefaultModel);

    std::cout << "Layer: " << section << " [Anchor(" << model << ")]"
              << std::endl;

    std::vector<AnchorCell_Frame_Kernels::Anchor> anchors;

    const AnchorCell_Frame_Kernels::Anchor::Anchoring anchoring
        = iniConfig.getProperty<AnchorCell_Frame_Kernels::Anchor::Anchoring>
            ("Anchoring", AnchorCell_Frame_Kernels::Anchor::Anchoring::TopLeft);

    // First method: specify anchor by anchor with (root area, ratio) pairs
    unsigned int nextAnchor = 0;
    std::stringstream nextProperty;
    nextProperty << "Anchor[" << nextAnchor << "]";

    while (iniConfig.isProperty(nextProperty.str())) {
        std::stringstream anchorValues(
            iniConfig.getProperty<std::string>(nextProperty.str()));

        unsigned int rootArea;
        double ratio;

        if (!(anchorValues >> rootArea) || !(anchorValues >> ratio)) {
            throw std::runtime_error(
                "Unreadable anchor in section [" + section
                + "] in network configuration file: "
                + iniConfig.getFileName());
        }

        anchors.push_back(AnchorCell_Frame_Kernels::Anchor(rootArea*rootArea,
                                                           ratio,
                                                           1.0,
                                                           anchoring));

        ++nextAnchor;
        nextProperty.str(std::string());
        nextProperty << "Anchor[" << nextAnchor << "]";
    }

    // Second method: specify a base root area and a list of ratios and scales
    // Both methods can be used simultaneously
    const double rootArea = iniConfig.getProperty<double>("RootArea", 16);
    const std::vector<double> ratios = iniConfig.getProperty
        <std::vector<double> >("Ratios", std::vector<double>());
    const std::vector<double> scales = iniConfig.getProperty
        <std::vector<double> >("Scales", std::vector<double>(1, 1.0));

    for (std::vector<double>::const_iterator itRatios = ratios.begin(),
        itRatiosEnd = ratios.end(); itRatios != itRatiosEnd; ++itRatios)
    {
        for (std::vector<double>::const_iterator itScales = scales.begin(),
            itScalesEnd = scales.end(); itScales != itScalesEnd; ++itScales)
        {
            anchors.push_back(AnchorCell_Frame_Kernels::Anchor(
                rootArea*rootArea,
                (*itRatios),
                (*itScales),
                anchoring));
        }
    }

    const unsigned int scoresCls = iniConfig.getProperty
                                     <unsigned int>("ScoresCls", 1);

    // Cell construction
    std::shared_ptr<AnchorCell> cell = Registrar
        <AnchorCell>::create(model)(section, sp, anchors, scoresCls);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }

    // Set configuration parameters defined in the INI file
    cell->setParameters(getConfig(model, iniConfig));

    // Load configuration file (if exists)
    cell->loadParameters(section + ".cfg", true);

    const unsigned int x0 = iniConfig.getProperty
                            <unsigned int>("InputOffsetX", 0);
    const unsigned int y0 = iniConfig.getProperty
                            <unsigned int>("InputOffsetY", 0);
    const unsigned int width = iniConfig.getProperty
                               <unsigned int>("InputWidth", 0);
    const unsigned int height = iniConfig.getProperty
                                <unsigned int>("InputHeight", 0);

    // Connect the cell to the parents
    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
         = parents.begin(),
         itEnd = parents.end();
         it != itEnd;
         ++it) {
        if (!(*it))
            cell->addInput(sp, x0, y0, width, height);
        else
            cell->addInput((*it).get(), x0, y0, width, height);
    }

    std::cout << "  # Anchors: " << anchors.size() << std::endl;
    std::cout << "  # Outputs size: " << cell->getOutputsWidth() << "x"
              << cell->getOutputsHeight() << std::endl;
    std::cout << "  # Outputs: " << cell->getNbOutputs() << std::endl;

    return cell;
}
