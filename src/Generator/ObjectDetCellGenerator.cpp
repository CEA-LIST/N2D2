/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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
#include "Generator/ObjectDetCellGenerator.hpp"

N2D2::Registrar<N2D2::CellGenerator>
N2D2::ObjectDetCellGenerator::mRegistrar(ObjectDetCell::Type,
                                        N2D2::ObjectDetCellGenerator::generate);

std::shared_ptr<N2D2::ObjectDetCell>
N2D2::ObjectDetCellGenerator::generate(Network& /*network*/, const DeepNet& deepNet,
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

    const unsigned int nbAnchors = iniConfig.getProperty
                                   <unsigned int>("NbAnchors");
    const AnchorCell_Frame_Kernels::Format inputFormat 
        = iniConfig.getProperty<AnchorCell_Frame_Kernels::Format>
            ("InputFormat", AnchorCell_Frame_Kernels::Format::CA);         
    const AnchorCell_Frame_Kernels::PixelFormat pixelFormat 
        = iniConfig.getProperty<AnchorCell_Frame_Kernels::PixelFormat>
            ("PixelFormat", AnchorCell_Frame_Kernels::PixelFormat::XY);         

    const unsigned int nbProposals = iniConfig.getProperty
                                   <unsigned int>("NbProposals");
    const unsigned int nbCls = iniConfig.getProperty
                                   <unsigned int>("NbClass");

    const Float_T nmsThreshold = iniConfig.getProperty
                                   <Float_T>("NMS_Threshold", 0.5);

    std::vector<Float_T> scoreThresholds;

    if (iniConfig.isProperty("Score_Threshold")) {
        scoreThresholds = iniConfig.getProperty
                                     <std::vector<Float_T> >("Score_Threshold");
    }

    if(scoreThresholds.size() > nbCls)
    {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [Score_Threshold] in network configuration file: " 
            + iniConfig.getFileName());
    }

    if(scoreThresholds.size() == 1)
        scoreThresholds.resize(nbCls, scoreThresholds[0]);

    const std::vector<unsigned int> partsPerCls = iniConfig.getProperty
                <std::vector<unsigned int> >("NumParts", std::vector<unsigned int>(0, 0));

    const std::vector<unsigned int> templatesPerCls = iniConfig.getProperty
                <std::vector<unsigned int> >("NumTemplates", std::vector<unsigned int>(0, 0));

    unsigned int nbOutputParts = 0;
    unsigned int nbOutputTemplates = 0;

    if(partsPerCls.size() > 0)
        nbOutputParts = (*std::max_element(partsPerCls.begin(), partsPerCls.end()));

    if(templatesPerCls.size() > 0)
        nbOutputTemplates = (*std::max_element(templatesPerCls.begin(), templatesPerCls.end()));

    //const unsigned int nbOutputs = 5 + nbOutputParts*2 + nbOutputTemplates*3;
    const unsigned int nbOutputs = 6 + nbOutputParts*2 + nbOutputTemplates*3;

    std::vector<AnchorCell_Frame_Kernels::Anchor> anchors;

    //if(templatesPerCls.size() > 0 || partsPerCls.size() > 0)
    //{

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
                                                            AnchorCell_Frame_Kernels::Anchor::Anchoring::TopLeft));

            ++nextAnchor;
            nextProperty.str(std::string());
            nextProperty << "Anchor[" << nextAnchor << "]";
        }
        
        nextProperty.str(std::string());
        nextProperty << "AnchorBBOX[" << nextAnchor << "]";

        while (iniConfig.isProperty(nextProperty.str())) {
            std::stringstream anchorValues(
                iniConfig.getProperty<std::string>(nextProperty.str()));

            float x0;
            float y0;
            float w;
            float h;

            if (!(anchorValues >> x0) || !(anchorValues >> y0)
                    || !(anchorValues >> w) || !(anchorValues >> h)) {
                throw std::runtime_error(
                    "Unreadable anchor in section [" + section
                    + "] in network configuration file: "
                    + iniConfig.getFileName());
            }

            anchors.push_back(AnchorCell_Frame_Kernels::Anchor( x0,
                                                                y0,
                                                                w,
                                                                h));

            ++nextAnchor;
            nextProperty.str(std::string());
            nextProperty << "AnchorBBOX[" << nextAnchor << "]";
        }

        nextProperty.str(std::string());
        nextProperty << "AnchorXY[" << nextAnchor << "]";

        while (iniConfig.isProperty(nextProperty.str())) {
            std::stringstream anchorValues(
                iniConfig.getProperty<std::string>(nextProperty.str()));

            float x0;
            float y0;
            float w;
            float h;

            if (!(anchorValues >> x0) || !(anchorValues >> y0)) {
                throw std::runtime_error(
                    "Unreadable anchor in section [" + section
                    + "] in network configuration file: "
                    + iniConfig.getFileName());
            }
            w = std::abs(x0*2);
            h = std::abs(y0*2);

            anchors.push_back(AnchorCell_Frame_Kernels::Anchor( x0,
                                                                y0,
                                                                w,
                                                                h));

            ++nextAnchor;
            nextProperty.str(std::string());
            nextProperty << "AnchorXY[" << nextAnchor << "]";
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
                    AnchorCell_Frame_Kernels::Anchor::Anchoring::TopLeft));
            }
        }

    //}

    std::cout << "Layer: " << section << " [ObjectDet(" << model << ")]" 
              << std::endl;   

    // Cell construction
    std::shared_ptr<ObjectDetCell> cell = Registrar
        <ObjectDetCell>::create(model)(deepNet, section,
                                sp,
                                nbOutputs,
                                nbAnchors,
                                inputFormat,
                                pixelFormat,
                                nbProposals,
                                nbCls,
                                nmsThreshold,
                                scoreThresholds,
                                partsPerCls,
                                templatesPerCls,
                                anchors);

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

    std::cout << "  # Outputs: " << cell->getNbOutputs() << std::endl;

    return cell;
}