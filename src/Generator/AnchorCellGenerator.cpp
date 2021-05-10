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

#include "StimuliProvider.hpp"
#include "DeepNet.hpp"
#include "Generator/AnchorCellGenerator.hpp"
#include "StimuliProvider.hpp"

#ifdef JSONCPP
#include <jsoncpp/json/json.h>
#endif

N2D2::Registrar<N2D2::CellGenerator>
N2D2::AnchorCellGenerator::mRegistrar(AnchorCell::Type,
                                       N2D2::AnchorCellGenerator::generate);

std::shared_ptr<N2D2::AnchorCell>
N2D2::AnchorCellGenerator::generate(Network& /*network*/, const DeepNet& deepNet,
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
              
    const AnchorCell_Frame_Kernels::DetectorType detectorType 
        = iniConfig.getProperty<AnchorCell_Frame_Kernels::DetectorType>
            ("DetectorType");         
    const AnchorCell_Frame_Kernels::Format inputFormat 
        = iniConfig.getProperty<AnchorCell_Frame_Kernels::Format>
            ("InputFormat", AnchorCell_Frame_Kernels::Format::CA);         

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
    // Second method: specify handmade anchors with X, y, Width & Height
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

        anchors.push_back(AnchorCell_Frame_Kernels::Anchor(x0,
                                                           y0,
                                                           w,
                                                           h));

        ++nextAnchor;
        nextProperty.str(std::string());
        nextProperty << "AnchorBBOX[" << nextAnchor << "]";
    }

    // Third method: specify handmade anchors with X0 & Y0
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

        anchors.push_back(AnchorCell_Frame_Kernels::Anchor(x0,
                                                           y0,
                                                           w,
                                                           h));

        ++nextAnchor;
        nextProperty.str(std::string());
        nextProperty << "AnchorXY[" << nextAnchor << "]";
    }
#ifdef JSONCPP
    const std::string anchorsJSONpath = Utils::expandEnvVars(
        iniConfig.getProperty<std::string>("AnchorJSON", ""));
    std::ifstream jsonData(anchorsJSONpath);

    if (!jsonData.good()) {
        throw std::runtime_error("AnchorCellGenerator::generate: Could not open JSON Anchor file "
                                    "(missing?): " + anchorsJSONpath);
    }
    Json::Reader reader;
    Json::Value labels;
    if (!reader.parse(jsonData, labels)) {
        std::cerr << "AnchorCellGenerator::generate: Error parsing JSON file " 
            << anchorsJSONpath<< " at line "
            << reader.getFormattedErrorMessages() << std::endl;

        throw std::runtime_error("JSON file parsing failed");
    }
    const Json::Value& jsonAnnotations = labels["ANCHORS"];
    if(jsonAnnotations.size() < 1 ){
        std::cerr << "Error parsing JSON file " << anchorsJSONpath << " at field "
            << "annotations: Cannot have more than one"
            << " annotations mask per file, here it is " 
            << jsonAnnotations.size() << std::endl;

        throw std::runtime_error(" file parsing failed");
    }
    for(unsigned int cls = 0; cls < jsonAnnotations.size(); ++cls ){
        const Json::Value& clsAnchors = jsonAnnotations[cls];
        for(unsigned int idx = 0; idx < clsAnchors.size(); ++idx ) {
            const Json::Value& idxAnchors = clsAnchors[idx];
            if(idxAnchors.size() != 4 ) {
                std::cerr << "Error parsing JSON file " << anchorsJSONpath << " at field "
                    << "annotations: Cannot have an anchor field values size different than 4 " 
                    << idxAnchors.size() << std::endl;
                throw std::runtime_error(" file parsing failed");
            }
            
            const double x0 = idxAnchors[1].asDouble();
            const double y0 = idxAnchors[0].asDouble();
            const double w = std::abs(x0) + std::abs(idxAnchors[3].asDouble());
            const double h = std::abs(y0) + std::abs(idxAnchors[2].asDouble());

            anchors.push_back(AnchorCell_Frame_Kernels::Anchor(x0,
                                                            y0,
                                                            w,
                                                            h));
            
        }
    }
#endif
    // Fourth method: specify a base root area and a list of ratios and scales
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
                                     <unsigned int>("ScoresCls", 0);

    // Cell construction
    std::shared_ptr<AnchorCell> cell = Registrar
        <AnchorCell>::create(model)(deepNet, section, sp, detectorType, inputFormat, anchors, scoresCls);

    if (!cell) {
        throw std::runtime_error(
            "Cell model \"" + model + "\" is not valid in section [" + section
            + "] in network configuration file: " + iniConfig.getFileName());
    }
    const std::string mapLabelFile = iniConfig.getProperty<std::string>("LabelMapping", "");

    if(!mapLabelFile.empty())
    {
        cell->labelsMapping(mapLabelFile);
        std::vector<int> mapLabel = cell->getLabelMapping();

        const bool generateAnchors = 
                iniConfig.getProperty<bool>("GenerateAnchors", false);
        if(generateAnchors)
        {
            std::vector<AnchorBBOX_T> computeAnchors 
                = generateAnchors_kmeans(sp, Database::Learn, mapLabel, anchors.size());

            if (computeAnchors.size() < anchors.size())
            {
                std::cout << "N2D2::AnchorCellGenerator Cannot compute KMeans clustering to generate anchors" << std::endl;
            }
            else
            {
                anchors.clear();

                for(unsigned int i = 0; i < computeAnchors.size(); ++i)
                {
                    anchors.push_back(AnchorCell_Frame_Kernels::Anchor( -computeAnchors[i].w * 0.5,
                                                                        -computeAnchors[i].h * 0.5,
                                                                        computeAnchors[i].w,
                                                                        computeAnchors[i].h));
                }

                cell->setAnchors(anchors);
            }
        }
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
    std::cout << "  # Inputs dims: " << cell->getInputsDims() << std::endl;
    std::cout << "  # Outputs dims: " << cell->getOutputsDims() << std::endl;

    return cell;
}

std::vector<N2D2::AnchorCellGenerator::AnchorBBOX_T> 
N2D2::AnchorCellGenerator::generateAnchors_kmeans(StimuliProvider& sp,
                                                    Database::StimuliSet set,
                                                    std::vector<int>& labels,
                                                    unsigned int nbAnchors) 
{
    const unsigned int nbTest = sp.getDatabase().getNbStimuli(set);
    const unsigned int batchSize = sp.getBatchSize();
    const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);
    std::vector<AnchorBBOX_T> total_bbox;
    std::cout << "AnchorCell: Compute kMeans clustering for anchors generation" << std::endl;
    std::cout << "Loading labels..." << std::flush;
    unsigned int progress = 0, progressPrev = 0;

    for (unsigned int b = 0; b < nbBatch; ++b) {
        const unsigned int i = b * batchSize;
        const unsigned int idx = i;
        sp.readBatch(set, idx);
        progress = (unsigned int)(10.0 * b / (double)nbBatch);
        if (progress > progressPrev) {
            std::cout << std::string(progress - progressPrev,
                                        '.') << std::flush;
            progressPrev = progress;
        }


        for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        {
            const std::vector<std::shared_ptr<ROI> >& labelROIs = sp.getLabelsROIs(batchPos);

            for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel = labelROIs.begin(),
                itLabelEnd = labelROIs.end(); itLabel != itLabelEnd; ++itLabel)
            {
                const int target = labels[(*itLabel)->getLabel()];
                if(target != -1)
                {
                    cv::Rect labelRect = (*itLabel)->getBoundingRect();
                    AnchorBBOX_T bbox = AnchorBBOX_T(   labelRect.tl().x, 
                                                        labelRect.tl().y, 
                                                        labelRect.width,
                                                        labelRect.height, (float) target);
                    if (labelRect.width > 0.0 && labelRect.height > 0.0) 
                        total_bbox.push_back(bbox);
                }
                
            }
        }

    }
    progress = 0; 
    progressPrev = 0;

    std::cout << "Find: " << total_bbox.size() << " Bounding box" << std::endl;

    const unsigned int cluster = nbAnchors;
    const long unsigned int nbIter = 1000;
    std::vector<AnchorBBOX_T> clusters(cluster, AnchorBBOX_T(0.0, 0.0, 0.0, 0.0, 0.0));

    for(unsigned int kL = 0; kL < cluster; ++ kL)
    {   
        int randIdx = (int) Random::randUniform(0, total_bbox.size());
        clusters[kL] = total_bbox[randIdx];
    }


    //std::vector<Float_T> distances(total_bbox.size(), 0);
    Tensor<Float_T> distances({total_bbox.size(), cluster}, 1.0);
    Tensor<unsigned int> distances_min({total_bbox.size()}, 0);
    Tensor<unsigned int> distances_min_prev({total_bbox.size()}, 0);
    unsigned int iT = 0;

    while(iT < nbIter)
    {
        ++iT;
        Float_T AvgIoU = 0.0;

        for(unsigned int l = 0; l < total_bbox.size(); ++l)
        {   
            float w;
            float h;
            float min_distance = 1.0;
            for(unsigned int kL = 0; kL < cluster; ++kL)
            {
                w = std::min(clusters[kL].w, total_bbox[l].w);
                h = std::min(clusters[kL].h, total_bbox[l].h);
                Float_T iou = (w*h) / ((total_bbox[l].w * total_bbox[l].h)
                                        + (clusters[kL].w*clusters[kL].h) - (w*h));

                distances(l, kL) = 1.0 - iou;
                
                if(kL == 0)
                {
                    min_distance = distances(l, kL);
                }
                else
                {
                    if(distances(l, kL) < min_distance)
                    {
                        distances_min(l) = kL;
                        min_distance = distances(l, kL);
                    }
                }
                AvgIoU += iou;

            }
        }

        AvgIoU /= total_bbox.size();

        std::vector<bool> isSame;
        for(unsigned int l = 0; l < total_bbox.size(); ++l)
        {
            if(distances_min_prev(l) == distances_min(l))
                isSame.push_back(1);
        }
        if(isSame.size() == total_bbox.size())
        {
            std::cout << "clustering is finished" << std::endl;
            break;
        }
        else
            std::cout << std::setprecision(2) << std::fixed 
            << "Equality (" << ((float) isSame.size() / (float) total_bbox.size()) * 100.0 << " %) " 
            << AvgIoU << " average IoU \r" << std::flush ; 
        std::vector<Float_T> width_sum(cluster, 0.0);
        std::vector<Float_T> height_sum(cluster, 0.0);

        for(unsigned int l = 0; l < total_bbox.size(); ++l)
        {
            width_sum[distances_min(l)] += total_bbox[l].w;
            height_sum[distances_min(l)] += total_bbox[l].h;
        }
        for(unsigned int kL = 0; kL < cluster; ++kL)
        {
            std::vector<Float_T> width(cluster, 0.0);
            std::vector<Float_T> height(cluster, 0.0);

            unsigned int totalSum = 0;
            for(unsigned int l = 0; l < total_bbox.size(); ++l)
            {
                if(distances_min(l) == kL)
                    totalSum += 1;  
            }
            if(totalSum == 0)
            {
                std::cout << "Stop at iteration " << iT << std::endl;
                //throw std::runtime_error("totalSum is 0, cannot compute kmeans");
                std::cout << "totalSum is 0, cannot compute kmeans" << std::endl;
                return std::vector<AnchorBBOX_T>();

            }
            width[kL] = width_sum[kL] / (float) totalSum;
            height[kL] = height_sum[kL] / (float) totalSum;

            clusters[kL] = AnchorBBOX_T(0.0, 0.0, width[kL], height[kL], (float) kL);
        }

        for(unsigned int l = 0; l < total_bbox.size(); ++l)
            distances_min_prev(l) = distances_min(l); 
    }

    std::cout << "Find clusters: " << "(" << iT << " iterations)" << std::endl;
    for(unsigned int i = 0; i < cluster; ++i)
        std::cout << "{" << -clusters[i].w * 0.5 << " " 
                    << -clusters[i].h * 0.5 << " "
                    << clusters[i].w << " "
                    << clusters[i].h << "}" << std::endl; 

    double AvgIoU = 0.0;
    for(unsigned int l = 0; l < total_bbox.size(); ++l)
    {   
        float w;
        float h;
        float maxIoU = 0.0;
        for(unsigned int kL = 0; kL < cluster; ++kL)
        {
            w = std::min(clusters[kL].w, total_bbox[l].w);
            h = std::min(clusters[kL].h, total_bbox[l].h);
            Float_T iou = (w*h) / ((total_bbox[l].w * total_bbox[l].h)
                                    + (clusters[kL].w*clusters[kL].h) - (w*h));
            maxIoU = iou > maxIoU ? iou : maxIoU;
        }
        AvgIoU += maxIoU;
    }
    AvgIoU = AvgIoU / total_bbox.size();
    std::cout << "Average IoU: " << AvgIoU << std::endl; 

    //const std::vector<int>& batch = sp.getBatch();

    //for (int batchPos = 0; batchPos < (int)batch.size(); ++batchPos) {
    //    logEstimatedLabels("AnchorsSamples/");
    //}

    const std::string dataFileName = "n2d2_anchors_generated.coord";
    std::ofstream dataFile(dataFileName);

    if (!dataFile.good())
        throw std::runtime_error("Could not create anchors file: "
                                    + dataFileName);
    for(unsigned int i = 0; i < clusters.size(); ++ i)
    {
        dataFile << -clusters[i].w * 0.5 << " " 
                << -clusters[i].h * 0.5 << " " 
                << clusters[i].w << " " 
                << clusters[i].h << "\n";
    }
    dataFile.close();

    return clusters;
}
