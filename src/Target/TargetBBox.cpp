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

#include "Target/TargetBBox.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetBBox::mRegistrar("TargetBBox", N2D2::TargetBBox::create);

const char* N2D2::TargetBBox::Type = "TargetBBox";

N2D2::TargetBBox::TargetBBox(const std::string& name,
                            const std::shared_ptr<Cell>& cell,
                            const std::shared_ptr<StimuliProvider>& sp,
                            double targetValue,
                            double defaultValue,
                            unsigned int targetTopN,
                            const std::string& labelsMapping)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping)
{
    // ctor
}

void N2D2::TargetBBox::logConfusionMatrix(const std::string& /*fileName*/,
                                          Database::StimuliSet /*set*/) const
{
/**TODO**/
}

void N2D2::TargetBBox::clearConfusionMatrix(Database::StimuliSet /*set*/)
{
/**TODO**/
}

void N2D2::TargetBBox::initialize()
{
    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const Tensor4d<Float_T>& values = targetCell->getOutputs();
 
    if (values.dimZ() != 4 && values.dimZ() != 5) {
        //throw std::runtime_error("TargetBBox::initialize(): cell must have 4 or 5"
        //                        " output channels for BBox TargetBBox " + mName);

        std::cout << "Target BBOX" << std::endl;
    }

    mTargets.resize(mCell->getOutputsWidth(),
                    mCell->getOutputsHeight(),
                    values.dimZ(),
                    values.dimB());

}

void N2D2::TargetBBox::process(Database::StimuliSet /*set*/)
{

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);

    const std::vector<int>& batch = mStimuliProvider->getBatch();
    const int nbBBoxMax = (int)mTargets.dimB()/batch.size();
    const Tensor4d<Float_T>& values = targetCell->getOutputs();

    mBatchDetectedBBox.assign(mTargets.dimB(), std::vector<DetectedBBox>());

    //#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)batch.size(); ++batchPos) {
        //const int id = mStimuliProvider->getBatch()[batchPos];

        std::vector<DetectedBBox> bbox;

        for(int bboxIdx = 0; bboxIdx < nbBBoxMax; ++ bboxIdx)
        {
            const Tensor3d<Float_T>& value = values[bboxIdx + batchPos*nbBBoxMax];
            Float_T x = value(0);
            Float_T y = value(1);
            Float_T w = value(2);
            Float_T h = value(3);
            Float_T cls = 0.0;   
            if(values.dimZ() > 4)
                cls = value(4);
            
            std::cout << "BBox{" << x << ", " << y << ", " << w << ", " << h << "}[" << cls << "]" << std::endl;
            if(w > 0.0 && h > 0.0)
            {
                bbox.push_back(DetectedBBox(x, y, w ,h, cls));
                if(values.dimZ() > 5)
                {
                    std::cout << "{";
                    for(unsigned int p = 5; p < values.dimZ(); ++p)
                        std::cout << value(p) << " ";
                    std::cout << "}" << std::endl;
                }
            }  
        }

        mBatchDetectedBBox[batchPos].resize(bbox.size());

        for(unsigned int i = 0; i < bbox.size(); ++i)
            mBatchDetectedBBox[batchPos][i] = bbox[i];

        //mBatchDetectedBBox[batchPos].swap(bbox);

        std::cout << "TargetBBox: Number of BBox detected on batch " 
            << batchPos << ": " << bbox.size() << std::endl; 

    }

}



cv::Mat N2D2::TargetBBox::drawEstimatedBBox(unsigned int batchPos) const
{
    const std::vector<DetectedBBox>& detectedBB = mBatchDetectedBBox[batchPos];

    const std::vector<std::string> labelsName = getTargetLabelsName();
    //const int defaultLabel = getLabelTarget(mStimuliProvider->getDatabase()
    //                                            .getDefaultLabelID());

    // Input image
    cv::Mat img = (cv::Mat)mStimuliProvider->getData(0, batchPos);
    cv::Mat img8U;
    // img.convertTo(img8U, CV_8U, 255.0);

    // Normalize image
    cv::Mat imgNorm;
    cv::normalize(img.reshape(1), imgNorm, 0, 255, cv::NORM_MINMAX);
    img = imgNorm.reshape(img.channels());
    img.convertTo(img8U, CV_8U);

    cv::Mat imgBB;
    cv::cvtColor(img8U, imgBB, CV_GRAY2BGR);

    // Draw targets ROIs
    const std::vector<std::shared_ptr<ROI> >& labelROIs
        = mStimuliProvider->getLabelsROIs(batchPos);

    for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
         = labelROIs.begin(),
         itLabelEnd = labelROIs.end();
         itLabel != itLabelEnd;
         ++itLabel)
    {
        const int target = getLabelTarget((*itLabel)->getLabel());

        if (target >= 0) {
            (*itLabel)->draw(imgBB, cv::Scalar(255, 0, 0));

            // Draw legend
            const cv::Rect rect = (*itLabel)->getBoundingRect();
            std::stringstream legend;
            legend << target << " " << labelsName[target];

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(
                legend.str(), cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseline);
            cv::putText(imgBB,
                        legend.str(),
                        cv::Point(std::max(0, rect.x) + 2,
                                  std::max(0, rect.y) + textSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.25,
                        cv::Scalar(255, 0, 0),
                        1,
                        CV_AA);
        }
    }

    // Draw detected BB


    for (std::vector<DetectedBBox>::const_iterator it = detectedBB.begin(),
                                                 itEnd = detectedBB.end();
                                                 it != itEnd; ++it)
    {
      
        cv::Scalar color = cv::Scalar(255 * ((*it).cls / labelsName.size()),
                                      255 - 255*((*it).cls / labelsName.size()),
                                      255);

        RectangularROI<int> bbox( (int) (*it).cls, cv::Point((*it).x, (*it).y), (*it).w, (*it).h);
        bbox.draw(imgBB, color);

        // Draw legend

        const cv::Rect rect = bbox.getBoundingRect();
        std::stringstream legend;
        legend << bbox.getLabel();

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            legend.str(), cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseline);
        cv::putText(imgBB,
                    legend.str(),
                    cv::Point(rect.x + 2, rect.y + textSize.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.25,
                    color,
                    1,
                    CV_AA);
            

        
    }

    return imgBB;
}


void N2D2::TargetBBox::logEstimatedLabels(const std::string& dirName) const
{
    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    const std::vector<int>& batch = mStimuliProvider->getBatch();

#pragma omp parallel for if (batch.size() > 4)
    for (int batchPos = 0; batchPos < (int)batch.size(); ++batchPos) {
        const int id = batch[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const std::string imgFile
            = mStimuliProvider->getDatabase().getStimulusName(id);
        const std::string baseName = Utils::baseName(imgFile);
        const std::string fileName = dirPath + "/" + baseName;

        // Draw image

        if (!cv::imwrite(fileName, drawEstimatedBBox(batchPos)))
            throw std::runtime_error("Unable to write image: " + fileName);
        
    }
}

void N2D2::TargetBBox::log(const std::string& /*fileName*/,
                           Database::StimuliSet /*set*/)
{
/**TODO**/
}

void N2D2::TargetBBox::clear(Database::StimuliSet set)
{
    clearConfusionMatrix(set);
}

N2D2::TargetBBox::~TargetBBox()
{

}