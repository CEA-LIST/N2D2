/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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

#include "Cell/AnchorCell.hpp"
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"

const char* N2D2::AnchorCell::Type = "Anchor";

N2D2::AnchorCell::AnchorCell(const DeepNet& deepNet, 
    const std::string& name,
    StimuliProvider& sp,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
    unsigned int scoresCls)
    : Cell(deepNet, name, 6*anchors.size()),
      mPositiveIoU(this, "PositiveIoU", 0.7),
      mNegativeIoU(this, "NegativeIoU", 0.3),
      mLossLambda(this, "LossLambda", 10.0),
      mLossPositiveSample(this, "LossPositiveSample", 128U),
      mLossNegativeSample(this, "LossNegativeSample", 128U),
      mFeatureMapWidth(this, "FeatureMapWidth", 0U),
      mFeatureMapHeight(this, "FeatureMapHeight", 0U),
      mFlip(this, "Flip", false),
      mNegativeRatioSSD(this, "NegativeRatio", 3U),
      mMaxLabelGT(this, "MaxLabelPerFrame", 1000U),
      mNbClass(this, "NbClass", -1),
      mDetectorType(this, "DetectorType", AnchorCell_Frame_Kernels::DetectorType::LapNet),
      mInputFormat(this, "InputFormat", AnchorCell_Frame_Kernels::Format::CA),
      mStimuliProvider(sp),
      mScoresCls(scoresCls)
{
    // ctor
}

void N2D2::AnchorCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::AnchorCell::setAnchors(const std::vector<AnchorCell_Frame_Kernels::Anchor>& /*anchors*/)
{

}


void N2D2::AnchorCell::setOutputsDims()
{
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}

void N2D2::AnchorCell::labelsMapping(const std::string& fileName)
{
    //for(unsigned int i = 0; i < mStimuliProvider.getDatabase().getNbLabels(); ++i)
    //    mLabelsMapping.insert(std::make_pair(i, -1));
    mLabelsMapping.resize(mStimuliProvider.getDatabase().getNbLabels(), -1);
    std::cout << "mLabelsMapping.size(): " << mLabelsMapping.size() << std::endl;
    //mLabelsMapping.clear();
    std::string fullFileName = Utils::expandEnvVars(fileName);

    if (fullFileName.empty())
        return;

    std::ifstream clsFile(fullFileName.c_str());

    if (!clsFile.good())
        throw std::runtime_error("Could not open class mapping file: "
                                 + fullFileName);

    std::string line;

    while (std::getline(clsFile, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '#'), line.end());
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        std::string className;
        int output;

        std::stringstream value(line);
        std::stringstream classNameStr;

        int wordInString = std::count_if(line.begin(), line.end(), [](char ch) { return isspace(ch); });

        for(int idx = 0; idx < wordInString; ++idx)
        {
            std::string str;
            if (!(value >> Utils::quoted(str)))
                throw std::runtime_error("Unreadable class name: " + line + " in file "
                                         + fullFileName);
             if(idx > 0)
                classNameStr << " ";

             classNameStr << str;
        }

        className = classNameStr.str();

        //if (!(value >> Utils::quoted(className)) || !(value >> output)
        //    || (output < 0 && output != -1) || !value.eof())
        //    throw std::runtime_error("Unreadable value: " + line + " in file "
         //                            + fileName);

        if (!(value >> output) || (output < 0 && output != -1) || !value.eof())
            throw std::runtime_error("Unreadable value: " + line + " in file "
                                     + fullFileName);


        if (className == "default") {
            std::cout << "dont care about default value" << std::endl;
        } else {
            int label = -1;
            bool corruptedLabel = false;
            if (className != "*") {

                if (!mStimuliProvider.getDatabase().isLabel(className)) {
                    std::cout
                        << Utils::cwarning
                        << "No label exists in the database with the name: "
                        << className << " in file " << fullFileName << Utils::cdef
                        << std::endl;

                    corruptedLabel = true;
                } else
                    label = mStimuliProvider.getDatabase().getLabelID(className);
            }

            if (!corruptedLabel && label > -1) {
                mLabelsMapping[label] = output;
            }
        }
    }

    for(unsigned int i = 0; i < mStimuliProvider.getDatabase().getNbLabels(); ++i)
        std::cout << "LabelID(" << i << "): " << mLabelsMapping[i] << std::endl;

}
