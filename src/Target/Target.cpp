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

#include "Target/Target.hpp"

N2D2::Registrar<N2D2::Target> N2D2::Target::mRegistrar("Target",
                                                       N2D2::Target::create);

const char* N2D2::Target::Type = "Target";

N2D2::Target::Target(const std::string& name,
                     const std::shared_ptr<Cell>& cell,
                     const std::shared_ptr<StimuliProvider>& sp,
                     double targetValue,
                     double defaultValue,
                     unsigned int targetTopN,
                     const std::string& labelsMapping_)
    : mDataAsTarget(this, "DataAsTarget", false),
      mNoDisplayLabel(this, "NoDisplayLabel", -1),
      mLabelsHueOffset(this, "LabelsHueOffset", 0),
      mMaskedLabel(this, "MaskedLabel", -1),
      mName(name),
      mCell(cell),
      mStimuliProvider(sp),
      mTargetValue(targetValue),
      mDefaultValue(defaultValue),
      mTargetTopN(targetTopN),
      mDefaultTarget(-2),
      mPopulateTargets(true)
{
    // ctor
    Utils::createDirectories(name);

    if (!labelsMapping_.empty())
        labelsMapping(labelsMapping_);
}

unsigned int N2D2::Target::getNbTargets() const
{
    return (mCell->getNbOutputs() > 1) ? mCell->getNbOutputs() : 2;
}

void N2D2::Target::labelsMapping(const std::string& fileName)
{
    mLabelsMapping.clear();

    if (fileName.empty())
        return;

    std::ifstream clsFile(fileName.c_str());

    if (!clsFile.good())
        throw std::runtime_error("Could not open class mapping file: "
                                 + fileName);

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

        if (!(value >> Utils::quoted(className)) || !(value >> output)
            || (output < 0 && output != -1) || !value.eof())
            throw std::runtime_error("Unreadable value: " + line + " in file "
                                     + fileName);

        if (className == "default") {
            if (mDefaultTarget >= -1)
                throw std::runtime_error(
                    "Default mapping already exists in file " + fileName);

            mDefaultTarget = output;
        } else {
            int label = -1;
            bool corruptedLabel = false;
            if (className != "*") {

                if (!mStimuliProvider->getDatabase().isLabel(className)) {
                    std::cout
                        << Utils::cwarning
                        << "No label exists in the database with the name: "
                        << className << " in file " << fileName << Utils::cdef
                        << std::endl;

                    corruptedLabel = true;
                } else
                    label
                        = mStimuliProvider->getDatabase().getLabelID(className);
            }

            if (!corruptedLabel) {
                bool newInsert;

                std::tie(std::ignore, newInsert)
                    = mLabelsMapping.insert(std::make_pair(label, output));
                if (!newInsert)
                    throw std::runtime_error(
                        "Mapping already exists for label: " + line
                        + " in file " + fileName);
            }
        }
    }
}

void N2D2::Target::setLabelTarget(int label, int output)
{
    mLabelsMapping[label] = output;
}

void N2D2::Target::setDefaultTarget(int output)
{
    mDefaultTarget = output;
}

int N2D2::Target::getLabelTarget(int label) const
{
    if (mLabelsMapping.empty())
        return label;
    else {
        const std::map<int, int>::const_iterator it
            = mLabelsMapping.find(label);

        if (it != mLabelsMapping.end())
            return (*it).second;
        else if (mDefaultTarget >= -1)
            return mDefaultTarget;
        else {
            std::stringstream labelStr;
            labelStr << label;

            const std::string labelName
                = mStimuliProvider->getDatabase().getLabelName(label);

            throw std::runtime_error(
                "Incomplete class mapping: no output specified for label #"
                + labelStr.str() + " (" + labelName + ")");
        }
    }
}

int N2D2::Target::getDefaultTarget() const
{
    if (mDefaultTarget >= -1)
        return mDefaultTarget;
    else
        throw std::runtime_error("No default target mapping");
}

std::vector<int> N2D2::Target::getTargetLabels(int output) const
{
    if (mLabelsMapping.empty())
        return std::vector<int>(1, output);
    else {
        std::vector<int> labels;

        for (std::map<int, int>::const_iterator it = mLabelsMapping.begin(),
                                                itEnd = mLabelsMapping.end();
             it != itEnd;
             ++it) {
            if ((*it).second == output)
                labels.push_back((*it).first);
        }

        return labels;
    }
}

std::vector<std::string> N2D2::Target::getTargetLabelsName() const
{
    const unsigned int nbTargets = getNbTargets();
    std::vector<std::string> labelsName;
    labelsName.reserve(nbTargets);

    for (int target = 0; target < (int)nbTargets; ++target) {
        std::stringstream labelName;

        if (target == mDefaultTarget)
            labelName << "default";
        else {
            const std::vector<int> cls = getTargetLabels(target);
            const Database& db = mStimuliProvider->getDatabase();

            if (!cls.empty()) {
                labelName << ((cls[0] >= 0 && cls[0] < (int)db.getNbLabels())
                                ? db.getLabelName(cls[0]) :
                              (cls[0] >= 0)
                                ? "" :
                                  "*");

                if (cls.size() > 1)
                    labelName << "...";
            }
        }

        labelsName.push_back(labelName.str());
    }

    return labelsName;
}

void N2D2::Target::logLabelsMapping(const std::string& fileName) const
{
    if (mDataAsTarget)
        return;

    const std::string dataFileName = mName + "/" + fileName + ".dat";
    std::ofstream labelsData(dataFileName);

    if (!labelsData.good())
        throw std::runtime_error("Could not save log class mapping data file: "
                                 + dataFileName);

    labelsData << "label name output\n";

    for (unsigned int label = 0,
                      size = mStimuliProvider->getDatabase().getNbLabels();
         label < size;
         ++label)
        labelsData << label << " "
                   << mStimuliProvider->getDatabase().getLabelName(label) << " "
                   << getLabelTarget(label) << "\n";
}

void N2D2::Target::process(Database::StimuliSet set)
{
    const unsigned int nbTargets = getNbTargets();

    if (mDataAsTarget) {
        if (set == Database::Learn) {
            std::shared_ptr<Cell_Frame_Top> targetCell
                = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);

            // Set targets
            targetCell->setOutputTargets(mStimuliProvider->getData());
        }
    } else {
        const Tensor4d<int>& labels = mStimuliProvider->getLabelsData();

        if (mTargets.empty()) {
            mTargets.resize(mCell->getOutputsWidth(),
                            mCell->getOutputsHeight(),
                            1,
                            labels.dimB());
            mEstimatedLabels.resize(mCell->getOutputsWidth(),
                                    mCell->getOutputsHeight(),
                                    mTargetTopN,
                                    labels.dimB());
            mEstimatedLabelsValue.resize(mCell->getOutputsWidth(),
                                         mCell->getOutputsHeight(),
                                         mTargetTopN,
                                         labels.dimB());
        }

        if (mPopulateTargets) {
            // Generate targets
            if (mCell->getOutputsWidth() > 1 || mCell->getOutputsHeight() > 1) {
                const double xRatio = labels.dimX()
                                      / (double)mCell->getOutputsWidth();
                const double yRatio = labels.dimY()
                                      / (double)mCell->getOutputsHeight();

#pragma omp parallel for if (labels.dimB() > 16)
                for (int batchPos = 0; batchPos < (int)labels.dimB();
                    ++batchPos)
                {
                    const Tensor2d<int> label = labels[batchPos][0];
                    Tensor2d<int> target = mTargets[batchPos][0];

                    for (unsigned int x = 0; x < mTargets.dimX(); ++x) {
                        for (unsigned int y = 0; y < mTargets.dimY(); ++y) {
                            target(x, y) = getLabelTarget(
                                label((int)std::floor((x + 0.5) * xRatio),
                                      (int)std::floor((y + 0.5) * yRatio)));
                        }
                    }
                }
            } else {
                for (int batchPos = 0; batchPos < (int)labels.dimB();
                    ++batchPos)
                {
                    const Tensor3d<int> label = labels[batchPos];
                    Tensor3d<int> target = mTargets[batchPos];

                    // target only has 1 channel, whereas label has as many
                    // channels as environment channels
                    target(0) = getLabelTarget(label(0));
                }
            }
        }

        std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
            <Cell_Frame_Top>(mCell);

        if (set == Database::Learn) {
            // Set targets
            if (mTargets.dimX() == 1 && mTargets.dimY() == 1) {
                for (unsigned int batchPos = 0; batchPos < mTargets.dimB();
                     ++batchPos) {
                    if (mTargets(0, batchPos) < 0) {
                        std::cout << Utils::cwarning
                                  << "Target::setTargetsValue(): ignore label "
                                     "with 1D output for stimuli ID "
                                  << mStimuliProvider->getBatch()[batchPos]
                                  << Utils::cdef << std::endl;
                    }
                }

                targetCell->setOutputTarget(
                    mTargets, mTargetValue, mDefaultValue);
            } else
                targetCell->setOutputTargets(
                    mTargets, mTargetValue, mDefaultValue);
        }

        // Retrieve estimated labels
        std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
            = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);
        const Tensor4d<Float_T>& values
            = (targetCell) ? targetCell->getOutputs()
                           : targetCellCSpike->getOutputsActivity();

#pragma omp parallel for if (mTargets.dimB() > 4)
        for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of
                // the set)
                continue;
            }

            const Tensor3d<Float_T> value = values[batchPos];
            const Tensor3d<int> target = mTargets[batchPos];
            Tensor3d<int> estimatedLabels = mEstimatedLabels[batchPos];
            Tensor3d<Float_T> estimatedLabelsValue
                = mEstimatedLabelsValue[batchPos];

            if (mTargetTopN > value.dimZ())
                throw std::runtime_error("Target::process(): target 'TopN' "
                                         "parameter must be <= to the network "
                                         "output size");

            if (target.size() == 1) {
                if (target(0) >= 0) {
                    if (target(0) >= (int)nbTargets) {
#pragma omp critical
                        {
                            std::cout << Utils::cwarning << "Stimulus #"
                                << id << " has target " << target(0) << " but "
                                "number of output target is " << nbTargets
                                << Utils::cdef << std::endl;

                            throw std::runtime_error("Target::process(): target"
                                                     " out of range.");
                        }
                    }

                    if (value.size() > 1) {
                        std::vector
                            <std::pair<Float_T, size_t> > sortedLabelsValues;
                        sortedLabelsValues.reserve(value.size());

                        for (unsigned int index = 0; index < value.size();
                             ++index)
                            sortedLabelsValues.push_back(
                                std::make_pair(value(index), index));

                        // Top-n accuracy sorting
                        std::partial_sort(
                            sortedLabelsValues.begin(),
                            sortedLabelsValues.begin() + mTargetTopN,
                            sortedLabelsValues.end(),
                            std::greater<std::pair<Float_T, size_t> >());

                        for (unsigned int i = 0; i < mTargetTopN; ++i) {
                            estimatedLabels(i) = sortedLabelsValues[i].second;
                            estimatedLabelsValue(i)
                                = sortedLabelsValues[i].first;
                        }
                    } else {
                        estimatedLabels(0) = (value(0) > 0.5);
                        estimatedLabelsValue(0) = value(0);
                    }

                    static bool display = true;

                    if (set == Database::Test && batchPos == 0 && display) {
                        std::cout << "[";

                        for (int i = 0, size = value.size(); i < size; ++i) {
                            if (i == estimatedLabels(0))
                                std::cout << std::setprecision(2) << std::fixed
                                          << "(" << value(i) << ") ";
                            else
                                std::cout << std::setprecision(2) << std::fixed
                                          << value(i) << " ";
                        }

                        std::cout << "]" << std::endl;
                        display = false;
                    }
                }
            } else {
                const unsigned int nbOutputs = value.dimZ();

                for (unsigned int oy = 0; oy < value.dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < value.dimX(); ++ox) {
                        if (target(ox, oy, 0) >= (int)nbTargets) {
#pragma omp critical
                            {
                                std::cout << Utils::cwarning << "Stimulus #"
                                    << id << " has target " << target(ox, oy, 0)
                                    << " @ (" << ox << "," << oy << ") but "
                                    "number of output target is " << nbTargets
                                    << Utils::cdef << std::endl;

                                throw std::runtime_error("Target::process(): "
                                                         "target out of "
                                                         "range.");
                            }
                        }

                        if (nbOutputs > 1) {
                            std::vector<std::pair<Float_T, size_t> >
                            sortedLabelsValues;
                            sortedLabelsValues.reserve(nbOutputs);

                            for (unsigned int index = 0; index < nbOutputs;
                                 ++index)
                                sortedLabelsValues.push_back(std::make_pair(
                                    value(ox, oy, index), index));

                            // Top-n accuracy sorting
                            std::partial_sort(
                                sortedLabelsValues.begin(),
                                sortedLabelsValues.begin() + mTargetTopN,
                                sortedLabelsValues.end(),
                                std::greater<std::pair<Float_T, size_t> >());

                            for (unsigned int i = 0; i < mTargetTopN; ++i) {
                                estimatedLabels(ox, oy, i)
                                    = sortedLabelsValues[i].second;
                                estimatedLabelsValue(ox, oy, i)
                                    = sortedLabelsValues[i].first;
                            }
                        } else {
                            estimatedLabels(ox, oy, 0)
                                = (value(ox, oy, 0) > 0.5);
                            estimatedLabelsValue(ox, oy, 0)
                                = (estimatedLabels(ox, oy, 0) == 1)
                                      ? value(ox, oy, 0)
                                      : (1.0 - value(ox, oy, 0));
                        }
                    }
                }
            }
        }
    }
}

void N2D2::Target::logEstimatedLabels(const std::string& dirName) const
{
    if (mTargets.dimX() == 1 && mTargets.dimY() == 1)
        return;

    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

#ifndef WIN32
    const int ret = symlink(N2D2_PATH("tools/target_viewer.py"),
                            (dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning
#endif

    if (mDataAsTarget) {
        std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
            <Cell_Frame_Top>(mCell);
        std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
            = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);
        const int size = mStimuliProvider->getBatch().size();

#pragma omp parallel for if (size > 4)
        for (int batchPos = 0; batchPos < size; ++batchPos) {
            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of
                // the set)
                continue;
            }

            // Retrieve estimated labels
            const Tensor4d<Float_T>& values
                = (targetCell) ? targetCell->getOutputs()
                               : targetCellCSpike->getOutputsActivity();
            const std::string imgFile
                = mStimuliProvider->getDatabase().getStimulusName(id);
            const std::string baseName = Utils::baseName(imgFile);
            const std::string fileBaseName = Utils::fileBaseName(baseName);
            const std::string fileExtension = Utils::fileExtension(baseName);

            // Input image
            cv::Mat inputImg = (cv::Mat)mStimuliProvider->getData(0, batchPos);
            cv::Mat inputImg8U;
            inputImg.convertTo(inputImg8U, CV_8U, 255.0);

            std::string fileName = dirPath + "/" + fileBaseName + "_target."
                                   + fileExtension;

            if (!cv::imwrite(fileName, inputImg8U))
                throw std::runtime_error("Unable to write image: " + fileName);

            // Output image
            const cv::Mat outputImg = (cv::Mat)values[batchPos][0];
            cv::Mat outputImg8U;
            outputImg.convertTo(outputImg8U, CV_8U, 255.0);

            fileName = dirPath + "/" + fileBaseName + "_estimated."
                       + fileExtension;

            if (!cv::imwrite(fileName, outputImg8U))
                throw std::runtime_error("Unable to write image: " + fileName);
        }

        return;
    }

    const unsigned int nbTargets = getNbTargets();

#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const Tensor2d<int> target = mTargets[batchPos][0];
        const Tensor2d<int> estimatedLabels = mEstimatedLabels[batchPos][0];
        const Tensor2d<Float_T> estimatedLabelsValue
            = mEstimatedLabelsValue[batchPos][0];

        cv::Mat targetImgHsv(cv::Size(mTargets.dimX(), mTargets.dimY()),
                             CV_8UC3,
                             cv::Scalar(0, 0, 0));
        cv::Mat estimatedImgHsv(cv::Size(mTargets.dimX(), mTargets.dimY()),
                                CV_8UC3,
                                cv::Scalar(0, 0, 0));

        const Tensor2d<int> mask = (mMaskLabelTarget && mMaskedLabel >= 0)
            ? mMaskLabelTarget->getEstimatedLabels()[batchPos][0]
            : Tensor2d<int>();

        for (unsigned int oy = 0; oy < mTargets.dimY(); ++oy) {
            for (unsigned int ox = 0; ox < mTargets.dimX(); ++ox) {
                const int targetHue = (180 * target(ox, oy) / nbTargets
                                       + mLabelsHueOffset) % 180;

                targetImgHsv.at<cv::Vec3b>(oy, ox)
                    = (target(ox, oy) >= 0)
                        ? ((target(ox, oy) != mNoDisplayLabel)
                            ? cv::Vec3f(targetHue, 255, 255)
                            : cv::Vec3f(targetHue, 10, 127)) // no color
                        : cv::Vec3f(0, 0, 127); // ignore = no color

                const int estimatedHue = (180 * estimatedLabels(ox, oy)
                                          / nbTargets + mLabelsHueOffset) % 180;

                estimatedImgHsv.at<cv::Vec3b>(oy, ox)
                    = (mask.empty() || mask(ox, oy) == mMaskedLabel)
                        ? ((estimatedLabels(ox, oy) != mNoDisplayLabel)
                            ? cv::Vec3f(estimatedHue, 255,
                                        255 * estimatedLabelsValue(ox, oy))
                            : cv::Vec3f(estimatedHue, 10, 127)) // no color
                        : cv::Vec3f(0, 0, 127); // not masked = no color
            }
        }

        const double alpha = 0.75;
        const std::string imgFile
            = mStimuliProvider->getDatabase().getStimulusName(id);
        const std::string baseName = Utils::baseName(imgFile);
        const std::string fileBaseName = Utils::fileBaseName(baseName);
        const std::string fileExtension = Utils::fileExtension(baseName);

        // Input image
        cv::Mat inputImg = (cv::Mat)mStimuliProvider->getData(0, batchPos);
        cv::Mat inputImg8U;
        // inputImg.convertTo(inputImg8U, CV_8U, 255.0);

        // Normalize image
        cv::Mat inputImgNorm;
        cv::normalize(
            inputImg.reshape(1), inputImgNorm, 0, 255, cv::NORM_MINMAX);
        inputImg = inputImgNorm.reshape(inputImg.channels());
        inputImg.convertTo(inputImg8U, CV_8U);

        cv::Mat inputImgColor;
        cv::cvtColor(inputImg8U, inputImgColor, CV_GRAY2BGR);

        // Target image
        cv::Mat imgColor;
        cv::cvtColor(targetImgHsv, imgColor, CV_HSV2BGR);

        cv::Mat imgSampled;
        cv::resize(imgColor,
                   imgSampled,
                   cv::Size(mStimuliProvider->getSizeX(),
                            mStimuliProvider->getSizeY()),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);

        cv::Mat imgBlended;
        cv::addWeighted(
            inputImgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

        std::string fileName = dirPath + "/" + fileBaseName + "_target."
                               + fileExtension;

        if (!cv::imwrite(fileName, imgBlended))
            throw std::runtime_error("Unable to write image: " + fileName);

        // Estimated image
        cv::cvtColor(estimatedImgHsv, imgColor, CV_HSV2BGR);

        cv::resize(imgColor,
                   imgSampled,
                   cv::Size(mStimuliProvider->getSizeX(),
                            mStimuliProvider->getSizeY()),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);

        cv::addWeighted(
            inputImgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

        fileName = dirPath + "/" + fileBaseName + "_estimated." + fileExtension;

        if (!cv::imwrite(fileName, imgBlended))
            throw std::runtime_error("Unable to write image: " + fileName);
    }
}

void N2D2::Target::logLabelsLegend(const std::string& fileName) const
{
    if (mDataAsTarget)
        return;

    if (mTargets.dimX() == 1 && mTargets.dimY() == 1)
        return;

    // Legend image
    const unsigned int margin = 5;
    const unsigned int labelWidth = 300;
    const unsigned int cellWidth = 50;
    const unsigned int cellHeight = 50;

    const unsigned int nbTargets = getNbTargets();
    const std::vector<std::string> labelsName = getTargetLabelsName();

    cv::Mat legendImg(cv::Size(cellWidth + labelWidth, cellHeight * nbTargets),
                      CV_8UC3,
                      cv::Scalar(0, 0, 0));

    for (unsigned int target = 0; target < nbTargets; ++target) {
        cv::rectangle(
            legendImg,
            cv::Point(margin, target * cellHeight + margin),
            cv::Point(cellWidth - margin, (target + 1) * cellHeight - margin),
            cv::Scalar((180 * target / nbTargets + mLabelsHueOffset) % 180,
                        255, 255),
            CV_FILLED);

        std::stringstream legendStr;
        legendStr << target << " " << labelsName[target];

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            legendStr.str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::putText(legendImg,
                    legendStr.str(),
                    cv::Point(cellWidth + margin,
                              (target + 1) * cellHeight
                              - (cellHeight - textSize.height) / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 0, 255),
                    2);
    }

    cv::Mat imgColor;
    cv::cvtColor(legendImg, imgColor, CV_HSV2BGR);

    if (!cv::imwrite(mName + "/" + fileName, imgColor))
        throw std::runtime_error("Unable to write image: " + mName + "/"
                                 + fileName);
}

std::pair<int, N2D2::Float_T>
N2D2::Target::getEstimatedLabel(const std::shared_ptr<ROI>& roi,
                                unsigned int batchPos) const
{
    const Tensor4d<int>& labels = mStimuliProvider->getLabelsData();
    const double xRatio = labels.dimX() / (double)mCell->getOutputsWidth();
    const double yRatio = labels.dimY() / (double)mCell->getOutputsHeight();

    const cv::Rect rect = roi->getBoundingRect();
    const unsigned int x0 = rect.tl().x / xRatio;
    const unsigned int y0 = rect.tl().y / yRatio;
    const unsigned int x1 = rect.br().x / xRatio;
    const unsigned int y1 = rect.br().y / yRatio;

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
        = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);
    const Tensor3d<Float_T>& value
        = (targetCell) ? targetCell->getOutputs()[batchPos]
                       : targetCellCSpike->getOutputsActivity()[batchPos];

    if (x1 >= value.dimX() || y1 >= value.dimY())
        throw std::runtime_error(
            "Target::getEstimatedLabel(): bounding box out of range");

    const unsigned int nbOutputs = value.dimZ();
    std::vector<Float_T> bbLabels((nbOutputs > 1) ? nbOutputs : 2, 0.0);

    for (unsigned int oy = y0; oy <= y1; ++oy) {
        for (unsigned int ox = x0; ox <= x1; ++ox) {
            if (nbOutputs > 1) {
                for (unsigned int index = 0; index < nbOutputs; ++index)
                    bbLabels[index] += value(ox, oy, index);
            } else {
                bbLabels[0] += 1.0 - value(ox, oy, 0);
                bbLabels[1] += value(ox, oy, 0);
            }
        }
    }

    const std::vector<Float_T>::const_iterator it
        = std::max_element(bbLabels.begin(), bbLabels.end());
    return std::make_pair(it - bbLabels.begin(),
                          (*it) / ((x1 - x0 + 1) * (y1 - y0 + 1)));
}
