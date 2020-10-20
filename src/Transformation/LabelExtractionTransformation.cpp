/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Benjamin BERTELONE (benjamin.bertelone@cea.fr)
                    Alexandre CARBON (alexandre.carbon@cea.fr)

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

#include "Transformation/LabelExtractionTransformation.hpp"
#include "ROI/ROI.hpp"

const char* N2D2::LabelExtractionTransformation::Type = "LabelExtraction";

N2D2::LabelExtractionTransformation::LabelExtractionTransformation(
    const std::string& widths,
    const std::string& heights,
    int label,
    std::string distributions)
    : mLabel(label)
{
    //ctor
    std::vector<std::string> vec_widths;
    std::vector<std::string> vec_heights;

    vec_widths = Utils::split(widths, ",");
    vec_heights = Utils::split(heights, ",");

    if (vec_widths.size() <= 0)
        throw std::runtime_error("LabelExtractionTransformation: at least one "
                                 "patch size is needed.");

    if (vec_widths.size() != vec_heights.size())
        throw std::runtime_error("LabelExtractionTransformation: there must be "
                                 "the same number of widths and heights.");

    unsigned int nbPatch = vec_widths.size();

    try
    {
        for (unsigned int i = 0; i < nbPatch; ++i) {
            mWidths.push_back(std::stoi(vec_widths[i]));
            mHeights.push_back(std::stoi(vec_heights[i]));
        }
    }
    catch (const std::exception& /*e*/)
    {
        throw std::runtime_error("LabelExtractionTransformation: Error when "
                                 "reading widths or heights.");
    }

    std::fill(mDuration, mDuration + 1000, 0);

    if (distributions == "Auto")
        mAutoWantedLabelDistribution = true;
    else {
        mAutoWantedLabelDistribution = false;
        std::vector<std::string> vec_distributions;

        vec_distributions = Utils::split(distributions, ",");

        if (vec_distributions.size() == 0)
            mAutoWantedLabelDistribution = true;
        else {
            for (unsigned int i = 0; i < vec_distributions.size(); ++i) {
                std::vector<std::string> t
                    = Utils::split(vec_distributions[i], ":");
                if (t.size() != 2)
                    throw std::runtime_error(
                        "LabelExtractionTransformation: distributions must be "
                        "\"Auto\" or "
                        "\"<label>:<percent>,<label-range>:<percent>,...\" ie "
                        "\"0:0.8,1:0.2\" in " + vec_distributions[i]);

                size_t sepPos = t[0].find("-");
                if (sepPos == std::string::npos) {
                    try
                    {
                        mWantedLabelDistribution[std::stoi(t[0])]
                            = std::stof(t[1]);
                    }
                    catch (const std::exception& /*e*/)
                    {
                        throw std::runtime_error(
                            "LabelExtractionTransformation: Error when reading "
                            "distributions " + t[0] + ":" + t[1]);
                    }
                } else {
                    int startLabel, endLabel;
                    float value;

                    try
                    {
                        startLabel = std::stoi(t[0]);
                        endLabel = std::stoi(t[0].substr(sepPos + 1));
                        value = std::stof(t[1]);
                    }
                    catch (const std::exception& /*e*/)
                    {
                        throw std::runtime_error(
                            "LabelExtractionTransformation: Error when reading "
                            "distributions " + t[0] + ":" + t[1]);
                    }

                    if (startLabel > endLabel)
                        throw std::runtime_error("LabelExtractionTransformation"
                                                 ": Error, start labell is "
                                                 "greater than end label "
                                                 + t[0] + ":" + t[1]);

                    value /= endLabel - startLabel + 1;
                    for (int l = startLabel; l <= endLabel; ++l)
                        mWantedLabelDistribution[l] = value;
                }
            }
        }
    }
}

void
N2D2::LabelExtractionTransformation::apply(cv::Mat& frame,
                                           cv::Mat& labels,
                                           std::vector
                                           <std::shared_ptr<ROI> >& labelsROI,
                                           int id)
{
    std::chrono::high_resolution_clock::time_point startTime
        = std::chrono::high_resolution_clock::now();
    static int dspl = 0;
    static int durationIndex = 0;

    // Choose a random patch size
    unsigned int patch;
    unsigned int width;
    unsigned int height;
    unsigned int smartErrodeWidth = 0;
    unsigned int smartErrodeHeight = 0;

    if (id >= 0) {
        std::vector<int> availablePatch;
        std::vector<std::shared_ptr<ROI> >::const_iterator it_roi;

        int maxErrodeWidth = 0;
        int maxErrodeHeight = 0;

        for (unsigned int i = 0; i < mWidths.size(); ++i) {
            bool roi_available = false;
            for (it_roi = labelsROI.begin(); it_roi != labelsROI.end();
                 it_roi++) {
                if ((*it_roi)->getLabel() == -1)
                    continue;

                const cv::Rect br = (*it_roi)->getBoundingRect();

                if (mWidths[i] <= br.width && br.width < 2 * mWidths[i]
                    && mHeights[i] <= br.height
                    && br.height < 2 * mHeights[i]) {
                    roi_available = true;
                    break;
                } else {
                    maxErrodeWidth = br.width - 1;
                    maxErrodeHeight = br.height - 1;
                }
            }

            if (roi_available)
                availablePatch.push_back(i);
        }

        if (!availablePatch.empty()) {
            patch = availablePatch
                [Random::randUniform(0, availablePatch.size() - 1)];

            width = mWidths[patch];
            height = mHeights[patch];

            smartErrodeWidth = width;
            smartErrodeHeight = height;
        } else if (maxErrodeWidth > 0 && maxErrodeHeight > 0) {
            patch = 0;

            width = mWidths[patch];
            height = mHeights[patch];

            smartErrodeWidth = maxErrodeWidth;
            smartErrodeHeight = maxErrodeHeight;
        } else // No ROI
        {
            patch = Random::randUniform(0, mWidths.size() - 1);

            width = mWidths[patch];
            height = mHeights[patch];

            smartErrodeWidth = width;
            smartErrodeHeight = height;
        }

        /*std::vector<int>::iterator it = availablePatch.begin();
        std::cout << Utils::cnotice << "Patch " << patch << " selected on [[ ";
        for(;it != availablePatch.end(); it++)
            std::cout << *it << " ";
        std::cout << "]]." << Utils::cdef << std::endl;*/
    } else {
        patch = Random::randUniform(0, mWidths.size() - 1);

        width = mWidths[patch];
        height = mHeights[patch];

        smartErrodeWidth = width;
        smartErrodeHeight = height;
    }

    int selectedLabel = -2;

    bool selectedFromCache = false;

    if (mLabel == -1) {
        std::vector<int> uniqueLabels;

        if (id >= 0 && mUniqueLabels.count(id) == 1
            && mUniqueLabels[id].count(patch) == 1) {
            uniqueLabels = mUniqueLabels[id][patch];

            selectedFromCache = true;

            // std::cout << Utils::cnotice << "Selecting uniqueLabels from cache
            // id" << id << "_patch" << patch << Utils::cdef << std::endl;
        } else {
            bool unknowLabelInWantedLabelDistribution = false;
            smartErode(
                labels, labelsROI, smartErrodeWidth, smartErrodeHeight, id);

            // Find all the different labels
            uniqueLabels = unique(labels);

            // Adding new labels to the label distribution map
            std::vector<int>::iterator uniqueLabels_it = uniqueLabels.begin();
            for (uniqueLabels_it = uniqueLabels.begin();
                 uniqueLabels_it != uniqueLabels.end();
                 uniqueLabels_it++) {
                if (mLabelDistribution.count(*uniqueLabels_it) == 0)
                    mLabelDistribution[*uniqueLabels_it] = 0;
                if (!mWantedLabelDistribution.count(*uniqueLabels_it))
                    unknowLabelInWantedLabelDistribution = true;
            }

            // Updatting wanted label distribution if set to Auto
            if (mAutoWantedLabelDistribution) {
                std::map<int, int>::iterator mLabelDistribution_it
                    = mLabelDistribution.begin();
                mWantedLabelDistribution.clear();
                for (; mLabelDistribution_it != mLabelDistribution.end();
                     mLabelDistribution_it++)
                    mWantedLabelDistribution[mLabelDistribution_it->first]
                        = 1 / (float)mLabelDistribution.size();
            } else if (unknowLabelInWantedLabelDistribution) {
                throw std::runtime_error("LabelExtractionTransformation: "
                                         "Unknow label in distribution.");
            }

            if (id >= 0) {
                mUniqueLabels[id][patch] = uniqueLabels;
                // std::cout << Utils::cnotice << "Creating cache of
                // uniqueLabels id" << id << "_patch" << patch << Utils::cdef <<
                // std::endl;
            }
        }

        // Choose a random label
        selectedLabel = getRandomLabel(uniqueLabels);

        /*if(selectedLabel == -2)
            std::cout << Utils::cwarning << "Warning: no label ID in label
           extraction" << Utils::cdef << std::endl;*/

        if (selectedLabel >= 0)
            mLabelDistribution[selectedLabel]++;
    } else
        selectedLabel = mLabel;

    Pos_T pos;
    bool validPos = false;

    std::stringstream cacheFile;
    cacheFile << "_cache/leTrans_labelPos_id" << id << "_label" << selectedLabel
              << "_patch" << patch << ".bin";

    if (selectedLabel >= 0) {
        if (id >= 0 && std::ifstream(cacheFile.str().c_str()).good()) {
            // std::cout << Utils::cnotice << "Selecting pos from cache " <<
            // cacheFile.str() << Utils::cdef << std::endl;
            validPos = loadLabelRandomPos(cacheFile.str(), pos);
        } else {
            std::vector<Pos_T> labelPos;

            if (selectedFromCache)
                smartErode(
                    labels, labelsROI, smartErrodeWidth, smartErrodeHeight, id);

            // Mask with the chosen label
            const cv::Mat matLabel = (labels == selectedLabel);

            for (int i = height / 2,
                     iMax = matLabel.rows - (int)std::ceil(height / 2.0) + 1;
                 i < iMax;
                 ++i) {
                for (int j = width / 2,
                         jMax = matLabel.cols - (int)std::ceil(width / 2.0) + 1;
                     j < jMax;
                     ++j) {
                    if (matLabel.at<unsigned char>(i, j)) {
                        labelPos.push_back(Pos_T(j, i));
                    }
                }
            }

            if (id >= 0) {
                saveLabelPosCache(cacheFile.str(), labelPos);
                // std::cout << Utils::cnotice << "Creating cache of pos " <<
                // cacheFile.str() << Utils::cdef << std::endl;
            }

            if (!labelPos.empty()) {
                const unsigned int index
                    = Random::randUniform(0, labelPos.size() - 1);
                pos = labelPos[index];
                validPos = true;
            }
        }
    }

    if (!validPos) {
        if (selectedLabel >= 0) {
            std::cout << Utils::cwarning
                      << "Warning: no valid slice for label ID "
                      << selectedLabel << " in label extraction" << std::endl;

            std::cout << "\tpatch : " << patch << "(" << width << "x" << height
                      << ")" << std::endl;
            if (smartErrodeWidth != width)
                std::cout << "\tUsing resized erode." << std::endl;

            std::cout << Utils::cdef;
        }

        // No valid pos, ignore this frame
        pos.x = width / 2;
        pos.y = height / 2;
        selectedLabel = -1;
    }

    cv::Rect slice
        = cv::Rect(pos.x - width / 2, pos.y - height / 2, width, height);

    mLastSlice = slice;
    mLastLabel = selectedLabel;

    frame = frame(slice);
    labels = cv::Mat(1, 1, CV_32S, cv::Scalar(selectedLabel));

    return;

    double timeElapsed
        = std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::high_resolution_clock::now() - startTime).count();

    mDuration[durationIndex] = timeElapsed;
    durationIndex = (durationIndex + 1) % 1000;

    dspl = (dspl + 1) % 2000;
    if (dspl == 1) {
        std::map<int, int>::iterator labelDistribution_it;
        int totalSum = 0;

        std::ios flags(NULL);
        flags.copyfmt(std::cout);

        std::cout << Utils::cnotice << "Label distribution : [[ " << std::fixed
                  << std::setprecision(10);

        for (labelDistribution_it = mLabelDistribution.begin();
             labelDistribution_it != mLabelDistribution.end();
             labelDistribution_it++)
            totalSum += (*labelDistribution_it).second;

        for (labelDistribution_it = mLabelDistribution.begin();
             labelDistribution_it != mLabelDistribution.end();
             labelDistribution_it++) {
            std::cout << (*labelDistribution_it).first << " "
                      << ((*labelDistribution_it).second / (float)totalSum
                          * 100) << "% (" << (*labelDistribution_it).second
                      << ") ";
        }

        std::cout << "]] total : " << totalSum;

        double meanTime = 0;
        for (int i = 0; i < 1000; ++i)
            meanTime += mDuration[i];

        std::cout << ", mean execution time : " << meanTime << "ms";

        std::cout << Utils::cdef << std::endl;

        std::cout.copyfmt(flags);
    }
}

void N2D2::LabelExtractionTransformation::smartErode(
    cv::Mat& labels,
    std::vector<std::shared_ptr<ROI> >& labelsROI,
    int width,
    int height,
    int id)
{

    if (id >= 0) {
        std::vector<std::shared_ptr<ROI> >::const_iterator it_roi
            = labelsROI.begin();
        for (; it_roi != labelsROI.end(); it_roi++) {
            const cv::Rect br = (*it_roi)->getBoundingRect();

            cv::Rect dilate_br = br;

            dilate_br.x -= width / 2;
            if (dilate_br.x < 0)
                dilate_br.x = 0;
            dilate_br.y -= height / 2;
            if (dilate_br.y < 0)
                dilate_br.y = 0;

            if (dilate_br.width + width < (int)labels.cols)
                dilate_br.width += width;
            else
                dilate_br.width = labels.cols - dilate_br.x;

            if (dilate_br.height + height < (int)labels.rows)
                dilate_br.height += height;
            else
                dilate_br.height = labels.rows - dilate_br.y;

            cv::rectangle(
                labels, dilate_br.tl(), dilate_br.br(), cv::Scalar(-1), -1);

            if (br.width > 2 * width || br.height > 2 * height)
                continue;

            if ((*it_roi)->getLabel() == -1)
                continue;

            cv::Rect erode_br = br;

            erode_br.width -= width;
            if (erode_br.width < 0)
                continue;
            erode_br.height -= height;
            if (erode_br.height < 0)
                continue;

            erode_br.x += width / 2;
            erode_br.y += height / 2;

            cv::rectangle(labels,
                          erode_br.tl(),
                          erode_br.br(),
                          cv::Scalar((*it_roi)->getLabel()),
                          -1);
        }
    } else {
        cv::Mat matLabelErode;
        cv::Mat matLabelDilate;
        cv::Mat matLabel;

        labels.convertTo(matLabel, CV_16SC1);

        const cv::Mat element = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(width, height), cv::Point(-1, -1));

        cv::erode(matLabel, matLabelErode, element);
        cv::dilate(matLabel, matLabelDilate, element);

        const cv::Mat matLabelMask = (matLabelErode == matLabelDilate);

        // To be tested
        cv::Mat labelsNew(labels.size(), labels.type(), cv::Scalar(-1));
        labels.copyTo(labelsNew, matLabelMask);
        labels = labelsNew;

        /*
        cv::MatIterator_<int> it1 = labels.begin<int>(), it1_end =
        labels.end<int>();
        cv::MatConstIterator_<unsigned char> it2 = matLabelMask.begin<unsigned
        char>();

        for(; it1 != it1_end; ++it1, ++it2)
            if(!(*it2))
                (*it1) = -1;
         */
    }

    cv::rectangle(labels,
                  cv::Point(0, 0),
                  cv::Point(width / 2, labels.rows),
                  cv::Scalar(-1),
                  -1);
    cv::rectangle(labels,
                  cv::Point(labels.cols - width / 2, 0),
                  cv::Point(labels.cols, labels.rows),
                  cv::Scalar(-1),
                  -1);
    cv::rectangle(labels,
                  cv::Point(0, 0),
                  cv::Point(labels.cols, height / 2),
                  cv::Scalar(-1),
                  -1);
    cv::rectangle(labels,
                  cv::Point(0, labels.rows - height / 2),
                  cv::Point(labels.cols, labels.rows),
                  cv::Scalar(-1),
                  -1);
}

int N2D2::LabelExtractionTransformation::getRandomLabel(std::vector
                                                        <int>& labels)
{
    if (labels.size() == 0)
        return -2;
    if (labels.size() == 1)
        return labels[0];

    float ponderate_sum = 0;

    std::vector<int>::iterator labels_it;
    for (labels_it = labels.begin(); labels_it != labels.end(); labels_it++)
        if (mWantedLabelDistribution[*labels_it] != 0)
            ponderate_sum
                += mLabelDistribution[*labels_it]
                   / std::pow(mWantedLabelDistribution[*labels_it], 2);

    ponderate_sum /= (mWantedLabelDistribution.size() - 1);

    if (ponderate_sum > 0) {
        unsigned int rnd = Random::mtRand();

        for (labels_it = labels.begin(); labels_it != labels.end();
             labels_it++) {
            if (mWantedLabelDistribution[*labels_it] == 0)
                continue;

            float pt = 1 - mLabelDistribution[*labels_it]
                           / std::pow(mWantedLabelDistribution[*labels_it], 2)
                           / ponderate_sum;
            pt *= MT_RAND_MAX;

            if (rnd <= pt)
                return *labels_it;
            else
                rnd -= pt;
        }
    }

    return labels[Random::randUniform(0, labels.size() - 1)];
}

std::vector<int> N2D2::LabelExtractionTransformation::unique(const cv::Mat
                                                             & mat) const
{
    std::vector<int> uniqueValues;

    for (int i = 0; i < mat.rows; ++i) {
        const int* rowPtr = mat.ptr<int>(i);

        for (int j = 0; j < mat.cols; ++j) {
            if (rowPtr[j] >= 0 && std::find(uniqueValues.begin(),
                                            uniqueValues.end(),
                                            rowPtr[j]) == uniqueValues.end())
                uniqueValues.push_back(rowPtr[j]);
        }
    }

    return uniqueValues;
}

bool N2D2::LabelExtractionTransformation::loadLabelRandomPos(const std::string
                                                             & fileName,
                                                             Pos_T& pos)
{
    std::ifstream cache(fileName.c_str(), std::ios::binary);

    if (!cache.good())
        throw std::runtime_error("Could not read cache file: " + fileName);

    cache.seekg(0, cache.end); // Get end-of-file position
    const unsigned int posSize = cache.tellg() / sizeof(pos);
    cache.seekg(0); // Get back to beginning

    if (posSize > 0) {
        const unsigned int index = Random::randUniform(0, posSize - 1);

        cache.seekg(index * sizeof(pos));
        cache.read(reinterpret_cast<char*>(&pos), sizeof(pos));

        if (!cache.good())
            throw std::runtime_error("Error reading cache file: " + fileName);

        return true;
    }

    return false;
}

void N2D2::LabelExtractionTransformation::loadLabelPosCache(const std::string
                                                            & fileName,
                                                            std::vector
                                                            <Pos_T>& labelPos)
{
    std::ifstream cache(fileName.c_str(), std::ios::binary);

    if (!cache.good())
        throw std::runtime_error("Could not read cache file: " + fileName);

    cache.seekg(0, cache.end); // Get end-of-file position
    const unsigned int fileSize = cache.tellg();
    cache.seekg(0); // Get back to beginning

    labelPos.resize(fileSize / sizeof(labelPos[0]));
    cache.read(reinterpret_cast<char*>(&labelPos[0]),
               labelPos.size() * sizeof(labelPos[0]));

    if (!cache.good() || cache.peek() != EOF)
        throw std::runtime_error("Error reading cache file: " + fileName);
}

void N2D2::LabelExtractionTransformation::saveLabelPosCache(const std::string
                                                            & fileName,
                                                            const std::vector
                                                            <Pos_T>& labelPos)
{
    std::ofstream cache(fileName.c_str(), std::ios::binary);

    if (!cache.good())
        throw std::runtime_error("Could not write cache file: " + fileName);

    cache.write(reinterpret_cast<const char*>(&labelPos[0]),
                labelPos.size() * sizeof(labelPos[0]));

    if (!cache.good())
        throw std::runtime_error("Error writing cache file: " + fileName);
}

N2D2::LabelExtractionTransformation::~LabelExtractionTransformation() {
    
}
