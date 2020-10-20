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

#include "Transformation/LabelSliceExtractionTransformation.hpp"
#include "Transformation/SliceExtractionTransformation.hpp"

const char* N2D2::LabelSliceExtractionTransformation::Type = "LabelSliceExtraction";

N2D2::LabelSliceExtractionTransformation::LabelSliceExtractionTransformation(
    unsigned int width, unsigned int height, int label)
    : mWidth(width),
      mHeight(height),
      mLabel(label),
      mLastLabel(label),
      mSlicesMargin(this, "SlicesMargin", 0),
      mKeepComposite(this, "KeepComposite", false),
      mRandomRotation(this, "RandomRotation", false),
      mRandomRotationRange(this, "RandomRotationRange",
                           std::vector<double>({0.0, 360.0}))
{
    // ctor
}

N2D2::LabelSliceExtractionTransformation::LabelSliceExtractionTransformation(
    const LabelSliceExtractionTransformation& trans)
    : mWidth(trans.mWidth),
      mHeight(trans.mHeight),
      mLabel(trans.mLabel),
      mLastLabel(trans.mLastLabel),
      mSlicesMargin(this, "SlicesMargin", trans.mSlicesMargin),
      mKeepComposite(this, "KeepComposite", trans.mKeepComposite),
      mRandomRotation(this, "RandomRotation", trans.mRandomRotation),
      mRandomRotationRange(this, "RandomRotationRange",
                           trans.mRandomRotationRange)
{
    // copy-ctor
}

void N2D2::LabelSliceExtractionTransformation::apply(
    cv::Mat& frame,
    cv::Mat& labels,
    std::vector<std::shared_ptr<ROI> >& labelsROI,
    int id)
{
    int lastLabel = mLabel;

    if (mLabel == -1) {
        // Find all the different labels
        std::vector<int> uniqueLabels;

        if (id >= 0) {
            // if an ID is provided, look for cached value
            bool found = false;

#pragma omp critical(LabelSliceExtractionTransformation__apply)
            {
                const std::map<int, std::vector<int> >::const_iterator it
                    = mUniqueLabels.find(id);

                if (it != mUniqueLabels.end()) {
                    // Cache exists: use the cached value
                    uniqueLabels = (*it).second;
                    found = true;
                }
            }

            if (!found) {
                // Cache doesn't exit: compute the value and store it in cache
                uniqueLabels = unique(labels);

#pragma omp critical(LabelSliceExtractionTransformation__apply)
                mUniqueLabels.insert(std::make_pair(id, uniqueLabels));
            }
        } else
            uniqueLabels = unique(labels);

        // Choose a random label
        lastLabel
            = uniqueLabels[Random::randUniform(0, uniqueLabels.size() - 1)];
    }

    Pos_T pos;
    bool validPos = false;

    std::stringstream cacheFile;
    cacheFile << "_cache/lseTrans_id" << id << "_label" << lastLabel << ".bin";

    if (id >= 0 && std::ifstream(cacheFile.str().c_str()).good()) {
        // loadLabelPosCache(cacheFile.str(), labelPos);
        validPos = loadLabelRandomPos(cacheFile.str(), pos);
    } else {
        std::vector<Pos_T> labelPos;

        // Mask with the chosen label
        const cv::Mat matLabel = (labels == lastLabel);

        // cv::Mat debug = matLabel.clone()/4;

        // Find valid slice positions
        if (mElementErode.empty()) {
#pragma omp critical(LabelSliceExtractionTransformation__apply)
            if (mElementErode.empty()) {
                mElementErode = cv::getStructuringElement(
                    cv::MORPH_RECT,
                    cv::Size(std::max(1, (int)mWidth + 2 * mSlicesMargin),
                             std::max(1, (int)mHeight + 2 * mSlicesMargin)),
                    cv::Point(-1, -1));
            }
        }

        if (mElementDilate.empty()) {
#pragma omp critical(LabelSliceExtractionTransformation__apply)
            if (mElementDilate.empty()) {
                mElementDilate = cv::getStructuringElement(
                    cv::MORPH_RECT,
                    cv::Size(std::max(1, -((int)mWidth + 2 * mSlicesMargin)),
                             std::max(1, -((int)mHeight + 2 * mSlicesMargin))),
                    cv::Point(-1, -1));
            }
        }

        cv::Mat matLabelDilate;
        cv::Mat matLabelErode;

        if (mElementDilate.rows > 1 || mElementDilate.cols > 1)
            cv::dilate(matLabel, matLabelDilate, mElementDilate);
        else
            matLabelDilate = matLabel;

        if (mElementErode.rows > 1 || mElementErode.cols > 1)
            cv::erode(matLabelDilate, matLabelErode, mElementErode);
        else
            matLabelErode = matLabelDilate;

        // debug+= matLabelErode/4;

        for (int i = mHeight / 2,
                 iMax = matLabel.rows - (int)std::ceil(mHeight / 2.0) + 1;
             i < iMax;
             ++i) {
            for (int j = mWidth / 2,
                     jMax = matLabel.cols - (int)std::ceil(mWidth / 2.0) + 1;
                 j < jMax;
                 ++j) {
                if (matLabelErode.at<unsigned char>(i, j)) {
                    labelPos.push_back(Pos_T(j, i));
                    // debug.at<unsigned char>(i,j)+= 64;
                }
            }
        }

        if (id >= 0)
            saveLabelPosCache(cacheFile.str(), labelPos);

        if (!labelPos.empty()) {
            const unsigned int index
                = Random::randUniform(0, labelPos.size() - 1);
            pos = labelPos[index];
            validPos = true;
        }
    }

    if (!validPos) {
        std::cout << Utils::cwarning << "Warning: no valid slice for label ID "
                  << lastLabel << " in label slice extraction (stimulus "
                  << id << ")" << Utils::cdef << std::endl;

        // No valid pos, ignore this frame
        pos.x = mWidth / 2;
        pos.y = mHeight / 2;
        lastLabel = -1;

        if (frame.rows < (int)mHeight || frame.cols < (int)mWidth)
            cv::resize(frame, frame, cv::Size(mWidth, mHeight));
    }

    // Extract the label slice
    const double rotation = (mRandomRotation)
        ? Random::randUniform(*(mRandomRotationRange->begin()),
                              *(mRandomRotationRange->begin() + 1))
        : 0.0;

    const cv::Rect lastSlice
        = SliceExtractionTransformation::extract(pos.x - mWidth / 2,
                                                 pos.y - mHeight / 2,
                                                 mWidth,
                                                 mHeight,
                                                 rotation,
                                                 cv::BORDER_CONSTANT,
                                                 cv::Scalar::all(0),
                                                 frame,
                                                 labels,
                                                 labelsROI,
                                                 id);

    /*
        debug(lastSlice)+= 50;
        debug.at<unsigned char>(pos.y, pos.x) = 0;

        cv::namedWindow("debug", CV_WINDOW_NORMAL);
        cv::imshow("debug", debug);
        cv::waitKey(0);
    */

    if (!mKeepComposite)
        labels = cv::Mat(1, 1, CV_32S, cv::Scalar(lastLabel));
    else if (!validPos)
        labels = cv::Mat(cv::Size(mWidth, mHeight), CV_32S, cv::Scalar(-1));

#pragma omp critical(LabelSliceExtractionTransformation__apply)
    {
        mLastSlice = lastSlice;
        mLastLabel = lastLabel;
    }
}

std::vector<int> N2D2::LabelSliceExtractionTransformation::unique(const cv::Mat
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

bool N2D2::LabelSliceExtractionTransformation::loadLabelRandomPos(
    const std::string& fileName, Pos_T& pos)
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

void N2D2::LabelSliceExtractionTransformation::loadLabelPosCache(
    const std::string& fileName, std::vector<Pos_T>& labelPos)
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

void N2D2::LabelSliceExtractionTransformation::saveLabelPosCache(
    const std::string& fileName, const std::vector<Pos_T>& labelPos)
{
    std::ofstream cache(fileName.c_str(), std::ios::binary);

    if (!cache.good())
        throw std::runtime_error("Could not write cache file: " + fileName);

    cache.write(reinterpret_cast<const char*>(&labelPos[0]),
                labelPos.size() * sizeof(labelPos[0]));

    if (!cache.good())
        throw std::runtime_error("Error writing cache file: " + fileName);
}

N2D2::LabelSliceExtractionTransformation::~LabelSliceExtractionTransformation()
{
    
}
