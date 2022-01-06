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

#include "Transformation/LabelFilterTransformation.hpp"

#include "Database/Database.hpp"
#include "StimuliProvider.hpp"

namespace N2D2 {
class ParallelFilter : public cv::ParallelLoopBody {
public:
    ParallelFilter(
        cv::Mat& labels_,
        const std::vector<int>& labelsToFilter_,
        const LabelFilterTransformation::LabelFilter& filter_,
        const int& defaultLabel_)
        : labels(labels_),
          labelsToFilter(labelsToFilter_),
          filter(filter_),
          defaultLabel(defaultLabel_)
    {}
    virtual void operator ()(const cv::Range& range) const override {
        for (int r = range.start; r < range.end; r++) {
            int i = r / labels.cols;
            int j = r % labels.cols;
            int& value = labels.ptr<int>(i)[j];

            if (std::find(labelsToFilter.begin(), labelsToFilter.end(), value)
                != labelsToFilter.end())
            {
                if (filter == LabelFilterTransformation::Remove)
                    value = defaultLabel;
                else if (filter == LabelFilterTransformation::Merge)
                    value = *(labelsToFilter.begin());
            }
            else {
                if (filter == LabelFilterTransformation::Keep)
                    value = defaultLabel;
            }
        }
    }
    ParallelFilter& operator=(const ParallelFilter&) {
        return *this;
    };
private:
    cv::Mat& labels;
    const std::vector<int>& labelsToFilter;
    const LabelFilterTransformation::LabelFilter& filter;
    const int& defaultLabel;
};
}

const char* N2D2::LabelFilterTransformation::Type = "LabelFilter";

N2D2::LabelFilterTransformation::LabelFilterTransformation(
    const std::vector<std::string>& labels)
    : mLabels(labels),
      mFilter(this, "Filter", Remove),
      mDefaultLabel(this, "DefaultLabel", -2)
{
    // ctor
}

N2D2::LabelFilterTransformation::LabelFilterTransformation(
    const LabelFilterTransformation& trans)
    : mLabels(trans.mLabels),
      mFilter(this, "Filter", trans.mFilter),
      mDefaultLabel(this, "DefaultLabel", trans.mDefaultLabel)
{
    // copy-ctor
}

void N2D2::LabelFilterTransformation::apply(cv::Mat& /*frame*/,
                                     cv::Mat& labels,
                                     std::vector
                                     <std::shared_ptr<ROI> >& labelsROI,
                                     int /*id*/)
{
    const Database& database = mStimuliProvider->getDatabase();
    const std::vector<int>& labelsIDs = database.getLabelsIDs(mLabels);
    const int defaultLabel = (mDefaultLabel < -1)
        ? database.getDefaultLabelID()
        : mDefaultLabel;

    // labels processing
    ParallelFilter parallelFilter(labels, labelsIDs, mFilter, defaultLabel);
    cv::parallel_for_(
        cv::Range(0, labels.rows * labels.cols),
        parallelFilter);

    // labelsROI processing
    for (std::vector<std::shared_ptr<ROI> >::iterator
        it = labelsROI.begin(), itEnd = labelsROI.end(); it != itEnd; )
    {
        if (std::find(labelsIDs.begin(), labelsIDs.end(), (*it)->getLabel())
            != labelsIDs.end())
        {
            if (mFilter == Remove) {
                it = labelsROI.erase(it);
                continue;
            }
            else if (mFilter == Merge) {
                (*it)->setLabel(*(labelsIDs.begin()));
            }
        }
        else {
            if (mFilter == Keep) {
                it = labelsROI.erase(it);
                continue;
            }
        }

        ++it;
    }
}
