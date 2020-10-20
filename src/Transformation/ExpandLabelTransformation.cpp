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

#include "Transformation/ExpandLabelTransformation.hpp"

const char* N2D2::ExpandLabelTransformation::Type = "ExpandLabel";

N2D2::ExpandLabelTransformation::ExpandLabelTransformation()
{
    // ctor
}

void N2D2::ExpandLabelTransformation::apply(cv::Mat& frame,
                                         cv::Mat& labels,
                                         std::vector
                                         <std::shared_ptr<ROI> >& /*labelsROI*/,
                                         int /*id*/)
{
    if (labels.rows != 1 || labels.cols != 1) {
        throw std::runtime_error("ExpandLabelTransformation::apply(): Label is"
            " already expanded!");
    }

    labels = cv::Mat(cv::Size(frame.cols, frame.rows),
                     CV_32S, cv::Scalar(labels.at<int>(0, 0)));
}

N2D2::ExpandLabelTransformation::~ExpandLabelTransformation() {
    
}
