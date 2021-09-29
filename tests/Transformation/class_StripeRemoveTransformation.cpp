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

#include "FloatT.hpp"
#include "ROI/RectangularROI.hpp"
#include "Transformation/StripeRemoveTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(StripeRemoveTransformation,
             apply,
             (int axis,
              unsigned int offset,
              unsigned int length,
              unsigned int nbIterations,
              unsigned int stepOffset),
             std::make_tuple(0, 0U, 10U, 1U, 0U),
             std::make_tuple(0, 256U, 128U, 1U, 0U),
             std::make_tuple(0, 256U, 256U, 1U, 0U),
             std::make_tuple(0, 64U, 64U, 2U, 0U),
             std::make_tuple(0, 64U, 64U, 2U, 64U),
             std::make_tuple(1, 0U, 10U, 1U, 0U),
             std::make_tuple(1, 256U, 128U, 1U, 0U),
             std::make_tuple(1, 256U, 256U, 1U, 0U),
             std::make_tuple(1, 64U, 64U, 2U, 0U),
             std::make_tuple(1, 64U, 64U, 2U, 64U))
{
    StripeRemoveTransformation trans(axis, offset, length);
    trans.setParameter<unsigned int>("NbIterations", nbIterations);
    trans.setParameter<unsigned int>("StepOffset", stepOffset);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));

    for (unsigned int i = 0; i < nbIterations; ++i) {
        RectangularROI<int> stripe = (axis == 0)
            ? RectangularROI<int>(255,
                cv::Point(offset + i * (length + stepOffset), 0), length, 512)
            : RectangularROI<int>(255,
                cv::Point(0, offset + i * (length + stepOffset)), 512, length);

        stripe.append(labels);
    }

    trans.apply(labels);

    unsigned int width, height;
    std::tie(width, height) = trans.getOutputsSize(512, 512);

    ASSERT_EQUALS(labels.cols, (int)width);
    ASSERT_EQUALS(labels.rows, (int)height);

    if (axis == 0) {
        ASSERT_EQUALS(labels.cols, 512 - (int)(nbIterations * length));
        ASSERT_EQUALS(labels.rows, 512);
    }
    else {
        ASSERT_EQUALS(labels.cols, 512);
        ASSERT_EQUALS(labels.rows, 512 - (int)(nbIterations * length));
    }

    ASSERT_EQUALS(cv::countNonZero(labels), 0);
}

RUN_TESTS()
