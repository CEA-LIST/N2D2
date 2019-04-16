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

/**
 * @file      Transformation.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Interface that must be implemented by all derived transformation
 *classes.
 *
 * @details   These class handle the application of all kind of transformation
*/

#ifndef N2D2_TRANSFORMATION_H
#define N2D2_TRANSFORMATION_H

#ifdef OPENCV_USE_OLD_HEADERS       //  before OpenCV 2.2.0
    #include "cv.h"
    #include "highgui.h"
#else
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 2
        #include "opencv2/core/core.hpp"
        #include "opencv2/imgproc/imgproc.hpp"
        #include "opencv2/highgui/highgui.hpp"
    #elif CV_MAJOR_VERSION >= 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
        #include "opencv2/highgui.hpp"
    #endif
#endif

#include "ROI/ROI.hpp"
#include "containers/Tensor.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {

class CompositeTransformation;

template <class T>
struct opencv_data_type {
    static const int value = -1;  // Dummy value by default
};

template <>
struct opencv_data_type<float> {
    static const int value = CV_32F;
};

template <>
struct opencv_data_type<double> {
    static const int value = CV_64F;
};

class Transformation : public Parameterizable {
public:
    inline void apply(cv::Mat& frame, int id = -1);
    inline void apply(cv::Mat& frame, cv::Mat& labels, int id = -1);
    virtual void apply(cv::Mat& frame,
                       cv::Mat& labels,
                       std::vector<std::shared_ptr<ROI> >& labelsROI,
                       int id = -1) = 0;
    virtual void reverse(cv::Mat& /*frame*/,
                         cv::Mat& /*labels*/,
                         std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
                         int /*id*/ = -1) {};
    template <class T1> void apply(Tensor<T1>& frame, int id = -1);
    template <class T1>
    Tensor<T1> apply(const Tensor<T1>& frame, int id = -1);
    template <class T1, class T2>
    void apply(Tensor<T1>& frame, Tensor<T2>& labels, int id = -1);
    std::shared_ptr<Transformation> clone() const
    {
        return std::shared_ptr<Transformation>(doClone());
    }
    virtual std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int /*width*/, unsigned int /*height*/) const
    {
        return std::make_pair(0U, 0U);
    };
    virtual ~Transformation() {};

protected:
    inline static void padCropLabelsROI(
        std::vector<std::shared_ptr<ROI> >& labelsROI,
        int offsetX,
        int offsetY,
        unsigned int width,
        unsigned int height);

private:
    virtual Transformation* doClone() const = 0;
};
}

void N2D2::Transformation::apply(cv::Mat& frame, int id)
{
    cv::Mat emptyLabels;
    apply(frame, emptyLabels, id);
}

void N2D2::Transformation::apply(cv::Mat& frame, cv::Mat& labels, int id)
{
    std::vector<std::shared_ptr<ROI> > emptyLabelsROI;
    apply(frame, labels, emptyLabelsROI, id);
}

template <class T1>
N2D2::Tensor<T1> N2D2::Transformation::apply(const Tensor<T1>& frame,
                                               int id)
{
    cv::Mat mat = (cv::Mat)frame;
    apply(mat, id);

    Tensor<T1> res(mat);
    // For single channel cv::Mat, the tensor dims will be {x, y} and not
    // {x, y, 1}. If frame original dims where 3D ({x, y, 1}), make sure to
    // return a 3D tensor.
    res.reshape(frame.dims());
    return res;
}

template <class T1>
void N2D2::Transformation::apply(Tensor<T1>& frame, int id)
{
    const std::vector<size_t> dims = frame.dims();

    cv::Mat mat = (cv::Mat)frame;
    apply(mat, id);
    frame = Tensor<T1>(mat);
    frame.reshape(dims); // same as above
}

template <class T1, class T2>
void
N2D2::Transformation::apply(Tensor<T1>& frame, Tensor<T2>& labels, int id)
{
    const std::vector<size_t> dims = frame.dims();
    const std::vector<size_t> labelsDims = labels.dims();

    cv::Mat mat = (cv::Mat)frame;
    cv::Mat labelsMat = (cv::Mat)labels;
    apply(mat, labelsMat, id);
    frame = Tensor<T1>(mat);
    frame.reshape(dims); // same as above
    labels = Tensor<T2>(labelsMat);
    labels.reshape(labelsDims); // same as above
}

void
N2D2::Transformation::padCropLabelsROI(std::vector<std::shared_ptr<ROI> >&
                                       labelsROI,
                                       int offsetX,
                                       int offsetY,
                                       unsigned int width,
                                       unsigned int height)
{
    for (std::vector<std::shared_ptr<ROI> >::iterator it = labelsROI.begin();
         it != labelsROI.end();)
    {
        // Crop ROI
        (*it)->padCrop(offsetX, offsetY, width, height);

        // Check ROI overlaps with current slice
        const cv::Rect roiRect = (*it)->getBoundingRect();

        if (roiRect.tl().x > (int)width
            || roiRect.tl().y > (int)height
            || roiRect.br().x < 0 || roiRect.br().y < 0)
        {
            // No overlap with current slice, discard ROI
            it = labelsROI.erase(it);
        }
        else
            ++it;
    }
}

#endif // N2D2_TRANSFORMATION_H
