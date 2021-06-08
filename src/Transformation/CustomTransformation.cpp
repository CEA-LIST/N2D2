/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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

#include "Transformation/CustomTransformation.hpp"

const char* N2D2::CustomTransformation::Type = "CustomTransformation";

N2D2::CustomTransformation::CustomTransformation()
{
    //ctor
}

N2D2::CustomTransformation::CustomTransformation(const CustomTransformation& trans)
{
    // copy-ctor
}

void N2D2::CustomTransformation::apply(cv::Mat& frame,
                                cv::Mat& labels,
                                std::vector<std::shared_ptr<ROI> >& labelsROI,
                                int id){
    Tensor<int> tensor_label = N2D2::Tensor<int>(labels);
    switch (frame.depth()) {
        case CV_8S:{
            Tensor<char> tensor_frame = N2D2::Tensor<char>(frame);
            this->apply_char(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_8U:{
            Tensor<unsigned char> tensor_frame = N2D2::Tensor<unsigned char>(frame);
            this->apply_unsigned_char(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_16U:{
            Tensor<unsigned short> tensor_frame = N2D2::Tensor<unsigned short>(frame);
            this->apply_unsigned_short(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_16S:{
            Tensor<short> tensor_frame = N2D2::Tensor<short>(frame);
            this->apply_short(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_32S:{
            Tensor<int> tensor_frame = N2D2::Tensor<int>(frame);
            this->apply_int(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_32F:{
            Tensor<float> tensor_frame = N2D2::Tensor<float>(frame);
            this->apply_float(tensor_frame, tensor_label, labelsROI, id);
        }break;
        case CV_64F:{
            Tensor<double> tensor_frame = N2D2::Tensor<double>(frame);
            this->apply_double(tensor_frame, tensor_label, labelsROI, id);
        }break;
        default:
            throw std::runtime_error("Cannot convert cv::Mat to Tensor: incompatible types.");
        // TODO : see if we need to update the cv::mat object or if it chare the same memory space as the Tensor.
    };

}
