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

#include "Transformation/DFTTransformation.hpp"

N2D2::DFTTransformation::DFTTransformation(bool twoDimensional)
    : mTwoDimensional(twoDimensional)
{
    // ctor
}

void N2D2::DFTTransformation::apply(cv::Mat& frame,
                                    cv::Mat& /*labels*/,
                                    std::vector
                                    <std::shared_ptr<ROI> >& /*labelsROI*/,
                                    int /*id*/)
{
    if (frame.channels() != 1)
        throw std::runtime_error(
            "DFTTransformation: require single channel input");
    /*
        if (mTwoDimensional) {*/
    // expand input image to optimal size
    cv::Mat padded;
    const int m = cv::getOptimalDFTSize(frame.rows);
    const int n
        = cv::getOptimalDFTSize(frame.cols); // on the border add zero values
    cv::copyMakeBorder(frame,
                       padded,
                       0,
                       m - frame.rows,
                       0,
                       n - frame.cols,
                       cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    cv::Mat planes[]
        = {cv::Mat_<double>(padded), cv::Mat::zeros(padded.size(), CV_64F)};
    cv::Mat complexMat;
    cv::merge(
        planes, 2, complexMat); // Add to the expanded another plane with zeros

    cv::dft(complexMat,
            complexMat,
            (!mTwoDimensional) ? cv::DFT_ROWS : 0); // this way the result may
    // fit in the source matrix
    frame
        = complexMat; /*
}
else {
   cv::Mat real;
   frame.convertTo(real, CV_64F);

   cv::Mat imag = cv::Mat::zeros(real.size(), CV_64F);

   // Compute the FFT for each row
   for (int i = 0; i < real.rows; ++i) {
       double* rowPtr = real.ptr<double>(i);
       // Convert the row to vector of complex
       std::vector<std::complex<double> > x =
DSP::toComplex(std::vector<double>(rowPtr, rowPtr + real.cols));

       // FFT
       DSP::fft(x);

       // Real part
       std::transform(x.begin(), x.end(), rowPtr,
           std::bind(static_cast<double(std::complex<double>::*)()const>(&std::complex<double>::real),
std::placeholders::_1));

       // Imag part
       rowPtr = imag.ptr<double>(i);
       std::transform(x.begin(), x.end(), rowPtr,
           std::bind(static_cast<double(std::complex<double>::*)()const>(&std::complex<double>::imag),
std::placeholders::_1));
   }

   std::vector<cv::Mat> comp;
   comp.push_back(real);
   comp.push_back(imag);

   cv::merge(comp, frame);
}*/
}
