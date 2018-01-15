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

#ifndef N2D2_COLORSPACETRANSFORMATION_H
#define N2D2_COLORSPACETRANSFORMATION_H

#include "Transformation.hpp"

namespace N2D2 {
class ColorSpaceTransformation : public Transformation {
public:
    using Transformation::apply;

    enum ColorSpace {
        BGR,
        RGB,
        HSV,
        HLS,
        YCrCb,
        CIELab,
        CIELuv
    };

    ColorSpaceTransformation(ColorSpace colorSpace);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    ColorSpace getColorSpace() const
    {
        return mColorSpace;
    };
    std::shared_ptr<ColorSpaceTransformation> clone() const
    {
        return std::shared_ptr<ColorSpaceTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    virtual ~ColorSpaceTransformation() {};

private:
    virtual ColorSpaceTransformation* doClone() const
    {
        return new ColorSpaceTransformation(*this);
    }

    const ColorSpace mColorSpace;
};

class RGBColorSpaceTransformation : public ColorSpaceTransformation {
public:
    RGBColorSpaceTransformation() : ColorSpaceTransformation(RGB)
    {
    }
};

class HSVColorSpaceTransformation : public ColorSpaceTransformation {
public:
    HSVColorSpaceTransformation() : ColorSpaceTransformation(HSV)
    {
    }
};

class HLSColorSpaceTransformation : public ColorSpaceTransformation {
public:
    HLSColorSpaceTransformation() : ColorSpaceTransformation(HLS)
    {
    }
};

class YCrCbColorSpaceTransformation : public ColorSpaceTransformation {
public:
    YCrCbColorSpaceTransformation() : ColorSpaceTransformation(YCrCb)
    {
    }
};

class CIELabColorSpaceTransformation : public ColorSpaceTransformation {
public:
    CIELabColorSpaceTransformation() : ColorSpaceTransformation(CIELab)
    {
    }
};

class CIELuvColorSpaceTransformation : public ColorSpaceTransformation {
public:
    CIELuvColorSpaceTransformation() : ColorSpaceTransformation(CIELuv)
    {
    }
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::ColorSpaceTransformation::ColorSpace>::data[]
    = {"BGR", "RGB", "HSV", "HLS", "YCrCb", "CIELab", "CIELuv"};
}

#endif // N2D2_COLORSPACETRANSFORMATION_H
