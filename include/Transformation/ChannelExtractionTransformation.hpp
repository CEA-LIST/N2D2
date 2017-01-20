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
 * @file      ChannelExtractionTransformation.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define channel extraction mode.
 *
 * \details   These classes allow to extract the different channels from an
 *OpenCV matrix, applying some modifications.
*/

#ifndef N2D2_CHANNELEXTRACTIONTRANSFORMATION_H
#define N2D2_CHANNELEXTRACTIONTRANSFORMATION_H

#include "Transformation.hpp"

namespace N2D2 {
/**
 * @class   ChannelExtractionTransformation
 * @brief   Interface shared by all extraction classes.
 *
*/
class ChannelExtractionTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Channel {
        Red,
        Green,
        Blue,
        Hue,
        Saturation,
        Value,
        Gray,
        Y,
        Cb,
        Cr
    };

    ChannelExtractionTransformation(Channel channel);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    Channel getChannel() const
    {
        return mChannel;
    };
    std::shared_ptr<ChannelExtractionTransformation> clone() const
    {
        return std::shared_ptr<ChannelExtractionTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    virtual ~ChannelExtractionTransformation() {};

private:
    virtual ChannelExtractionTransformation* doClone() const
    {
        return new ChannelExtractionTransformation(*this);
    }

    const Channel mChannel;
};

/**
 * @class   RedChannelExtractionTransformation
 * @brief   Extract Red channel.
 *
*/
class RedChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    RedChannelExtractionTransformation() : ChannelExtractionTransformation(Red)
    {
    }
};

/**
 * @class   GreenChannelExtractionTransformation
 * @brief   Extract Green channel.
 *
*/
class GreenChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    GreenChannelExtractionTransformation()
        : ChannelExtractionTransformation(Green)
    {
    }
};

/**
 * @class   BlueChannelExtractionTransformation
 * @brief   Extract Blue channel.
 *
*/
class BlueChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    BlueChannelExtractionTransformation()
        : ChannelExtractionTransformation(Blue)
    {
    }
};

/**
 * @class   HueChannelExtractionTransformation
 * @brief   Extract Hue channel.
 *
*/
class HueChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    HueChannelExtractionTransformation() : ChannelExtractionTransformation(Hue)
    {
    }
};

/**
 * @class   SaturationChannelExtractionTransformation
 * @brief   Extract Saturation channel.
 *
*/
class SaturationChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    SaturationChannelExtractionTransformation()
        : ChannelExtractionTransformation(Saturation)
    {
    }
};

/**
 * @class   ValueChannelExtractionTransformation
 * @brief   Extract Value channel.
 *
*/
class ValueChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    ValueChannelExtractionTransformation()
        : ChannelExtractionTransformation(Value)
    {
    }
};

/**
 * @class   GrayChannelExtractionTransformation
 * @brief   Extract Gray channel.
 *
*/
class GrayChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    GrayChannelExtractionTransformation()
        : ChannelExtractionTransformation(Gray)
    {
    }
};

/**
 * @class   YChannelExtractionTransformation
 * @brief   Extract Y channel.
 *
*/
class YChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    YChannelExtractionTransformation() : ChannelExtractionTransformation(Y)
    {
    }
};

/**
 * @class   CbChannelExtractionTransformation
 * @brief   Extract Cb channel.
 *
*/
class CbChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    CbChannelExtractionTransformation() : ChannelExtractionTransformation(Cb)
    {
    }
};

/**
 * @class   CrChannelExtractionTransformation
 * @brief   Extract Cr channel.
 *
*/
class CrChannelExtractionTransformation
    : public ChannelExtractionTransformation {
public:
    CrChannelExtractionTransformation() : ChannelExtractionTransformation(Cr)
    {
    }
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::ChannelExtractionTransformation::Channel>::data[]
    = {"Red",   "Green", "Blue", "Hue", "Saturation",
       "Value", "Gray",  "Y",    "Cb",  "Cr"};
}

#endif // N2D2_CHANNELEXTRACTIONTRANSFORMATION_H
