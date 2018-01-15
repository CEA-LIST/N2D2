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

#ifndef N2D2_ROIPOOLINGCELL_H
#define N2D2_ROIPOOLINGCELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
/*
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ROIPoolingCell_Frame_CUDA@N2D2@@0U?$Registrar@VROIPoolingCell@N2D2@@@2@A")
#endif
*/
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ROIPoolingCell_Frame@N2D2@@0U?$Registrar@VROIPoolingCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class ROIPoolingCell : public virtual Cell {
public:
    enum ROIPooling {
        Max,
        Average,
        Bilinear,   // Compatible with OpenCV resize() [INTER_LINEAR] function
        BilinearTF  // Compatible with TensorFlow crop_and_resize() function
    };

    typedef std::function
        <std::shared_ptr<ROIPoolingCell>(const std::string&,
                                         StimuliProvider&,
                                         unsigned int,
                                         unsigned int,
                                         unsigned int,
                                         ROIPooling)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ROIPoolingCell(const std::string& name,
                   StimuliProvider& sp,
                   unsigned int outputsWidth,
                   unsigned int outputsHeight,
                   unsigned int nbOutputs,
                   ROIPooling pooling);
    const char* getType() const
    {
        return Type;
    };
    ROIPooling getPooling() const
    {
        return mPooling;
    };
    bool isFlip()
    {
        return (bool) mFlip;
    };
    bool isIgnorePadding()
    {
        return (bool) mIgnorePad;
    };
    void getStats(Stats& stats) const;
    virtual ~ROIPoolingCell() {};
    unsigned int getParentProposals()
    {
        return mParentProposals;
    };

protected:
    virtual void setInputsSize(unsigned int width, unsigned int height);
    virtual void setOutputsSize();

    Parameter<bool> mFlip;
    Parameter<bool> mIgnorePad;

    StimuliProvider& mStimuliProvider;
    // Pooling type
    const ROIPooling mPooling;
    unsigned int mParentProposals;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ROIPoolingCell::ROIPooling>::data[]
    = {"Max", "Average", "Bilinear", "BilinearTF"};
}

#endif // N2D2_ROIPOOLINGCELL_H
