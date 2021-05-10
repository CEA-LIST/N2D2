/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_OBJECTDETCELL_H
#define N2D2_OBJECTDETCELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"
#include "AnchorCell_Frame_Kernels_struct.hpp"
#include "Database/Database.hpp"

namespace N2D2 {

class DeepNet;

class ObjectDetCell : public virtual Cell {
public:
    struct BBox_T {
        float x;
        float y;
        float w;
        float h;
        float s;

        BBox_T() {}
        BBox_T(float x_, float y_, float w_, float h_, float s_):
            x(x_), y(y_), w(w_), h(h_), s(s_) {}
    };

    typedef std::function
        <std::shared_ptr<ObjectDetCell>(const DeepNet&, const std::string&,
                                        StimuliProvider&,
                                        const unsigned int,
                                        unsigned int,
                                        const AnchorCell_Frame_Kernels::Format ,
                                        const AnchorCell_Frame_Kernels::PixelFormat ,
                                        unsigned int,
                                        unsigned int,
                                        Float_T,
                                        std::vector<Float_T>,
                                        std::vector<unsigned int>,
                                        std::vector<unsigned int>,
                                        const std::vector<AnchorCell_Frame_Kernels::Anchor>&)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ObjectDetCell(const DeepNet& deepNet, const std::string& name,
                 StimuliProvider& sp,
                 const unsigned int nbOutputs,
                 unsigned int nbAnchors,
                const AnchorCell_Frame_Kernels::Format inputFormat,
                const AnchorCell_Frame_Kernels::PixelFormat pixelFormat,
                 unsigned int nbProposals,
                 unsigned int nbClass,
                 Float_T nmsThreshold,
                 std::vector<Float_T> scoreThreshold,
                 std::vector<unsigned int> numParts
                                            = std::vector<unsigned int>(),
                 std::vector<unsigned int> numTemplates
                                            = std::vector<unsigned int>(),
                 const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors
                                            = std::vector<AnchorCell_Frame_Kernels::Anchor>());

    const char* getType() const
    {
        return Type;
    };

    unsigned int getNbProposals() const
    {
        return mNbProposals;
    };

    unsigned int getNbAnchors() const
    {
        return mNbAnchors;
    };

    Float_T getNMSParam() const { return (double) mNMS_IoU_Threshold; };
    std::vector<Float_T> getScoreThreshold() const { return mScoreThreshold; };

    unsigned int getNbClass() const { return mNbClass; };
    bool getWithParts() const { return (mMaxParts > 0 ? true: false); };
    bool getWithTemplates() const { return (mMaxTemplates > 0 ? true: false); };
    unsigned int getMaxParts() const { return mMaxParts; };
    unsigned int getMaxTemplates() const { return mMaxTemplates; };
    unsigned int getFeatureMapWidth() { return mFeatureMapWidth; };
    unsigned int getFeatureMapHeight() { return mFeatureMapHeight; };
    std::vector<unsigned int> getPartsPerClass() const { return mNumParts; };
    std::vector<unsigned int> getTemplatesPerClass() const { return mNumTemplates; };
    bool getIsCoordinateAnchors() { return (mInputFormat == AnchorCell_Frame_Kernels::Format::CA); };
    bool getIsPixelFormatXY() { return (mPixelFormat == AnchorCell_Frame_Kernels::PixelFormat::XY); };

    void getStats(Stats& stats) const;
    virtual std::vector<Float_T> getAnchor(const unsigned int idx) const = 0;
    virtual ~ObjectDetCell() {};

protected:
    virtual void setOutputsDims();

    StimuliProvider& mStimuliProvider;
    Parameter<Float_T> mForegroundRate;
    Parameter<Float_T> mForegroundMinIoU;
    Parameter<Float_T> mBackgroundMaxIoU;
    Parameter<Float_T> mBackgroundMinIoU;
    Parameter<unsigned int> mFeatureMapWidth;
    Parameter<unsigned int> mFeatureMapHeight;

    unsigned int mNbAnchors;
    AnchorCell_Frame_Kernels::Format mInputFormat;
    AnchorCell_Frame_Kernels::PixelFormat mPixelFormat;
    unsigned int mNbProposals;
    unsigned int mNbClass;
    unsigned int mMaxParts;
    unsigned int mMaxTemplates;

    Float_T mNMS_IoU_Threshold;
    std::vector<Float_T> mScoreThreshold;
    //std::vector<Tensor<int>::Index> mAnchors;
    std::vector<unsigned int> mNumParts;
    std::vector<unsigned int> mNumTemplates;

};
}

namespace {
template <>
const char* const EnumStrings<N2D2::AnchorCell_Frame_Kernels::Format>
::data[]
    = {"CA", "AC"};
}

namespace {
template <>
const char* const EnumStrings<N2D2::AnchorCell_Frame_Kernels::PixelFormat>
::data[]
    = {"XY", "YX"};
}

#endif // N2D2_OBJECTDETCELL_H
