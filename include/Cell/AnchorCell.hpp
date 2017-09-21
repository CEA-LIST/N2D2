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

#ifndef N2D2_ANCHORCELL_H
#define N2D2_ANCHORCELL_H

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
    "/include:?mRegistrar@AnchorCell_Frame_CUDA@N2D2@@0U?$Registrar@VAnchorCell@N2D2@@@2@A")
#endif
*/
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@AnchorCell_Frame@N2D2@@0U?$Registrar@VAnchorCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class AnchorCell : public virtual Cell {
public:
    struct Anchor {
        enum Anchoring {
            TopLeft,
            Centered,
            Original
        };

        Anchor(Float_T x0_,
               Float_T y0_,
               Float_T width_,
               Float_T height_)
            : x0(x0_),
              y0(y0_),
              x1(width_ + x0_ - 1),
              y1(height_ + y0_ - 1)
        {
        }
        Anchor(Float_T width, Float_T height, Anchoring anchoring = TopLeft);
        Anchor(unsigned int area,
               double ratio,
               double scale = 1.0,
               Anchoring anchoring = TopLeft);
        inline Float_T getWidth() const
        {
            return (x1 - x0 + 1.0);
        }
        inline Float_T getHeight() const
        {
            return (y1 - y0 + 1.0);
        }

        Float_T x0;
        Float_T y0;
        Float_T x1;
        Float_T y1;
    };
    typedef std::tuple<Float_T, Float_T, Float_T, Float_T> BBox_T;

    typedef std::function
        <std::shared_ptr<AnchorCell>(const std::string&,
                                     StimuliProvider&,
                                     const std::vector<Anchor>&,
                                     unsigned int)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    AnchorCell(const std::string& name,
               StimuliProvider& sp,
               const std::vector<Anchor>& anchors,
               unsigned int scoresCls = 1);
    const char* getType() const
    {
        return Type;
    };
    virtual const std::vector<BBox_T>& getGT(unsigned int batchPos) const = 0;
    virtual std::shared_ptr<ROI> getAnchorROI(const Tensor4d<int>::Index&
                                                index) const = 0;
    virtual BBox_T getAnchorBBox(const Tensor4d<int>::Index& index) const = 0;
    virtual BBox_T getAnchorGT(const Tensor4d<int>::Index& index) const = 0;
    virtual Float_T getAnchorIoU(const Tensor4d<int>::Index& index) const = 0;
    virtual int getAnchorArgMaxIoU(const Tensor4d<int>::Index& index) const = 0;
    void getStats(Stats& stats) const;
    virtual ~AnchorCell() {};

protected:
    virtual void setOutputsSize();

    Parameter<double> mPositiveIoU;
    Parameter<double> mNegativeIoU;
    Parameter<double> mLossLambda;
    Parameter<unsigned int> mLossPositiveSample;
    Parameter<unsigned int> mLossNegativeSample;

    StimuliProvider& mStimuliProvider;
    std::vector<Anchor> mAnchors;
    unsigned int mScoresCls;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::AnchorCell::Anchor::Anchoring>::data[]
    = {"TopLeft", "Centered", "Original"};
}

#endif // N2D2_ANCHORCELL_H
