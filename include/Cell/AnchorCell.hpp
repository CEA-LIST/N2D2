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

#include "AnchorCell_Frame_Kernels_struct.hpp"

namespace N2D2 {
class AnchorCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<AnchorCell>(
        const std::string&,
        StimuliProvider&,
        const std::vector<AnchorCell_Frame_Kernels::Anchor>&,
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
               const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
               unsigned int scoresCls = 1);
    const char* getType() const
    {
        return Type;
    };
    virtual const std::vector<AnchorCell_Frame_Kernels::BBox_T>&
        getGT(unsigned int batchPos) const = 0;
    virtual std::shared_ptr<ROI> getAnchorROI(
        const Tensor4d<int>::Index& index) const = 0;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorBBox(
        const Tensor4d<int>::Index& index) const = 0;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorGT(
        const Tensor4d<int>::Index& index) const = 0;
    virtual Float_T getAnchorIoU(const Tensor4d<int>::Index& index) const = 0;
    virtual int getAnchorArgMaxIoU(const Tensor4d<int>::Index& index) const = 0;
    void getStats(Stats& stats) const;
    virtual int getNbAnchors() const = 0;
    virtual std::vector<Float_T> getAnchor(const unsigned int idx) const = 0;
    bool isFlip()
    {
        bool flipStatus = mFlip;
        return flipStatus;
    };
    unsigned int getScoreCls() { return mScoresCls; };
    virtual ~AnchorCell() {};

protected:
    virtual void setOutputsSize();

    Parameter<double> mPositiveIoU;
    Parameter<double> mNegativeIoU;
    Parameter<double> mLossLambda;
    Parameter<unsigned int> mLossPositiveSample;
    Parameter<unsigned int> mLossNegativeSample;
    Parameter<bool> mFlip;

    StimuliProvider& mStimuliProvider;
    unsigned int mScoresCls;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::AnchorCell_Frame_Kernels::Anchor::Anchoring>
::data[]
    = {"TopLeft", "Centered", "Original", "OriginalFlipped"};
}

#endif // N2D2_ANCHORCELL_H
