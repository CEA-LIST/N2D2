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

#include "Cell/AnchorCell.hpp"

const char* N2D2::AnchorCell::Type = "Anchor";

N2D2::AnchorCell::Anchor::Anchor(Float_T width,
                                 Float_T height,
                                 Anchoring anchoring)
{
    if (anchoring == TopLeft) {
        x0 = 0;
        y0 = 0;
    }
    else {
        x0 = - (width - 1) / 2.0;
        y0 = - (height - 1) / 2.0;
    }

    x1 = x0 + width - 1;
    y1 = y0 + height - 1;
}

N2D2::AnchorCell::Anchor::Anchor(unsigned int area,
                                 double ratio,
                                 double scale,
                                 Anchoring anchoring)
{
    const double areaRatio = area / ratio;
    const double wr = Utils::round(std::sqrt(areaRatio));
    const double hr = Utils::round(wr * ratio);
    const double ws = wr * scale;
    const double hs = hr * scale;

    if (anchoring == TopLeft) {
        x0 = 0;
        y0 = 0;
    }
    else if (anchoring == Centered) {
        x0 = - (ws - 1) / 2.0;
        y0 = - (hs - 1) / 2.0;
    }
    else {
        const double size = std::sqrt(area);
        const double center = (size - 1.0) / 2.0;

        x0 = center - (ws - 1) / 2.0;
        y0 = center - (hs - 1) / 2.0;
    }

    x1 = x0 + ws - 1;
    y1 = y0 + hs - 1;
}

N2D2::AnchorCell::AnchorCell(const std::string& name,
                             StimuliProvider& sp,
                             const std::vector<Anchor>& anchors,
                             unsigned int scoresCls)
    : Cell(name, 6*anchors.size()),
      mPositiveIoU(this, "PositiveIoU", 0.7),
      mNegativeIoU(this, "NegativeIoU", 0.3),
      mLossLambda(this, "LossLambda", 10.0),
      mLossPositiveSample(this, "LossPositiveSample", 128U),
      mLossNegativeSample(this, "LossNegativeSample", 128U),
      mStimuliProvider(sp),
      mAnchors(anchors),
      mScoresCls(scoresCls)
{
    // ctor
}

void N2D2::AnchorCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::AnchorCell::setOutputsSize()
{
    mOutputsWidth = mChannelsWidth;
    mOutputsHeight = mChannelsHeight;
}
