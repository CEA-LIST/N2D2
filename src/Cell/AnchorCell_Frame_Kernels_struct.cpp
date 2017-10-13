/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include <cmath>

#include "Cell/AnchorCell_Frame_Kernels_struct.hpp"

N2D2::AnchorCell_Frame_Kernels::Anchor::Anchor(float width,
                                               float height,
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

N2D2::AnchorCell_Frame_Kernels::Anchor::Anchor(unsigned int area,
                                               double ratio,
                                               double scale,
                                               Anchoring anchoring)
{
    const double areaRatio = area / ratio;
    const double wr = round(sqrt(areaRatio));
    const double hr = round(wr * ratio);
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
        const double size = sqrt(area);
        double center = (size - 1.0) / 2.0;

        if (anchoring == OriginalFlipped)
            center = -center;

        x0 = center - (ws - 1) / 2.0;
        y0 = center - (hs - 1) / 2.0;
    }

    x1 = x0 + ws - 1;
    y1 = y0 + hs - 1;
}

float N2D2::AnchorCell_Frame_Kernels::Anchor::round(float x)
{
    return (x < 0.0) ? ceil(x - 0.5) : floor(x + 0.5);
}
