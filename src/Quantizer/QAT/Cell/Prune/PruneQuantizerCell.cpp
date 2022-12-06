/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell.hpp"

const char* N2D2::PruneQuantizerCell::Type = "Prune";

N2D2::PruneQuantizerCell::PruneQuantizerCell()                      
    : mPruningMode(this, "PruningMode", N2D2::PruneQuantizerCell::PruningMode::Identity),
      mThreshold(this, "Threshold", 0.2),
      mDelta(this, "Delta", 0.001)
{
    // ctor
}
