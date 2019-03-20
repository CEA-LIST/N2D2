/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#ifndef N2D2_CELL_CSPIKE_TOP_H
#define N2D2_CELL_CSPIKE_TOP_H

#include "Network.hpp"
#include "FloatT.hpp"

namespace N2D2 {

template<typename T>
class Tensor;

class Cell_CSpike_Top {
public:
    Cell_CSpike_Top()
    {
    }
    virtual bool tick(Time_T timestamp) = 0;
    virtual void reset(Time_T timestamp) = 0;
    virtual Tensor<int>& getOutputsActivity() = 0;
    virtual Tensor<int>& getOutputs() = 0;
    virtual bool isCuda() const = 0;
    virtual ~Cell_CSpike_Top() {};
};
}

#endif // N2D2_CELL_CSPIKE_TOP_H
