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

#ifndef N2D2_CELL_CSPIKE_H
#define N2D2_CELL_CSPIKE_H

#include "Cell/Cell.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#include "controler/Interface.hpp"

namespace N2D2 {
class Cell_CSpike : public virtual Cell, public Cell_CSpike_Top {
public:
    Cell_CSpike(const std::string& name, unsigned int nbOutputs);
    virtual void addInput(StimuliProvider& sp,
                          unsigned int channel,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width,
                          unsigned int height,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width = 0,
                          unsigned int height = 0);
    virtual bool tick(Time_T timestamp);
    virtual void reset(Time_T timestamp);
    virtual Tensor<Float_T>& getOutputsActivity()
    {
        return mOutputsActivity;
    };
    virtual Tensor<char>& getOutputs()
    {
        return mOutputs;
    };
    bool isCuda() const
    {
        return false;
    }
    virtual ~Cell_CSpike() {};

protected:
    Interface<char> mInputs;
    Tensor<char> mOutputs;
    Tensor<Float_T> mOutputsActivity;
};
}

#endif // N2D2_CELL_CSPIKE_H
