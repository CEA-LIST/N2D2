/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)

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

#ifndef N2D2_QUANTIZER_H
#define N2D2_QUANTIZER_H

//#include "containers/CudaTensor.hpp"
#include "Cell/Cell.hpp"
#include "utils/Parameterizable.hpp"


//TODO: Everything in float or templated? In principle during QAT everything can remain in float

namespace N2D2 {

class Quantizer:  public Parameterizable {
public:

    //void addInput(BaseTensor& inputs, BaseTensor& diffOutputs);
    //virtual void addWeights(CudaTensor<float>& weights);
    //virtual void addActivations(CudaTensor<float>& activations);
    //virtual void addCell(Cell* cell);

    //virtual void initialize();
    //virtual void update();
    virtual void propagate() = 0;
    virtual void back_propagate() = 0;

    // TODO: Remove if not needed anymore
    std::shared_ptr<Quantizer> clone() const
    {
        return std::shared_ptr<Quantizer>(doClone());
    }

    /*CudaTensor<float>& getQuantizedWeights()
    {
        return mQuantizedWeights;
    }*/

    /*CudaTensor<float>& getQuantizedActivations()
    {
        return mQuantizedActivations;
    }*/

    /*CudaTensor<float>& getDiffFullPrecisionWeights()
    {
        return mDiffFullPrecisionWeights;
    }*/

    /*CudaTensor<float>& getDiffFullPrecisionActivations()
    {
        return mDiffFullPrecisionActivations;
    }*/
    

    virtual ~Quantizer() {};

protected:

private:
    virtual Quantizer* doClone() const = 0;

  
};
}


#endif // N2D2_QUANTIZER_H

