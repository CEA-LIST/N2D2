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

#ifndef N2D2_QUANTIZER_CUDA_H
#define N2D2_QUANTIZER_CUDA_H

#include "Quantizer/Quantizer.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include "Cell/Cell.hpp"
//#include "utils/Parameterizable.hpp"
#include "controler/CudaInterface.hpp"


//TODO: Everything in float or templated? In principle during QAT everything can remain in float
//TODO: Make abstract class like Cell?

namespace N2D2 {

class Quantizer_CUDA:  public Quantizer {
public:

    Quantizer_CUDA();

    //void addInput(BaseTensor& inputs, BaseTensor& diffOutputs);
    virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights);
    virtual void addActivations(BaseTensor& activations, BaseTensor& diffActivations);
    virtual void addCell(Cell* cell);

    virtual void initialize();
    virtual void update();
    virtual void propagate();
    virtual void back_propagate();

    CudaTensor<float>& getQuantizedWeights(unsigned int k)
    {
        return mQuantizedWeights[k];
    }

    /*CudaTensor<float>& getQuantizedBiases()
    {
        return mQuantizedBiases;
    }*/

    CudaTensor<float>& getQuantizedActivations(unsigned int k)
    {
        return mQuantizedActivations[k];
    }

    CudaTensor<float>& getDiffFullPrecisionWeights(unsigned int k)
    {
        return mDiffFullPrecisionWeights[k];
    }

    CudaTensor<float>& getDiffQuantizedWeights(unsigned int k)
    {
        return mDiffFullPrecisionWeights[k];
    }

    /*CudaTensor<float>& getDiffFullPrecisionBiases()
    {
        return mDiffFullPrecisionBiases;
    }*/

    CudaTensor<float>& getDiffFullPrecisionActivations(unsigned int k)
    {
        return mDiffFullPrecisionActivations[k];
    }

    CudaTensor<float>& getDiffQuantizedActivations(unsigned int k)
    {
        return mDiffFullPrecisionActivations[k];
    }

    std::shared_ptr<Quantizer_CUDA> clone() const
    {
        return std::shared_ptr<Quantizer_CUDA>(doClone());
    }
    

    bool isCuda() const
    {
        return true;
    };

    virtual ~Quantizer_CUDA() /*{}*/;

protected:

    //TODO: Quantizer configuration parameters set with Cell type or separately? 
    //Parameter<float> mStepSize;
    // etc.

    /*
        Structures shared by all kind of quantizers :

       *mFullPrecisionWeights --->|    |---> mQuantizedWeights
                                  |    |
                                  |    |
    mDiffFullPrecisionWeights <---|    |<--- *mDiffQuantizedWeights

       *mFullPrecisionActivations --->|    |---> mQuantizedActivations
                                      |    |
                                      |    |
    mDiffFullPrecisionActivations <---|    |<--- *mDiffQuantizedActivations

    */

    // Tensors for forward propagation
    CudaInterface<> mFullPrecisionWeights;
    CudaInterface<float> mQuantizedWeights;

    //CudaTensor<float>* mFullPrecisionBiases;
    //CudaTensor<float> mQuantizedBiases;

    CudaInterface<> mFullPrecisionActivations;
    CudaInterface<float> mQuantizedActivations;

    /// Tensors for backpropagation
    CudaInterface<float> mDiffFullPrecisionWeights;
    CudaInterface<> mDiffQuantizedWeights;

    //CudaTensor<float> mDiffFullPrecisionBiases;
    //CudaTensor<float>* mDiffQuantizedBiases;

    CudaInterface<float> mDiffFullPrecisionActivations;
    CudaInterface<> mDiffQuantizedActivations;

private:
    virtual Quantizer_CUDA* doClone() const
    {
        return new Quantizer_CUDA(*this);
    }

  
};
}


#endif // N2D2_QUANTIZER_CUDA_H

