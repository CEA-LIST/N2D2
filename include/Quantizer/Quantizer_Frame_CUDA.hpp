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

#ifndef N2D2_QUANTIZER_FRAME_CUDA_H
#define N2D2_QUANTIZER_FRAME_CUDA_H

//#include "Cell/Cell.hpp"
//#include "utils/Parameterizable.hpp"
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#include "Quantizer/Quantizer.hpp"

namespace N2D2 {

template <class T> 
class Quantizer_Frame_CUDA: virtual public Quantizer {
public:

    //void addInput(BaseTensor& inputs, BaseTensor& diffOutputs);
    //virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights) = 0;
    //virtual void addActivations(BaseTensor& activations, BaseTensor& diffActivations) = 0;
    //virtual void addCell(Cell* cell);

    virtual void initialize(){};
    virtual void update(){};
    virtual void propagate() = 0;
    virtual void back_propagate() = 0;

    virtual BaseTensor& getQuantizedWeights(unsigned int k)
    {
        return mQuantizedWeights[k];
    }

    virtual BaseTensor& getQuantizedBiases()
    {
        return mQuantizedBiases;
    }

    virtual BaseTensor& getQuantizedActivations(unsigned int k)
    {
        return mQuantizedActivations[k];
    }

    virtual BaseTensor& getDiffFullPrecisionWeights(unsigned int k)
    {
        //mDiffFullPrecisionWeights[k].synchronizeDToH();
        return mDiffFullPrecisionWeights[k];
    }

    virtual BaseTensor& getDiffQuantizedWeights(unsigned int k)
    {
        return mDiffQuantizedWeights[k];
    }

    virtual BaseTensor& getDiffFullPrecisionBiases()
    {
        return mDiffFullPrecisionBiases;
    }

    virtual BaseTensor& getDiffQuantizedBiases()
    {
        return *mDiffQuantizedBiases;
    }

    virtual BaseTensor& getDiffFullPrecisionActivations(unsigned int k)
    {
        return mDiffFullPrecisionActivations[k];
    }

    virtual BaseTensor& getDiffQuantizedActivations(unsigned int k)
    {
        return mDiffQuantizedActivations[k];
    }

    virtual bool isCuda() const
    {
        return true;
    }
    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};
    virtual void importFreeParameters(const std::string& /*fileName*/, bool /*ignoreNoExists*/) {};

    //virtual ~Quantizer() {};

protected:

    /*
        Structures shared by all kind of quantizers :

        *mFullPrecisionWeights --->|    |---> *mQuantizedWeights
                                   |    |
                                   |    |
    *mDiffFullPrecisionWeights <---|    |<--- *mDiffQuantizedWeights

        *mFullPrecisionActivations --->|    |---> *mQuantizedActivations
                                       |    |
                                       |    |
    *mDiffFullPrecisionActivations <---|    |<--- *mDiffQuantizedActivations

    */

    // Tensors for forward propagation
    CudaInterface<> mFullPrecisionWeights;
    CudaInterface<T> mQuantizedWeights;

    // TODO: Possible to avoid raw pointer? 
    // Problem: mBias in Cell sometimes Tensor and sometimes shared_ptr
    CudaBaseTensor* mFullPrecisionBiases;
    CudaTensor<T> mQuantizedBiases;

    CudaInterface<> mFullPrecisionActivations;
    CudaInterface<T> mQuantizedActivations;

    /// Tensors for backpropagation
    CudaInterface<T> mDiffFullPrecisionWeights;
    CudaInterface<> mDiffQuantizedWeights;

    CudaTensor<T> mDiffFullPrecisionBiases;
    CudaBaseTensor* mDiffQuantizedBiases;

    CudaInterface<T> mDiffFullPrecisionActivations;
    CudaInterface<> mDiffQuantizedActivations;
    

private:

  
};
}

#endif // N2D2_QUANTIZER_FRAME_CUDA_H

