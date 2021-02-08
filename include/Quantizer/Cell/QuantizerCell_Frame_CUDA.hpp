/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef N2D2_QUANTIZERCELL_FRAME_CUDA_H
#define N2D2_QUANTIZERCELL_FRAME_CUDA_H

#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#include "Quantizer/Cell/QuantizerCell.hpp"

namespace N2D2 {

template <class T> 
class QuantizerCell_Frame_CUDA: virtual public QuantizerCell {
public:
    virtual void initialize(){};
    virtual void update(unsigned int /*batchSize = 1*/){};
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

    virtual bool isCuda() const
    {
        return true;
    }
    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};
    virtual void importFreeParameters(const std::string& /*fileName*/, bool /*ignoreNoExists*/) {};

    virtual ~QuantizerCell_Frame_CUDA() {};

protected:

    /*
        Structures shared by all kind of quantizers :

        *mFullPrecisionWeights --->|    |---> *mQuantizedWeights
                                   |    |
                                   |    |
    *mDiffFullPrecisionWeights <---|    |<--- *mDiffQuantizedWeights
    */

    // Tensors for forward propagation
    CudaInterface<> mFullPrecisionWeights;
    CudaInterface<T> mQuantizedWeights;

    // TODO: Possible to avoid raw pointer? 
    // Problem: mBias in Cell sometimes Tensor and sometimes shared_ptr
    CudaBaseTensor* mFullPrecisionBiases;
    CudaTensor<T> mQuantizedBiases;

    /// Tensors for backpropagation
    CudaInterface<T> mDiffFullPrecisionWeights;
    CudaInterface<> mDiffQuantizedWeights;

    CudaTensor<T> mDiffFullPrecisionBiases;
    CudaBaseTensor* mDiffQuantizedBiases;
    

private:

  
};
}

#endif // N2D2_QUANTIZERCELL_FRAME_CUDA_H

