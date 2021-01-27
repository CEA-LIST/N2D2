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
#ifndef N2D2_QUANTIZERCELL_FRAME_H
#define N2D2_QUANTIZERCELL_FRAME_H
#ifdef _OPENMP
#include <omp.h>
#endif
#include "controler/Interface.hpp"
#include "Quantizer/Cell/QuantizerCell.hpp"

namespace N2D2 {

template <class T> 
class QuantizerCell_Frame: virtual public QuantizerCell {
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
        return false;
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

    */
    Interface<> mFullPrecisionWeights;
    Interface<T> mQuantizedWeights;

    // TODO: Possible to avoid raw pointer? 
    // Problem: mBias in Cell sometimes Tensor and sometimes shared_ptr
    BaseTensor* mFullPrecisionBiases;
    Tensor<T> mQuantizedBiases;

    /// Tensors for backpropagation
    Interface<T> mDiffFullPrecisionWeights;
    Interface<> mDiffQuantizedWeights;

    Tensor<T> mDiffFullPrecisionBiases;
    BaseTensor* mDiffQuantizedBiases;

private:

  
};
}

#endif // N2D2_QUANTIZERCELL_FRAME_H

