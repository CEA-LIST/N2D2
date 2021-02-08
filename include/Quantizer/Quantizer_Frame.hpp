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
#ifndef N2D2_QUANTIZER_FRAME_H
#define N2D2_QUANTIZER_FRAME_H
#ifdef _OPENMP
#include <omp.h>
#endif

//#include "Cell/Cell.hpp"
//#include "utils/Parameterizable.hpp"
#include "controler/Interface.hpp"
#include "Quantizer/Quantizer.hpp"

namespace N2D2 {

template <class T> 
class Quantizer_Frame: virtual public Quantizer {
public:
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

        *mFullPrecisionActivations --->|    |---> *mQuantizedActivations
                                       |    |
                                       |    |
    *mDiffFullPrecisionActivations <---|    |<--- *mDiffQuantizedActivations

    */
    Interface<> mFullPrecisionWeights;
    Interface<T> mQuantizedWeights;

    // TODO: Possible to avoid raw pointer? 
    // Problem: mBias in Cell sometimes Tensor and sometimes shared_ptr
    BaseTensor* mFullPrecisionBiases;
    Tensor<T> mQuantizedBiases;

    Interface<> mFullPrecisionActivations;
    Interface<T> mQuantizedActivations;

    /// Tensors for backpropagation
    Interface<T> mDiffFullPrecisionWeights;
    Interface<> mDiffQuantizedWeights;

    Tensor<T> mDiffFullPrecisionBiases;
    BaseTensor* mDiffQuantizedBiases;

    Interface<T> mDiffFullPrecisionActivations;
    Interface<> mDiffQuantizedActivations;

private:

  
};
}

#endif // N2D2_QUANTIZER_FRAME_H

