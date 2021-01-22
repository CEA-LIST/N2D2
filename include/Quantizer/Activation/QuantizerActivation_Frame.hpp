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
#ifndef N2D2_QUANTIZERACTIVATION_FRAME_H
#define N2D2_QUANTIZERACTIVATION_FRAME_H
#ifdef _OPENMP
#include <omp.h>
#endif
#include "controler/Interface.hpp"
#include "Quantizer/Activation/QuantizerActivation.hpp"

namespace N2D2 {

template <class T> 
class QuantizerActivation_Frame: virtual public QuantizerActivation {
public:
    virtual void initialize(){};
    virtual void update(){};
    virtual void propagate() = 0;
    virtual void back_propagate() = 0;

    virtual BaseTensor& getQuantizedActivations(unsigned int k)
    {
        return mQuantizedActivations[k];
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

        *mFullPrecisionActivations --->|    |---> *mQuantizedActivations
                                       |    |
                                       |    |
    *mDiffFullPrecisionActivations <---|    |<--- *mDiffQuantizedActivations

    */
    Interface<> mFullPrecisionActivations;
    Interface<T> mQuantizedActivations;

    Interface<T> mDiffFullPrecisionActivations;
    Interface<> mDiffQuantizedActivations;

private:

  
};
}

#endif // N2D2_QUANTIZERACTIVATION_FRAME_H

