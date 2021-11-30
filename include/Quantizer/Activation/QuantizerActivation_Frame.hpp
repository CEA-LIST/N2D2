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
#include "Quantizer/Activation/QuantizerActivation.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {

template <class T> 
class QuantizerActivation_Frame: virtual public QuantizerActivation {
public:
    virtual void update(unsigned int /*batchSize = 1*/) = 0;
    virtual void propagate(BaseTensor& baseInOut,
                            bool inference= false) = 0;
    virtual void back_propagate(const BaseTensor& input,
                                const BaseTensor& output,
                                const BaseTensor& diffInput,
                                BaseTensor& diffOutput) = 0;
    virtual BaseTensor& getFullPrecisionActivations()
    {
        return mFullPrecisionActivations;
    }

    virtual bool isCuda() const
    {
        return false;
    }
    virtual void exportParameters(const std::string& /*fileName*/, const std::string& /*cellName*/) const {};
    virtual void importParameters(const std::string& /*dirName*/, const std::string& /*cellName*/, bool /*ignoreNotExists*/) {};

    virtual ~QuantizerActivation_Frame() {};

protected:
    Tensor<T> mFullPrecisionActivations;
private:

};
}

#endif // N2D2_QUANTIZERACTIVATION_FRAME_H

