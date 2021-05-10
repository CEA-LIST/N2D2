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

#ifndef N2D2_QUANTIZERACTIVATION_H
#define N2D2_QUANTIZERACTIVATION_H

#include "utils/Parameterizable.hpp"
#include "containers/CudaTensor.hpp"
#include "Solver/Solver.hpp"

namespace N2D2 {

class QuantizerActivation:  public Parameterizable {
public:

    QuantizerActivation()
      : mRange(this, "Range", 255)
    {};
    virtual void update(unsigned int /*batchSize = 1*/) = 0;
    virtual void propagate(BaseTensor& baseInOut,
                            bool inference= false) = 0;
    virtual void back_propagate(const BaseTensor& input,
                                const BaseTensor& output,
                                const BaseTensor& diffInput,
                                BaseTensor& diffOutput) = 0;
    virtual void exportParameters(const std::string& /*fileName*/, const std::string& /*cellName*/) const {};
    virtual void importParameters(const std::string& /*dirName*/, const std::string& /*cellName*/, bool /*ignoreNotExists*/) {};

    virtual void setSolver(const std::shared_ptr<Solver>& solver)
    {
       throw std::runtime_error("Error: Tried to set solver in QuantizerActivation" 
        " without learnable parameters!");
        mSolver = solver;
    };
    
    virtual std::shared_ptr<Solver> getSolver()
    {
        throw std::runtime_error("Error: Tried to get solver in Quantizer" 
        " without learnable parameters!");
        return mSolver;   
    };

    void setRange(size_t integerRange)
    {
        mRange = integerRange;
    };

    size_t getRange()
    {
        return ((size_t) mRange);
    };

    virtual const char* getType() const = 0;

    virtual BaseTensor& getFullPrecisionActivations() = 0;

    virtual bool isCuda() const
    {
        return false;
    }

    virtual ~QuantizerActivation() {};

protected:
    // NOTE: At the moment only one solver for all
    // trainable parameters in quantizer
    std::shared_ptr<Solver> mSolver;
    Parameter<size_t> mRange;

private:
  
};
}


#endif // N2D2_QUANTIZER_H

