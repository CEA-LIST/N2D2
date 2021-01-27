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

#ifndef N2D2_QUANTIZERCELL_H
#define N2D2_QUANTIZERCELL_H

#include "utils/Parameterizable.hpp"
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#include "Solver/Solver.hpp"

namespace N2D2 {

class QuantizerCell:  public Parameterizable {
public:

    QuantizerCell()
      : mRange(this, "Range", 255)
    {};

    virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights) = 0;
    virtual void addBiases(BaseTensor& biases, BaseTensor& diffBiases) = 0;

    virtual void initialize(){};
    virtual void update(unsigned int /*batchSize = 1*/){};

    virtual void propagate() = 0;
    virtual void back_propagate() = 0;
    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};
    virtual void importFreeParameters(const std::string& /*fileName*/, bool /*ignoreNoExists*/) {};

    virtual void setSolver(const std::shared_ptr<Solver>& solver)
    {
       throw std::runtime_error("Error: Tried to set solver in QuantizerCell" 
        " without learnable parameters!");
        mSolver = solver;
    };
    
    virtual std::shared_ptr<Solver> getSolver()
    {
        throw std::runtime_error("Error: Tried to get solver in QuantizerCell" 
        " without learnable parameters!");
        return mSolver;   
    };
    
    void setRange(size_t integerRange)
    {
        mRange = integerRange;
    };

    virtual const char* getType() const = 0;

    virtual BaseTensor& getQuantizedWeights(unsigned int k) = 0;

    virtual BaseTensor& getQuantizedBiases() = 0;

    virtual BaseTensor& getDiffFullPrecisionWeights(unsigned int k) = 0;
   
    virtual BaseTensor& getDiffQuantizedWeights(unsigned int k) = 0;

    virtual BaseTensor& getDiffFullPrecisionBiases() = 0;

    virtual BaseTensor& getDiffQuantizedBiases() = 0;

    virtual bool isCuda() const
    {
        return false;
    }

    //virtual ~Quantizer() {};

protected:
    // NOTE: At the moment only one solver for all
    // trainable parameters in quantizer
    std::shared_ptr<Solver> mSolver;
    
    Parameter<size_t> mRange;
    
private:

};
}


#endif // N2D2_QUANTIZERCELL_H

