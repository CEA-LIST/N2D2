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

#ifndef N2D2_QUANTIZER_H
#define N2D2_QUANTIZER_H

#include "utils/Parameterizable.hpp"
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#include "Solver/Solver.hpp"

namespace N2D2 {

class Quantizer:  public Parameterizable {
public:

    Quantizer()
      : mBitPrecision(this, "bitPrecision", 8)
    {};

     //void addInput(BaseTensor& inputs, BaseTensor& diffOutputs);
    virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights) = 0;
    virtual void addBiases(BaseTensor& biases, BaseTensor& diffBiases) = 0;
    virtual void addActivations(BaseTensor& activations, BaseTensor& diffActivations) = 0;
    virtual void addActivations(BaseTensor& activations) = 0;
    //virtual void addCell(Cell* cell);

    virtual void initialize(){};
    virtual void update(){};
    virtual void propagate() = 0;
    virtual void back_propagate() = 0;
    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};
    virtual void importFreeParameters(const std::string& /*fileName*/, bool /*ignoreNoExists*/) {};

    virtual void setSolver(const std::shared_ptr<Solver>& solver)
    {
       throw std::runtime_error("Error: Tried to set solver in Quantizer" 
        " without learnable parameters!");
        mSolver = solver;
    };
    
    virtual std::shared_ptr<Solver> getSolver()
    {
        throw std::runtime_error("Error: Tried to get solver in Quantizer" 
        " without learnable parameters!");
        return mSolver;   
    };
    
    void setBitPrecision(unsigned int bit)
    {
        mBitPrecision = bit;
    };

    virtual const char* getType() const = 0;


    virtual BaseTensor& getQuantizedWeights(unsigned int k) = 0;

    virtual BaseTensor& getQuantizedBiases() = 0;

    virtual BaseTensor& getQuantizedActivations(unsigned int k) = 0;

    virtual BaseTensor& getDiffFullPrecisionWeights(unsigned int k) = 0;
   
    virtual BaseTensor& getDiffQuantizedWeights(unsigned int k) = 0;

    virtual BaseTensor& getDiffFullPrecisionBiases() = 0;

    virtual BaseTensor& getDiffQuantizedBiases() = 0;

    virtual BaseTensor& getDiffFullPrecisionActivations(unsigned int k) = 0;

    virtual BaseTensor& getDiffQuantizedActivations(unsigned int k) = 0;

    virtual bool isCuda() const
    {
        return false;
    }

    //virtual ~Quantizer() {};

protected:
    // NOTE: At the moment only one solver for all
    // trainable parameters in quantizer
    std::shared_ptr<Solver> mSolver;
    
    Parameter<unsigned int> mBitPrecision;
    
private:

  
};
}


#endif // N2D2_QUANTIZER_H

