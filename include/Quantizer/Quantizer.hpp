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

#ifndef N2D2_QUANTIZER_H
#define N2D2_QUANTIZER_H

#include "utils/Parameterizable.hpp"
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#include "Solver/Solver.hpp"

namespace N2D2 {

class Quantizer:  public Parameterizable {
public:

     //void addInput(BaseTensor& inputs, BaseTensor& diffOutputs);
    virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights) = 0;
    virtual void addBiases(BaseTensor& biases, BaseTensor& diffBiases) = 0;
    virtual void addActivations(BaseTensor& activations, BaseTensor& diffActivations) = 0;
    //virtual void addCell(Cell* cell);

    virtual void initialize(){};
    virtual void update(){};
    virtual void propagate() = 0;
    virtual void back_propagate() = 0;

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
    
private:

  
};
}


#endif // N2D2_QUANTIZER_H

