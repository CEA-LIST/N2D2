/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#ifndef N2D2_LSQQUANTIZERCELL_FRAME_H
#define N2D2_LSQQUANTIZERCELL_FRAME_H

#include "Quantizer/QAT/Cell/QuantizerCell_Frame.hpp"
#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell.hpp"
#include "Solver/SGDSolver_Frame.hpp"

namespace N2D2 {

/**
 * @brief Implementation of the weights quantization using Learned Step Size Quantization (LSQ) method for CPU
 * 
 * @tparam T weights type.
 * Can be : double (64 bits)
 *          float (32 bits)
 *          half_float::half (16 bits)
 */
template <class T>
class LSQQuantizerCell_Frame: public LSQQuantizerCell, public QuantizerCell_Frame<T> {
public:

    using QuantizerCell_Frame<T>::mFullPrecisionWeights;
    using QuantizerCell_Frame<T>::mQuantizedWeights;
    using QuantizerCell_Frame<T>::mDiffFullPrecisionWeights;
    using QuantizerCell_Frame<T>::mDiffQuantizedWeights;

    using QuantizerCell_Frame<T>::mFullPrecisionBiases;
    using QuantizerCell_Frame<T>::mQuantizedBiases;
    using QuantizerCell_Frame<T>::mDiffFullPrecisionBiases;
    using QuantizerCell_Frame<T>::mDiffQuantizedBiases;


    static std::shared_ptr<LSQQuantizerCell_Frame<T>> create()
    {
        return std::make_shared<LSQQuantizerCell_Frame<T>>();
    };

    LSQQuantizerCell_Frame();

    virtual void addWeights(BaseTensor& weights, BaseTensor& diffWeights);
    virtual void addBiases(BaseTensor& biases, BaseTensor& diffBiases);

    virtual void initialize();
    virtual void update(unsigned int batchSize);
    virtual void propagate();
    virtual void back_propagate();

    virtual void setSolver(const std::shared_ptr<Solver>& solver)
    {
       mSolver = solver;
    };
    
    virtual std::shared_ptr<Solver> getSolver()
    {
        return mSolver;
    };

    Tensor<T>& getStepSize()
    {
        return mStepSize;
    };

    // Essentially used for unit tests
    Tensor<T>& getDiffStepSize()
    {
        return mDiffStepSize;
    };

    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName, bool ignoreNoExists);
    virtual ~LSQQuantizerCell_Frame();

protected:
    Interface<T> mDiffStepSizeInterface;
    Tensor<T> mStepSize; /**Tensor with the step size value*/
    Tensor<T> mDiffStepSize; /**Tensor with the step size gradient value of the current iteration*/

    T mGradScaleFactor; /**Gradient scale factor aiming at a better convergence during training*/
    bool mInitialized = false;

private:
    static Registrar<LSQQuantizerCell> mRegistrar;
};

}

#endif  // N2D2_LSQQUANTIZERCELL_FRAME_H