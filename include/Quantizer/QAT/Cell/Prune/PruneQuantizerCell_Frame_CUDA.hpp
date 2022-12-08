/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#ifndef N2D2_PRUNEQUANTIZERCELL_FRAME_CUDA_H
#define N2D2_PRUNEQUANTIZERCELL_FRAME_CUDA_H

#include "Quantizer/QAT/Cell/QuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "utils/Scheduler.hpp"

namespace N2D2 {
    
template <class T>
class PruneQuantizerCell_Frame_CUDA: public PruneQuantizerCell, public QuantizerCell_Frame_CUDA<T> {
public:

    using QuantizerCell_Frame_CUDA<T>::mFullPrecisionWeights;
    using QuantizerCell_Frame_CUDA<T>::mQuantizedWeights;
    using QuantizerCell_Frame_CUDA<T>::mDiffFullPrecisionWeights;
    using QuantizerCell_Frame_CUDA<T>::mDiffQuantizedWeights;

    using QuantizerCell_Frame_CUDA<T>::mFullPrecisionBiases;
    using QuantizerCell_Frame_CUDA<T>::mQuantizedBiases;
    using QuantizerCell_Frame_CUDA<T>::mDiffFullPrecisionBiases;
    using QuantizerCell_Frame_CUDA<T>::mDiffQuantizedBiases;


    static std::shared_ptr<PruneQuantizerCell_Frame_CUDA<T>> create()
    {
        return std::make_shared<PruneQuantizerCell_Frame_CUDA<T>>();
    };

    PruneQuantizerCell_Frame_CUDA();

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
    virtual BaseTensor& getMasksWeights(unsigned int k)
    {
        return mMasksWeights[k];
    };

    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName, bool ignoreNoExists);

    virtual ~PruneQuantizerCell_Frame_CUDA();

protected:
    CudaInterface<unsigned int> mMasksWeights;
    unsigned int mNbZeroWeights = 0U;
    unsigned int mNbZeroMaxWeights = 0U;
    
    bool mInitialized = false;

    // Variables for Gradual pruning mode
    std::shared_ptr<Scheduler> mScheduler;

private:
    static Registrar<PruneQuantizerCell> mRegistrar;
};

}

#endif  // N2D2_PRUNEQUANTIZERCELL_FRAME_CUDA_H