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

#ifndef N2D2_SATQUANTIZERACTIVATION_FRAME_CUDA_H
#define N2D2_SATQUANTIZERACTIVATION_FRAME_CUDA_H

#include "Quantizer/QAT/Activation/QuantizerActivation_Frame_CUDA.hpp"
#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"

namespace N2D2 {

template <class T>
class SATQuantizerActivation_Frame_CUDA: public SATQuantizerActivation, public QuantizerActivation_Frame_CUDA<T> {
public:

    using QuantizerActivation_Frame_CUDA<T>::mFullPrecisionActivations;

    static std::shared_ptr<SATQuantizerActivation_Frame_CUDA<T>> create()
    {
        return std::make_shared<SATQuantizerActivation_Frame_CUDA<T>>();
    };

    SATQuantizerActivation_Frame_CUDA();

    virtual void update(unsigned int batchSize = 1);
    virtual void propagate(BaseTensor& baseInOut,
                            bool inference);

    virtual void back_propagate(const BaseTensor& input,
                                const BaseTensor& output,
                                const BaseTensor& diffInput,
                                BaseTensor& diffOutput);

    virtual void setSolver(const std::shared_ptr<Solver>& solver)
    {
       mSolver = solver;
    };
    
    virtual std::shared_ptr<Solver> getSolver()
    {
        return mSolver;
    };

    void syncAlpha()
    {
        std::cout << "syncAlpha value " << mAlphaParameter << std::endl;
    };

    CudaTensor<T>& getAlpha()
    {
        if(!mSyncAlpha) {
            if(mAlphas.empty()) {
                mAlphas.resize({1, 1, 1, 1});
                mAlphas.fill(T(mAlphaParameter));
                mAlphas.synchronizeHToD();
            }
            mSyncAlpha = true;
        }

        return mAlphas;
    };

    // getDiffAlpha used for unitary test
    CudaTensor<T>& getDiffAlpha()
    {
        return mDiffAlphas;
    };

    virtual ~SATQuantizerActivation_Frame_CUDA();

    void exportParameters(const std::string& fileName, const std::string& cellName) const;
    void importParameters(const std::string& dirName, const std::string& cellName, bool ignoreNotExists);

protected:
    // Has to be saved for forward pass
    CudaTensor<T> mAlphas;
    CudaTensor<T> mDiffAlphas;
    CudaTensor<T> mDiffAlphasTensor;
    CudaTensor<T> mDeviceWorkspace;

private:
    static Registrar<SATQuantizerActivation> mRegistrar;
};

}

#endif  // N2D2_SATQUANTIZERACTIVATION_FRAME_CUDA_H
