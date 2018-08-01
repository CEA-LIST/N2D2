/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_SGDSOLVER_FRAME_CUDA_H
#define N2D2_SGDSOLVER_FRAME_CUDA_H

#include "Solver/SGDSolver.hpp"
#include "Solver/SGDSolver_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T> class SGDSolver_Frame_CUDA : public SGDSolver {
public:
    static std::shared_ptr<SGDSolver> create()
    {
        return std::make_shared<SGDSolver_Frame_CUDA<T> >();
    }

    SGDSolver_Frame_CUDA();
    void update(BaseTensor& data, BaseTensor& diffData, unsigned int batchSize);
    void update(CudaTensor<T>& data, CudaTensor<T>& diffData,
                unsigned int batchSize);
    void exportFreeParameters(const std::string& fileName) const;

    std::shared_ptr<SGDSolver_Frame_CUDA<T> > clone() const
    {
        return std::shared_ptr<SGDSolver_Frame_CUDA<T> >(doClone());
    }
    virtual ~SGDSolver_Frame_CUDA() {};

protected:
    // inline void setMomentum(unsigned int output, unsigned int channel,
    // unsigned int sx, unsigned int sy, float value);

    using SGDSolver::mLearningRate;
    using SGDSolver::mMomentum;
    using SGDSolver::mDecay;
    using SGDSolver::mPower;
    using SGDSolver::mIterationSize;
    using SGDSolver::mMaxIterations;
    using SGDSolver::mLearningRatePolicy;
    using SGDSolver::mLearningRateStepSize;
    using SGDSolver::mLearningRateDecay;
    using SGDSolver::mClamping;
    using SGDSolver::mIterationPass;
    using SGDSolver::mNbIterations;

    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;

    CudaTensor<T> mMomentumData;
    CudaTensor<T> mContinuousData;

private:
    virtual SGDSolver_Frame_CUDA<T>* doClone() const
    {
        return new SGDSolver_Frame_CUDA<T>(*this);
    }

    static Registrar<SGDSolver> mRegistrar;
};
}

template <class T>
N2D2::SGDSolver_Frame_CUDA<T>::SGDSolver_Frame_CUDA()
    : SGDSolver::SGDSolver(),
      mQuantizationLevels(this, "QuantizationLevels", 0U)
{
    // ctor
}

template <class T>
void N2D2::SGDSolver_Frame_CUDA<T>::update(BaseTensor& data,
                                           BaseTensor& diffData,
                                           unsigned int batchSize)
{
    update(dynamic_cast<CudaTensor<T>&>(data),
           dynamic_cast<CudaTensor<T>&>(diffData),
           batchSize);
}

template <class T>
void N2D2::SGDSolver_Frame_CUDA<T>::exportFreeParameters(
    const std::string& fileName) const
{
    if (mMomentum != 0.0 && mMomentumData.size() > 0) {
        std::ofstream syn(fileName);

        if (!syn.good())
            throw std::runtime_error("Could not create synaptic file : "
                                     + fileName);

        mMomentumData.synchronizeDToH();

        for (typename std::vector<T>::const_iterator it = mMomentumData.begin();
             it != mMomentumData.end();
             ++it) {
            syn << (*it) << " ";
            syn << "\n";
        }

        if (!syn.good())
            throw std::runtime_error("Error writing synaptic file: "
                                     + fileName);
    }
}

namespace N2D2 {
template <>
void SGDSolver_Frame_CUDA<float>::update(CudaTensor<float>& data,
                                         CudaTensor<float>& diffData,
                                         unsigned int batchSize);

template <>
void SGDSolver_Frame_CUDA<double>::update(CudaTensor<double>& data,
                                          CudaTensor<double>& diffData,
                                          unsigned int batchSize);
}

#endif // N2D2_SGDSOLVER_FRAME_CUDA_H
