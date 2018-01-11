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
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
template <class T> class SGDSolver_Frame_CUDA : public SGDSolver<T> {
public:
    static std::shared_ptr<SGDSolver<T> > create()
    {
        return std::make_shared<SGDSolver_Frame_CUDA<T> >();
    }

    SGDSolver_Frame_CUDA();
    void
    update(Tensor4d<T>* data, Tensor4d<T>* diffData, unsigned int batchSize);
    void exportFreeParameters(const std::string& fileName) const;

    std::shared_ptr<SGDSolver_Frame_CUDA<T> > clone() const
    {
        return std::shared_ptr<SGDSolver_Frame_CUDA<T> >(doClone());
    }
    virtual ~SGDSolver_Frame_CUDA() {};

protected:
    // inline void setMomentum(unsigned int output, unsigned int channel,
    // unsigned int sx, unsigned int sy, float value);

    using SGDSolver<T>::mLearningRate;
    using SGDSolver<T>::mMomentum;
    using SGDSolver<T>::mDecay;
    using SGDSolver<T>::mPower;
    using SGDSolver<T>::mIterationSize;
    using SGDSolver<T>::mMaxIterations;
    using SGDSolver<T>::mLearningRatePolicy;
    using SGDSolver<T>::mLearningRateStepSize;
    using SGDSolver<T>::mLearningRateDecay;
    using SGDSolver<T>::mClamping;
    using SGDSolver<T>::mIterationPass;
    using SGDSolver<T>::mNbIterations;

    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;

    CudaTensor4d<T> mMomentumData;
    CudaTensor4d<T> mContinuousData;

private:
    virtual SGDSolver_Frame_CUDA<T>* doClone() const
    {
        return new SGDSolver_Frame_CUDA<T>(*this);
    }

    static Registrar<SGDSolver<T> > mRegistrar;
};
}

template <class T>
N2D2::SGDSolver_Frame_CUDA<T>::SGDSolver_Frame_CUDA()
    : SGDSolver<T>::SGDSolver(),
      mQuantizationLevels(this, "QuantizationLevels", 0U)
{
    // ctor
}

namespace N2D2 {
template <>
void SGDSolver_Frame_CUDA<float>::update(Tensor4d<float>* data,
                                         Tensor4d<float>* diffData,
                                         unsigned int batchSize);

template <>
void SGDSolver_Frame_CUDA<double>::update(Tensor4d<double>* data,
                                          Tensor4d<double>* diffData,
                                          unsigned int batchSize);

template <>
void SGDSolver_Frame_CUDA
    <float>::exportFreeParameters(const std::string& fileName) const;

template <>
void SGDSolver_Frame_CUDA
    <double>::exportFreeParameters(const std::string& fileName) const;
}

#endif // N2D2_SGDSOLVER_FRAME_CUDA_H
