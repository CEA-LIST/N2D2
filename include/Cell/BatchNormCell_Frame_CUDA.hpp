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

#ifndef N2D2_BATCHNORMCELL_FRAME_CUDA_H
#define N2D2_BATCHNORMCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "BatchNormCell.hpp"
#include "Cell_Frame_CUDA.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include "DeepNet.hpp"

namespace N2D2 {
template <class T>
class BatchNormCell_Frame_CUDA : public virtual BatchNormCell,
                                 public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::keepInSync;
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivationDesc;
    using Cell_Frame_CUDA<T>::mKeepInSync;

    BatchNormCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                             unsigned int nbOutputs,
                             const std::shared_ptr
                             <Activation>& activation
                             = std::make_shared
                             <TanhActivation_Frame_CUDA<T> >());
    static std::shared_ptr<BatchNormCell>
    create(const DeepNet& deepNet, const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation>& activation
           = std::make_shared<TanhActivation_Frame_CUDA<T> >())
    {
        return std::make_shared
            <BatchNormCell_Frame_CUDA>(deepNet, name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getScale(unsigned int index, BaseTensor& value) const;
    inline void getBias(unsigned int index, BaseTensor& value) const;
    inline void getMean(unsigned int index, BaseTensor& value) const;
    inline void getVariance(unsigned int index, BaseTensor& value) const;
    inline void setScale(unsigned int index, const BaseTensor& value);
    inline void setBias(unsigned int index, const BaseTensor& value);
    inline void setMean(unsigned int index, const BaseTensor& value);
    inline void setVariance(unsigned int index, const BaseTensor& value);
    inline std::shared_ptr<BaseTensor> getScales() const
    {
        return mScale;
    };
    void setScales(const std::shared_ptr<BaseTensor>& scales);
    inline std::shared_ptr<BaseTensor> getBiases() const
    {
        return mBias;
    };
    void setBiases(const std::shared_ptr<BaseTensor>& biases);
    inline std::shared_ptr<BaseTensor> getMeans() const
    {
        return mMean;
    };
    void setMeans(const std::shared_ptr<BaseTensor>& means);
    inline std::shared_ptr<BaseTensor> getVariances() const
    {
        return mVariance;
    };
    void setVariances(const std::shared_ptr<BaseTensor>& variances);
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);

    void synchronizeToH(bool keepInSync_) const;
    void synchronizeToD(bool keepInSync_);
    virtual ~BatchNormCell_Frame_CUDA();

protected:
    typedef typename Cuda::cudnn_scaling_type<T>::type ParamT;

    cudnnBatchNormMode_t mMode;
    unsigned int mNbPropagate;
    std::shared_ptr<CudaTensor<ParamT> > mScale;
    std::shared_ptr<CudaTensor<ParamT> > mBias;
    CudaTensor<ParamT> mDiffScale;
    CudaTensor<ParamT> mDiffBias;
    std::shared_ptr<CudaTensor<ParamT> > mMean;
    std::shared_ptr<CudaTensor<ParamT> > mVariance;
    CudaTensor<ParamT> mSavedMean;
    CudaTensor<ParamT> mSavedVariance;

private:
    static Registrar<BatchNormCell> mRegistrar;
};
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::setScale(unsigned int index,
                                                 const BaseTensor& value)
{
    (*mScale)(index) = tensor_cast<ParamT>(value)(0);

    if (mKeepInSync)
        mScale->synchronizeHToD(index, 1);
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::getScale(unsigned int index,
                                                 BaseTensor& value) const
{
    if (mKeepInSync)
        mScale->synchronizeDToH(index, 1);

    value.resize({1});
    value = Tensor<ParamT>({1}, (*mScale)(index));
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::setBias(unsigned int index,
                                                const BaseTensor& value)
{
    (*mBias)(index) = tensor_cast<ParamT>(value)(0);

    if (mKeepInSync)
        mBias->synchronizeHToD(index, 1);
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::getBias(unsigned int index,
                                                BaseTensor& value) const
{
    if (mKeepInSync)
        mBias->synchronizeDToH(index, 1);

    value.resize({1});
    value = Tensor<ParamT>({1}, (*mBias)(index));
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::setMean(unsigned int index,
                                                const BaseTensor& value)
{
    (*mMean)(index) = tensor_cast<ParamT>(value)(0);

    if (mKeepInSync)
        mMean->synchronizeHToD(index, 1);
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::getMean(unsigned int index,
                                                BaseTensor& value) const
{
    if (mKeepInSync)
        mMean->synchronizeDToH(index, 1);

    value.resize({1});
    value = Tensor<ParamT>({1}, (*mMean)(index));
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::setVariance(unsigned int index,
                                                    const BaseTensor& value)
{
    (*mVariance)(index) = tensor_cast<ParamT>(value)(0);

    if (mKeepInSync)
        mVariance->synchronizeHToD(index, 1);
}

template <class T>
void N2D2::BatchNormCell_Frame_CUDA<T>::getVariance(unsigned int index,
                                                    BaseTensor& value) const
{
    if (mKeepInSync)
        mVariance->synchronizeDToH(index, 1);

    value.resize({1});
    value = Tensor<ParamT>({1}, (*mVariance)(index));
}

#endif // N2D2_BATCHNORMCELL_FRAME_CUDA_H
