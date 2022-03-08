/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifdef CUDA

#include <stdexcept>
#include <string>

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_CUDA.hpp"
#include "Cell/DistanceCell.hpp"
#include "Cell/DistanceCell_Frame_CUDA.hpp"
#include "Cell/DistanceCell_Frame_CUDA_Kernels.hpp"
#include "Filler/Filler.hpp"
#include "Filler/XavierFiller.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "utils/Utils.hpp"

static const N2D2::Registrar<N2D2::DistanceCell> registrarFloat(
                    "Frame_CUDA", N2D2::DistanceCell_Frame_CUDA<float>::create,
                    N2D2::Registrar<N2D2::DistanceCell>::Type<float>());

template<class T>
N2D2::DistanceCell_Frame_CUDA<T>::DistanceCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                                        unsigned int nbOutputs, double margin, double centercoef)
    : Cell(deepNet, name, nbOutputs),
      DistanceCell(deepNet, name, nbOutputs, std::move(margin), std::move(centercoef)),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs),
      mMean(std::make_shared<CudaTensor<T> >())
{
    mWeightsFiller = std::make_shared<XavierFiller<T> >(XavierFiller<T>::VarianceNorm::Average, XavierFiller<T>::Distribution::Uniform, 2.0);
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::save(const std::string& dirName) const
{
    Cell_Frame_CUDA<T>::save(dirName);

    std::stringstream solverName;
    solverName << "WeightsSolver";

    mWeightsSolver->save(dirName + "/" + solverName.str());
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::load(const std::string& dirName)
{
    Cell_Frame_CUDA<T>::load(dirName);

    std::stringstream solverName;
    solverName << "WeightsSolver";

    mWeightsSolver->load(dirName + "/" + solverName.str());
}

template<class T>
void N2D2::DistanceCell_Frame_CUDA<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for DistanceCell '" + mName + "'.");
    }

    if (mMean->empty()) 
    {
        mMean->resize({1, 1, mInputs[0].dimZ(), this->getNbOutputs()});
        mWeightsFiller->apply((*mMean));
        mMean->synchronizeHToD();
    }
    
    if (mDiffMean.empty()) 
    {       
        mDiffMean.resize({1, 1, mInputs[0].dimZ(), this->getNbOutputs()});
        mDiffMean.fill(T(0.0));
        mDiffMean.synchronizeHToD();
    }

    if (mDist.empty()) 
    { 
        mDist.resize(mOutputs.dims());
    }

    if (mLabels.empty()) 
    {
        mLabels.resize({1, 1, this->getNbOutputs(), mInputs[0].dimB()});
    }
}

template<class T>
void N2D2::DistanceCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    std::shared_ptr<CudaDeviceTensor<T> > input0
            = cuda_device_tensor_cast<T>(mInputs[0]);

    unsigned int size = this->getNbOutputs() * mInputs[0].dimB();
    unsigned int feat_dim = mMean->dimZ();

    cudaDistanceL2Forward(size, 
                            this->getNbOutputs(), 
                            feat_dim,
                            input0->getDevicePtr(), 
                            mMean->getDevicePtr(), 
                            //mSigma.getDevicePtr(), 
                            mDist.getDevicePtr(),
                            mOutputs.getDevicePtr());

    if (!inference) {
        mCurrentMargin = (T)mMargin;
        if (mItCounter < mEndIT) {
            mCurrentMargin = (T)mItCounter*((T)mMargin/(T)mEndIT);
        } 
        mItCounter+=mInputs[0].dimB();

        mLabels.deviceTensor() = mDiffInputs.deviceTensor();

        cudaMargin(size, 
                    mLabels.getDevicePtr(), //labels
                    (T)mCurrentMargin,
                    mOutputs.getDevicePtr());
    }

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
    mDiffMean.clearValid();
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<T>::backPropagate();

    std::shared_ptr<CudaDeviceTensor<T> > diffOutput0
        = (mDiffOutputs[0].isValid())
            ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
            : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

    std::shared_ptr<CudaDeviceTensor<T> > input0
            = cuda_device_tensor_cast<T>(mInputs[0]);


    unsigned int size_mean = mDiffMean.size();
    unsigned int nb_class = this->getNbOutputs();
    unsigned int feat_dim = mDiffMean.dimZ();
    unsigned int batchsize = mDiffOutputs[0].dimB();
    T center_coef = (T)mCenterCoef / (T)batchsize;

    cudaDistanceL2Backward_mean(size_mean,
                                    nb_class, 
                                    feat_dim,
                                    batchsize,
                                    (T)mCurrentMargin,
                                    center_coef,
                                    mLabels.getDevicePtr(),
                                    input0->getDevicePtr(),
                                    mMean->getDevicePtr(), 
                                    mDiffInputs.getDevicePtr(), 
                                    mDiffMean.getDevicePtr());
    
    mDiffMean.setValid();
    mDiffMean.synchronizeDToH();

    unsigned int size_input = mDiffOutputs[0].size();

    cudaDistanceL2Backward_input(size_input,
                                    nb_class, 
                                    feat_dim,
                                    (T)mCurrentMargin,
                                    center_coef,
                                    mLabels.getDevicePtr(),
                                    input0->getDevicePtr(),
                                    mMean->getDevicePtr(), 
                                    mDiffInputs.getDevicePtr(), 
                                    diffOutput0->getDevicePtr());

    mDiffOutputs[0].setValid();
    mDiffOutputs.synchronizeDToHBased();
}

template<class T>
void N2D2::DistanceCell_Frame_CUDA<T>::update() {
    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    if (mDiffMean.isValid()) {
        mDiffMean.aggregateAllTo(dev, mDevices);
        mWeightsSolver->update((*mMean), mDiffMean, mInputs.dimB());
        mMean->broadcastAllFrom(dev, mDevices);
    }

    Cell_Frame_CUDA<T>::update();
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::setWeights(
    const std::shared_ptr<BaseTensor>& weights)
{
    std::shared_ptr<CudaTensor<T> > cudaWeights
        = std::dynamic_pointer_cast<CudaTensor<T> >(weights);

     if (!cudaWeights) {
        throw std::runtime_error("DistanceCell_Frame_CUDA::setWeights(): weights"
                                 " must be a CudaTensor");
    }

    mMean = cudaWeights;
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::importFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    keepInSync(false);
    DistanceCell::importFreeParameters(fileName, ignoreNotExists);
    synchronizeToD(true);
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::exportFreeParameters(const std::string
                                                   & fileName) const
{
    synchronizeToH(false);
    DistanceCell::exportFreeParameters(fileName);
    keepInSync(true);
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::synchronizeToH(bool keepInSync_) const
{
    mMean->synchronizeDToH();
    keepInSync(keepInSync_);
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::synchronizeToD(bool keepInSync_)
{
    mMean->synchronizeHToD();
    keepInSync(keepInSync_);

    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
    
    mMean->broadcastAllFrom(dev);
}

template<class T>
double N2D2::DistanceCell_Frame_CUDA<T>::applyLoss(
    double /*targetVal*/,
    double /*defaultVal*/)
{
    // Distance takes a target for the forward pass only, which should no set
    // mDiffInputs
    return 0.0;
}

template<class T>
double N2D2::DistanceCell_Frame_CUDA<T>::applyLoss()
{
    // Distance takes a target for the forward pass only, which should no set
    // mDiffInputs
    return 0.0;
}

template<class T>
void N2D2::DistanceCell_Frame_CUDA<T>::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
   
   /* GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&DistanceCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&DistanceCell_Frame_CUDA<T>::backPropagate, this));

    for (unsigned int k = 0; k < mInputs.size(); ++k) {
        if (mDiffOutputs[k].empty()) {
            std::cout << Utils::cwarning << "Empty diff. outputs #" << k
                    << " for cell " << mName
                    << ", could not check the gradient!" << Utils::cdef
                    << std::endl;
            continue;
        }

        std::stringstream name;
        name << mName + "_mDiffOutputs[" << k << "]";

        gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
    }*/
}

namespace N2D2 {
    template class DistanceCell_Frame_CUDA<float>;
}



#endif