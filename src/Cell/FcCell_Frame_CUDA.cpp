/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#include "Cell/FcCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                    N2D2::FcCell_Frame_CUDA::create);

N2D2::FcCell_Frame_CUDA::FcCell_Frame_CUDA(const std::string& name,
                                           unsigned int nbOutputs,
                                           const std::shared_ptr
                                           <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      FcCell(name, nbOutputs),
      Cell_Frame_CUDA(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mOnesVector(NULL),
      mSynchronized(false)
{
    // ctor
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();
}

void N2D2::FcCell_Frame_CUDA::initialize()
{
    if (!mNoBias) {
        mBias.resize(mOutputs.dimZ());
        mDiffBias.resize(mOutputs.dimZ());
        mBiasFiller->apply(mBias);
        mBias.synchronizeHToD();

        //  1   <-->    batch   <-->    mInputs.b()
        CHECK_CUDA_STATUS(
            cudaMalloc(&mOnesVector, mInputs.dimB() * sizeof(Float_T)));
        std::vector<Float_T> onesVec(mInputs.dimB(), 1.0);
        CHECK_CUDA_STATUS(cudaMemcpy(mOnesVector,
                                     &onesVec[0],
                                     mInputs.dimB() * sizeof(Float_T),
                                     cudaMemcpyHostToDevice));
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FcCell " + mName);

        mWeightsSolvers.push_back(mWeightsSolver->clone());
        mSynapses.push_back(new CudaTensor4d<Float_T>(
            1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()));
        mDiffSynapses.push_back(new CudaTensor4d<Float_T>(
            1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()));
        mWeightsFiller->apply(mSynapses.back());
        mSynapses.back().synchronizeHToD();
    }
}

void N2D2::FcCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    float beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        if (k > 0)
            beta = 1.0f;

        // Computes mOutputs = alpha*mSynapses'*mInputs + beta*mOutputs
        CHECK_CUBLAS_STATUS(cublasSgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_T, // mSynapses'
            CUBLAS_OP_N, // mInputs
            mOutputs.dimZ(), // nb rows in mSynapses' and mOutputs
            mInputs.dimB(), // nb cols in mInputs and mOutputs
            inputSize, // nb cols in mSynapses' and nb rows in mInputs
            &alpha,
            mSynapses[k].getDevicePtr(),
            inputSize,
            mInputs[k].getDevicePtr(),
            inputSize,
            &beta,
            mOutputs.getDevicePtr(),
            mOutputs.dimZ()));
    }

    if (!mNoBias) {
        // Computes mOutputs = alpha*mBias*mOnesVector + alpha*mOutputs
        CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        1,
                                        &alpha,
                                        mBias.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &alpha,
                                        mOutputs.getDevicePtr(),
                                        mOutputs.dimZ()));
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::FcCell_Frame_CUDA::backPropagate()
{
    Cell_Frame_CUDA::backPropagate();

    //  1   <-->    batch   <-->    mInputs.b()

    const float alpha = 1.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        const float beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

        // mDiffSynapses.getDevicePtr() = mInputs.getDevicePtr *
        // mDiffInputs.getDevicePtr*
        CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        inputSize,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        mInputs[k].getDevicePtr(),
                                        inputSize,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        &beta,
                                        mDiffSynapses[k].getDevicePtr(),
                                        inputSize));
    }

    if (!mNoBias) {
        const float beta = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

        // mDiffBias.getDevicePtr() = mDiffInputs.getDevicePtr * mOnesVector
        CHECK_CUBLAS_STATUS(cublasSgemv(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &beta,
                                        mDiffBias.getDevicePtr(),
                                        1));
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const float betaData = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;
            const unsigned int diffOutputSize = mDiffOutputs[k].dimX()
                                                * mDiffOutputs[k].dimY()
                                                * mDiffOutputs[k].dimZ();

            // mDiffOutputs.getDevicePtr = mSynapses.getDevicePtr() *
            // mDiffInputs.getDevicePtr
            CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            diffOutputSize,
                                            mInputs.dimB(),
                                            mOutputs.dimZ(),
                                            &alpha,
                                            mSynapses[k].getDevicePtr(),
                                            diffOutputSize,
                                            mDiffInputs.getDevicePtr(),
                                            mOutputs.dimZ(),
                                            &betaData,
                                            mDiffOutputs[k].getDevicePtr(),
                                            diffOutputSize));

            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeDToHBased();
    } // Sinon il s'agit de la première couche, inutile de calculer
}

void N2D2::FcCell_Frame_CUDA::update()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]
            ->update(&mSynapses[k], &mDiffSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(&mBias, &mDiffBias, mInputs.dimB());
}

void N2D2::FcCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&FcCell_Frame_CUDA::propagate, this, false),
                  std::bind(&FcCell_Frame_CUDA::backPropagate, this));

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream name;
        name << mName + "mDiffSynapses[" << k << "]";

        gc.check(name.str(), mSynapses[k], mDiffSynapses[k]);
    }

    if (!mNoBias)
        gc.check(mName + "_mDiffBias", mBias, mDiffBias);

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

void N2D2::FcCell_Frame_CUDA::logFreeParameters(const std::string& fileName,
                                                unsigned int output) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParameters(fileName, output);
    mSynchronized = false;
}

void N2D2::FcCell_Frame_CUDA::logFreeParameters(const std::string
                                                & dirName) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParameters(dirName);
    mSynchronized = false;
}

void N2D2::FcCell_Frame_CUDA::saveFreeParameters(const std::string
                                                 & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    mSynapses.synchronizeDToH();

    for (unsigned int k = 0; k < mSynapses.size(); ++k) {
        for (std::vector<Float_T>::const_iterator it = mSynapses[k].begin();
             it != mSynapses[k].end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    mBias.synchronizeDToH();

    for (std::vector<Float_T>::const_iterator it = mBias.data().begin();
         it != mBias.data().end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

void N2D2::FcCell_Frame_CUDA::loadFreeParameters(const std::string& fileName,
                                                 bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file (.SYN): "
                                     + fileName);
    }

    for (unsigned int k = 0; k < mSynapses.size(); ++k) {
        for (std::vector<Float_T>::iterator it = mSynapses[k].begin();
             it != mSynapses[k].end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    mSynapses.synchronizeHToD();

    for (std::vector<Float_T>::iterator it = mBias.data().begin();
         it != mBias.data().end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    mBias.synchronizeHToD();

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

void N2D2::FcCell_Frame_CUDA::exportFreeParameters(const std::string
                                                   & fileName) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::exportFreeParameters(fileName);
    mSynchronized = false;
}

void N2D2::FcCell_Frame_CUDA::exportSolverParameters(const std::string
                                                     & fileName) const
{
    for (unsigned int i = 0; i < mSynapses.size(); ++i)
        mWeightsSolvers[i]->exportFreeParameters(fileName);
}

void N2D2::FcCell_Frame_CUDA::importFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    mSynchronized = true;
    FcCell::importFreeParameters(fileName, ignoreNotExists);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

void N2D2::FcCell_Frame_CUDA::logFreeParametersDistrib(const std::string
                                                       & fileName) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParametersDistrib(fileName);
    mSynchronized = false;
}

void N2D2::FcCell_Frame_CUDA::discretizeFreeParameters(unsigned int nbLevels)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::discretizeFreeParameters(nbLevels);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA::getFreeParametersRange() const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRange();
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();

    return range;
}

void N2D2::FcCell_Frame_CUDA::processFreeParameters(const std::function
                                                <double(const double&)>& func)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::processFreeParameters(func);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

N2D2::FcCell_Frame_CUDA::~FcCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k)
        delete &mSynapses[k];

    if (mOnesVector != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mOnesVector));
        mOnesVector = NULL;
    }
}

#endif
