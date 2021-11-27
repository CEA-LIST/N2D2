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

#include "GradientCheck.hpp"
#include "Cell/ElemWiseCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::ElemWiseCell>
N2D2::ElemWiseCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                      N2D2::ElemWiseCell_Frame_CUDA::create);

N2D2::ElemWiseCell_Frame_CUDA::ElemWiseCell_Frame_CUDA(
    const DeepNet& deepNet, 
    const std::string& name,
    unsigned int nbOutputs,
    Operation operation,
   CoeffMode mode,
    const std::vector<Float_T>& weights,
    const std::vector<Float_T>& shifts,
    const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      ElemWiseCell(deepNet, name,
               nbOutputs,
               operation,
               mode,
               weights,
               shifts),
      Cell_Frame_CUDA<Float_T>(deepNet, name, nbOutputs, activation)
{
    // ctor
}

void N2D2::ElemWiseCell_Frame_CUDA::initialize()
{
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].dimZ() != mOutputs.dimZ()
            || mInputs[k].dimB() != mOutputs.dimB())
        {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame_CUDA::initialize(): for cell "
                << mName << ": the input tensor dimensions ("
                << mInputs[k].dims() << ") must match the output dimensions ("
                << mOutputs.dims() << ") for input #" << k << ".";

            throw std::runtime_error(errorMsg.str());
        }
    }
    if(mWeights.empty()) {
        std::cout << Utils::cwarning << "ElemWiseCell_Frame::initialize(): Empty weights for cell " 
                    << mName << ": Initialize weights to default value(1.0)" 
                    << Utils::cdef << std::endl;
    }
    if(mShifts.empty()) {
        std::cout << Utils::cwarning << "ElemWiseCell_Frame::initialize(): Empty shifts for cell " 
                    << mName << ": Initialize shifts to default value(0.0)" 
                    << Utils::cdef << std::endl;
    }

    if(mCoeffMode == ElemWiseCell::PerChannel) {
        std::size_t nbChannels = 0;
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            nbChannels += mInputs[k].dimZ();
        }

        mWeights.resize(nbChannels, 1.0);
        mShifts.resize(nbChannels, 0.0);
    } 
    else {
        mWeights.resize(mInputs.size(), 1.0);
        mShifts.resize(mInputs.size(), 0.0);
    }

    if (mOperation == Max)
        mArgMax.resize(mOutputs.dims());

    if (mOperation == EuclideanSum || mOperation == Prod)
        mInterTerm.resize(mOutputs.dims());
}


void N2D2::ElemWiseCell_Frame_CUDA::initializeDataDependent()
{
    // NOTE: this is addition to initialize()
    Cell_Frame_CUDA<Float_T>::initializeDataDependent();

    initialize();
}

void N2D2::ElemWiseCell_Frame_CUDA::propagate(bool inference)
{
    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    mInputs.synchronizeHBasedToD();

    std::shared_ptr<CudaDeviceTensor<Float_T> > input0
        = cuda_device_tensor_cast<Float_T>(mInputs[0]);

    if (mOperation == Sum) {
        // mOutputs <- mWeights[0] * mInputs[0] + mShifts[0]
        if(mCoeffMode == ElemWiseCell::PerInput 
                || mCoeffMode == ElemWiseCell::PerLayer) {
            cudaScale(nbElems,
                    input0->getDevicePtr(),
                    mWeights[0],
                    mShifts[0],
                    0.0f,
                    mOutputs.getDevicePtr());

            for (unsigned int k = 1; k < nbInputs; ++k) {
                // mOutputs <- mWeights[k] * mInputs[k] + mOutputs
                std::shared_ptr<CudaDeviceTensor<Float_T> > input
                    = cuda_device_tensor_cast<Float_T>(mInputs[k]);
                CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                                nbElems,
                                                &(mWeights[k]),
                                                input->getDevicePtr(),
                                                1,
                                                mOutputs.getDevicePtr(),
                                                1));

                // mOutputs <- mOutputs + mShifts[k]
                if(mShifts[k] != 0.0)
                    cudaScale(nbElems,
                                mOutputs.getDevicePtr(),
                                1.0f,
                                mShifts[k],
                                0.0f,
                                mOutputs.getDevicePtr());
            }
        }
        else if(mCoeffMode == ElemWiseCell::PerChannel){
            const unsigned int nbChElems = mInputs.dimY()*mInputs.dimX();

            for(unsigned int b = 0; b < mInputs.dimB(); ++b) {
                const unsigned int batchoffset 
                            = b*mInputs[0].dimZ()*nbChElems;
                unsigned int chOffset = batchoffset;
                for(unsigned int ch = 0; ch < mInputs[0].dimZ(); ++ch) {
                    cudaScale(  nbChElems,
                                input0->getDevicePtr() + chOffset,
                                mWeights[ch],
                                mShifts[ch],
                                0.0f,
                                mOutputs.getDevicePtr() + chOffset);

                    chOffset += nbChElems;
                }
                for (unsigned int k = 1; k < nbInputs; ++k) {
                    std::shared_ptr<CudaDeviceTensor<Float_T> > input
                        = cuda_device_tensor_cast<Float_T>(mInputs[k]);
                    chOffset = batchoffset;
                    for(unsigned int ch = 0; ch < mInputs[k].dimZ(); ++ch) {
                        cudaScale(  nbChElems,
                                    input->getDevicePtr() + chOffset,
                                    mWeights[ch],
                                    mShifts[ch],
                                    1.0f,
                                    mOutputs.getDevicePtr() + chOffset);
                        chOffset += nbChElems;
                    }
                }
            }
        }
    }
    else if (mOperation == AbsSum) {
        if(mCoeffMode == ElemWiseCell::PerInput 
                || mCoeffMode == ElemWiseCell::PerLayer) {
            // mOutputs <- mWeights[0] * |mInputs[0]|
            cudaScaleAbs(nbElems,
                    input0->getDevicePtr(),
                    mWeights[0],
                    0.0f,
                    mOutputs.getDevicePtr());

            for (unsigned int k = 1; k < nbInputs; ++k) {
                // mOutputs <- mWeights[k] * |mInputs[k]| + mOutputs
                std::shared_ptr<CudaDeviceTensor<Float_T> > input
                    = cuda_device_tensor_cast<Float_T>(mInputs[k]);

                cudaScaleAbs(nbElems,
                            input->getDevicePtr(),
                            mWeights[k],
                            1.0f,
                            mOutputs.getDevicePtr());
            }
        }
        else if (mCoeffMode == ElemWiseCell::PerChannel) {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame_CUDA::propagate(): for cell "
                << mName << ": the PerChannel AbsoluteSum operation is not yet supported"
                << " by N2D2";
            throw std::runtime_error(errorMsg.str());
        }
    }
    else if (mOperation == EuclideanSum) {
        if(mCoeffMode == ElemWiseCell::PerInput 
                || mCoeffMode == ElemWiseCell::PerLayer) {
            // mInterTerm <- (mWeights[0] * mInputs[0])^2
            cudaScaleSquare(nbElems,
                            input0->getDevicePtr(),
                            mWeights[0] * mWeights[0],
                            mShifts[0] * mShifts[0],
                            0.0f,
                            mInterTerm.getDevicePtr());

            for (unsigned int k = 1; k < nbInputs; ++k) {
                // mInterTerm <- (mWeights[k] * mInputs[k])^2 + mInterTerm
                std::shared_ptr<CudaDeviceTensor<Float_T> > input
                    = cuda_device_tensor_cast<Float_T>(mInputs[k]);

                cudaScaleSquare(nbElems,
                                input->getDevicePtr(),
                                mWeights[k] * mWeights[k],
                                mShifts[k] * mShifts[k],
                                1.0f,
                                mInterTerm.getDevicePtr());
            }

            // mInterTerm <- sqrt(mInterTerm)
            cudaSqrt(nbElems, mInterTerm.getDevicePtr());

            // mOutputs <- mInterTerm
            CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                            nbElems,
                                            mInterTerm.getDevicePtr(),
                                            1,
                                            mOutputs.getDevicePtr(),
                                            1));
        }
        else if (mCoeffMode == ElemWiseCell::PerChannel) {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame_CUDA::propagate(): for cell "
                << mName << ": the PerChannel EuclideanSum operation is not yet supported"
                << " by N2D2";
            throw std::runtime_error(errorMsg.str());
        }
    }
    else if (mOperation == Prod) {
        if(mCoeffMode == ElemWiseCell::PerInput 
                || mCoeffMode == ElemWiseCell::PerLayer) {
            if (nbInputs > 1) {
                // mOutputs <- mInputs[0] * mInputs[1]
                std::shared_ptr<CudaDeviceTensor<Float_T> > input1
                    = cuda_device_tensor_cast<Float_T>(mInputs[1]);

                cudaMult(nbElems,
                        input0->getDevicePtr(),
                        input1->getDevicePtr(),
                        0.0f,
                        mOutputs.getDevicePtr());

                for (unsigned int k = 2; k < nbInputs; ++k) {
                    // mOutputs <- mInputs[k] * mOutputs
                    std::shared_ptr<CudaDeviceTensor<Float_T> > input
                        = cuda_device_tensor_cast<Float_T>(mInputs[k]);

                    cudaMult(nbElems,
                            mOutputs.getDevicePtr(),
                            input->getDevicePtr(),
                            0.0f,
                            mOutputs.getDevicePtr());
                }
            }
            else {
                // mOutputs <- mInputs[0]
                CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                                nbElems,
                                                input0->getDevicePtr(),
                                                1,
                                                mOutputs.getDevicePtr(),
                                                1));
            }
        }
        else if (mCoeffMode == ElemWiseCell::PerChannel) {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame_CUDA::propagate(): for cell "
                << mName << ": the PerChannel Prod operation is not yet supported"
                << " by N2D2";
            throw std::runtime_error(errorMsg.str());
        }
    }
    else if (mOperation == Max) {
        if(mCoeffMode == ElemWiseCell::PerInput 
                || mCoeffMode == ElemWiseCell::PerLayer) {
            // mOutputs <- mInputs[0]
            CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                            nbElems,
                                            input0->getDevicePtr(),
                                            1,
                                            mOutputs.getDevicePtr(),
                                            1));
            // mArgMax <- 0
            cudaZeroInit(nbElems, mArgMax.getDevicePtr());

            for (unsigned int k = 1; k < nbInputs; ++k) {
                std::shared_ptr<CudaDeviceTensor<Float_T> > input
                    = cuda_device_tensor_cast<Float_T>(mInputs[k]);

                cudaMaxForward(nbElems,
                                input->getDevicePtr(),
                                mOutputs.getDevicePtr(),
                                k,
                                mArgMax.getDevicePtr());
            }
        }
        else if (mCoeffMode == ElemWiseCell::PerChannel) {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame_CUDA::propagate(): for cell "
                << mName << ": the PerChannel Max operation is not yet supported"
                << " by N2D2";
            throw std::runtime_error(errorMsg.str());
        }
    }
    else {
        throw std::runtime_error("ElemWiseCell_Frame_CUDA::propagate(): "
                                 "unknown operation type.");
    }

    Cell_Frame_CUDA<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::ElemWiseCell_Frame_CUDA::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    Cell_Frame_CUDA<Float_T>::backPropagate();

    for (unsigned int k = 0; k < nbInputs; ++k) {
        if (mDiffOutputs[k].empty())
            continue;

        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        std::shared_ptr<CudaDeviceTensor<Float_T> > input
            = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[k]);
        std::shared_ptr<CudaDeviceTensor<Float_T> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<Float_T>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        if (mOperation == Sum) {
            if (mDiffOutputs[k].isValid()) {
                // mDiffOutputs[k] <- mWeights[k] * mDiffInputs + mDiffOutputs[k]
                CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                                nbElems,
                                                &(mWeights[k]),
                                                mDiffInputs.getDevicePtr(),
                                                1,
                                                diffOutput->getDevicePtr(),
                                                1));
            }
            else {
                if(mCoeffMode == ElemWiseCell::PerInput 
                        || mCoeffMode == ElemWiseCell::PerLayer) {

                    if (mWeights[k] == 1.0) {
                        // mDiffOutputs[k] <- mDiffInputs
                        CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                                        nbElems,
                                                        mDiffInputs.getDevicePtr(),
                                                        1,
                                                        diffOutput->getDevicePtr(),
                                                        1));
                    }
                    else {
                        // mDiffOutputs[k] <- mWeights[k] * mDiffInputs
                        cudaScale(nbElems,
                                mDiffInputs.getDevicePtr(),
                                mWeights[k],
                                0.0f,
                                0.0f,
                                diffOutput->getDevicePtr());
                    }
                }
                else if (mCoeffMode == ElemWiseCell::PerChannel) {
                    const unsigned int nbChElems = mDiffInputs.dimY()*mDiffInputs.dimX();

                    for(unsigned int b = 0; b < mDiffInputs.dimB(); ++b) {
                        const unsigned int batchoffset 
                                    = b*mDiffInputs.dimZ()*nbChElems;
                        unsigned int chOffset = batchoffset;
                        for(unsigned int ch = 0; ch < mDiffInputs.dimZ(); ++ch) {
                            cudaScale(  nbChElems,
                                        mDiffInputs.getDevicePtr() + chOffset,
                                        mWeights[ch],
                                        0.0f,
                                        0.0f,
                                        diffOutput->getDevicePtr() + chOffset);

                            chOffset += nbChElems;
                        }
                    }
                }
            }
        }
        else if (mOperation == AbsSum) {
            // mDiffOutputs[k] <- mWeights[k] * sign(mInputs[k]) * mDiffInputs
            //                      + beta * mDiffOutputs[k]
            cudaScaleSign(nbElems,
                           mDiffInputs.getDevicePtr(),
                           input->getDevicePtr(),
                           mWeights[k],
                           beta,
                           diffOutput->getDevicePtr());
        }
        else if (mOperation == EuclideanSum) {
            // mDiffOutputs[k] <- (mWeights[k] * mWeights[k])
            //                      * (mInputs[k] / mInterTerm) * mDiffInputs
            //                      + beta * mDiffOutputs[k]
            cudaEuclideanSumBackward(nbElems,
                                      mDiffInputs.getDevicePtr(),
                                      input->getDevicePtr(),
                                      mInterTerm.getDevicePtr(),
                                      mWeights[k] * mWeights[k],
                                      beta,
                                      diffOutput->getDevicePtr());
        }
        else if (mOperation == Prod) {
            bool init = false;

            for (unsigned int i = 0; i < nbInputs; ++i) {
                if (i == k)
                    continue;

                std::shared_ptr<CudaDeviceTensor<Float_T> > input_i
                    = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[i]);

                if (!init) {
                    // mInterTerm <- mInputs[i]
                    CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                                    nbElems,
                                                    input_i->getDevicePtr(),
                                                    1,
                                                    mInterTerm.getDevicePtr(),
                                                    1));
                    init = true;
                }
                else {
                    // mInterTerm <- mInputs[i] * mInterTerm
                    cudaMult(nbElems,
                              mInterTerm.getDevicePtr(),
                              input_i->getDevicePtr(),
                              0.0f,
                              mInterTerm.getDevicePtr());

                }
            }

            // mDiffOutputs[k] <- mDiffInputs * mInterTerm
            //                      + beta * mDiffOutputs[k]
            cudaMult(nbElems,
                      mInterTerm.getDevicePtr(),
                      mDiffInputs.getDevicePtr(),
                      beta,
                      diffOutput->getDevicePtr());
        }
        else if (mOperation == Max) {
            cudaMaxBackward(nbElems,
                             mDiffInputs.getDevicePtr(),
                             k,
                             mArgMax.getDevicePtr(),
                             beta,
                             diffOutput->getDevicePtr());
        }
        else {
            throw std::runtime_error("ElemWiseCell_Frame_CUDA::propagate(): "
                                     "unknown operation type.");
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::ElemWiseCell_Frame_CUDA::update()
{
    Cell_Frame_CUDA<float>::update();
}

void N2D2::ElemWiseCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ElemWiseCell_Frame_CUDA::propagate, this, false),
                  std::bind(&ElemWiseCell_Frame_CUDA::backPropagate, this),
                  (mOperation == Max));

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
    }
}

std::pair<double, double> N2D2::ElemWiseCell_Frame_CUDA::getOutputsRange() const {
    const auto& activation = Cell_Frame_CUDA<Float_T>::getActivation();
    return activation?activation->getOutputRange():ElemWiseCell::getOutputsRange();
}

N2D2::ElemWiseCell_Frame_CUDA::~ElemWiseCell_Frame_CUDA()
{

}

#endif
