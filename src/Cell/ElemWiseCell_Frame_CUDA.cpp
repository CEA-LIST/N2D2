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

#include "Cell/ElemWiseCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::ElemWiseCell>
N2D2::ElemWiseCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                      N2D2::ElemWiseCell_Frame_CUDA::create);

N2D2::ElemWiseCell_Frame_CUDA::ElemWiseCell_Frame_CUDA(
    const std::string& name,
    unsigned int nbOutputs,
    Operation operation,
    const std::vector<Float_T>& weights,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      ElemWiseCell(name,
               nbOutputs,
               operation,
               weights),
      Cell_Frame_CUDA(name, nbOutputs, activation)
{
    // ctor
}

void N2D2::ElemWiseCell_Frame_CUDA::initialize()
{
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].dimZ() != mOutputs.dimZ()
            || mInputs[k].dimB() != mOutputs.dimB())
        {
            throw std::runtime_error("ElemWiseCell_Frame_CUDA::initialize(): "
                                     "the input tensors dimensions must match "
                                     "the output dimensions.");
        }
    }

    mWeights.resize(mInputs.size(), 1.0);

    if (mOperation == Max) {
        mArgMax.resize(mOutputs.dimX(),
                       mOutputs.dimY(),
                       mOutputs.dimZ(),
                       mOutputs.dimB());
    }

    if (mOperation == EuclideanSum || mOperation == Prod) {
        mInterTerm.resize(mOutputs.dimX(),
                          mOutputs.dimY(),
                          mOutputs.dimZ(),
                          mOutputs.dimB());
    }
}

void N2D2::ElemWiseCell_Frame_CUDA::propagate(bool /*inference*/)
{
    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    mInputs.synchronizeHBasedToD();

    if (mOperation == Sum) {
        // mOutputs <- mWeights[0] * mInputs[0]
        cudaSScale(nbElems,
                   mInputs[0].getDevicePtr(),
                   mWeights[0],
                   0.0f,
                   mOutputs.getDevicePtr());

        for (unsigned int k = 1; k < nbInputs; ++k) {
            // mOutputs <- mWeights[k] * mInputs[k] + mOutputs
            CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                            nbElems,
                                            &(mWeights[k]),
                                            mInputs[k].getDevicePtr(),
                                            1,
                                            mOutputs.getDevicePtr(),
                                            1));
        }
    }
    else if (mOperation == AbsSum) {
        // mOutputs <- mWeights[0] * |mInputs[0]|
        cudaSScaleAbs(nbElems,
                   mInputs[0].getDevicePtr(),
                   mWeights[0],
                   0.0f,
                   mOutputs.getDevicePtr());

        for (unsigned int k = 1; k < nbInputs; ++k) {
            // mOutputs <- mWeights[k] * |mInputs[k]| + mOutputs
            cudaSScaleAbs(nbElems,
                          mInputs[k].getDevicePtr(),
                          mWeights[k],
                          1.0f,
                          mOutputs.getDevicePtr());
        }
    }
    else if (mOperation == EuclideanSum) {
        // mInterTerm <- (mWeights[0] * mInputs[0])^2
        cudaSScaleSquare(nbElems,
                         mInputs[0].getDevicePtr(),
                         mWeights[0] * mWeights[0],
                         0.0f,
                         mInterTerm.getDevicePtr());

        for (unsigned int k = 1; k < nbInputs; ++k) {
            // mInterTerm <- (mWeights[k] * mInputs[k])^2 + mInterTerm
            cudaSScaleSquare(nbElems,
                             mInputs[k].getDevicePtr(),
                             mWeights[k] * mWeights[k],
                             1.0f,
                             mInterTerm.getDevicePtr());
        }

        // mInterTerm <- sqrt(mInterTerm)
        cudaSSqrt(nbElems, mInterTerm.getDevicePtr());

        // mOutputs <- mInterTerm
        CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                        nbElems,
                                        mInterTerm.getDevicePtr(),
                                        1,
                                        mOutputs.getDevicePtr(),
                                        1));
    }
    else if (mOperation == Prod) {
        if (nbInputs > 1) {
            // mOutputs <- mInputs[0] * mInputs[1]
            cudaSMult(nbElems,
                      mInputs[0].getDevicePtr(),
                      mInputs[1].getDevicePtr(),
                      0.0f,
                      mOutputs.getDevicePtr());

            for (unsigned int k = 2; k < nbInputs; ++k) {
                // mOutputs <- mInputs[k] * mOutputs
                cudaSMult(nbElems,
                          mOutputs.getDevicePtr(),
                          mInputs[k].getDevicePtr(),
                          0.0f,
                          mOutputs.getDevicePtr());
            }
        }
        else {
            // mOutputs <- mInputs[0]
            CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                            nbElems,
                                            mInputs[0].getDevicePtr(),
                                            1,
                                            mOutputs.getDevicePtr(),
                                            1));
        }
    }
    else if (mOperation == Max) {
        // mOutputs <- mInputs[0]
        CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                        nbElems,
                                        mInputs[0].getDevicePtr(),
                                        1,
                                        mOutputs.getDevicePtr(),
                                        1));
        // mArgMax <- 0
        cudaUZeroInit(nbElems, mArgMax.getDevicePtr());

        for (unsigned int k = 1; k < nbInputs; ++k) {
            cudaSMaxForward(nbElems,
                            mInputs[k].getDevicePtr(),
                            mOutputs.getDevicePtr(),
                            k,
                            mArgMax.getDevicePtr());
        }
    }
    else {
        throw std::runtime_error("ElemWiseCell_Frame_CUDA::propagate(): "
                                 "unknown operation type.");
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::ElemWiseCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    Cell_Frame_CUDA::backPropagate();

    for (unsigned int k = 0; k < nbInputs; ++k) {
        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        if (mOperation == Sum) {
            if (mDiffOutputs[k].isValid()) {
                // mDiffOutputs[k] <- mWeights[k] * mDiffInputs + mDiffOutputs[k]
                CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                                nbElems,
                                                &(mWeights[k]),
                                                mDiffInputs.getDevicePtr(),
                                                1,
                                                mDiffOutputs[k].getDevicePtr(),
                                                1));
            }
            else if (mWeights[k] == 1.0) {
                // mDiffOutputs[k] <- mDiffInputs
                CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                                nbElems,
                                                mDiffInputs.getDevicePtr(),
                                                1,
                                                mDiffOutputs[k].getDevicePtr(),
                                                1));
            }
            else {
                // mDiffOutputs[k] <- mWeights[k] * mDiffInputs
                cudaSScale(nbElems,
                           mDiffInputs.getDevicePtr(),
                           mWeights[k],
                           0.0f,
                           mDiffOutputs[k].getDevicePtr());
            }
        }
        else if (mOperation == AbsSum) {
            // mDiffOutputs[k] <- mWeights[k] * sign(mInputs[k]) * mDiffInputs
            //                      + beta * mDiffOutputs[k]
            cudaSScaleSign(nbElems,
                           mDiffInputs.getDevicePtr(),
                           mInputs[k].getDevicePtr(),
                           mWeights[k],
                           beta,
                           mDiffOutputs[k].getDevicePtr());
        }
        else if (mOperation == EuclideanSum) {
            // mDiffOutputs[k] <- (mWeights[k] * mWeights[k])
            //                      * (mInputs[k] / mInterTerm) * mDiffInputs
            //                      + beta * mDiffOutputs[k]
            cudaSEuclideanSumBackward(nbElems,
                                      mDiffInputs.getDevicePtr(),
                                      mInputs[k].getDevicePtr(),
                                      mInterTerm.getDevicePtr(),
                                      mWeights[k] * mWeights[k],
                                      beta,
                                      mDiffOutputs[k].getDevicePtr());
        }
        else if (mOperation == Prod) {
            bool init = false;

            for (unsigned int i = 0; i < nbInputs; ++i) {
                if (i == k)
                    continue;

                if (!init) {
                    // mInterTerm <- mInputs[i]
                    CHECK_CUBLAS_STATUS(cublasScopy(CudaContext::cublasHandle(),
                                                    nbElems,
                                                    mInputs[i].getDevicePtr(),
                                                    1,
                                                    mInterTerm.getDevicePtr(),
                                                    1));
                    init = true;
                }
                else {
                    // mInterTerm <- mInputs[i] * mInterTerm
                    cudaSMult(nbElems,
                              mInterTerm.getDevicePtr(),
                              mInputs[i].getDevicePtr(),
                              0.0f,
                              mInterTerm.getDevicePtr());

                }
            }

            // mDiffOutputs[k] <- mDiffInputs * mInterTerm
            //                      + beta * mDiffOutputs[k]
            cudaSMult(nbElems,
                      mInterTerm.getDevicePtr(),
                      mDiffInputs.getDevicePtr(),
                      beta,
                      mDiffOutputs[k].getDevicePtr());
        }
        else if (mOperation == Max) {
            cudaSMaxBackward(nbElems,
                             mDiffInputs.getDevicePtr(),
                             k,
                             mArgMax.getDevicePtr(),
                             beta,
                             mDiffOutputs[k].getDevicePtr());
        }
        else {
            throw std::runtime_error("ElemWiseCell_Frame_CUDA::propagate(): "
                                     "unknown operation type.");
        }

        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::ElemWiseCell_Frame_CUDA::update()
{
}

void N2D2::ElemWiseCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ElemWiseCell_Frame_CUDA::propagate, this, false),
                  std::bind(&ElemWiseCell_Frame_CUDA::backPropagate, this),
                  (mOperation == Max));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

N2D2::ElemWiseCell_Frame_CUDA::~ElemWiseCell_Frame_CUDA()
{

}

#endif
