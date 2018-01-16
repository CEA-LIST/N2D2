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

#include "Cell/BatchNormCell_Frame.hpp"

N2D2::Registrar<N2D2::BatchNormCell>
N2D2::BatchNormCell_Frame::mRegistrar("Frame",
                                      N2D2::BatchNormCell_Frame::create);

N2D2::BatchNormCell_Frame::BatchNormCell_Frame(
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      BatchNormCell(name, nbOutputs),
      Cell_Frame(name, nbOutputs, activation),
      mScale(std::make_shared<Tensor4d<Float_T> >()),
      mBias(std::make_shared<Tensor4d<Float_T> >()),
      mMean(std::make_shared<Tensor4d<Float_T> >()),
      mVariance(std::make_shared<Tensor4d<Float_T> >())
{
    // ctor
    mScaleSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame<Float_T> >();
}

void N2D2::BatchNormCell_Frame::initialize()
{
    mInputs.synchronizeDToH();

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("BatchNormCell_Frame::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    mNbPropagate = 0;

    if (mEpsilon == 0.0)
        mEpsilon = 1.0e-5; // Same as CUDNN_BN_MIN_EPSILON

    Tensor4d<Float_T>* input = *mInputs.begin();

    if (mScale->empty())
        mScale->resize(1, 1, input->dimZ(), 1, 1.0);
    else {
        if (mScale->dimX() != 1 || mScale->dimY() != 1
            || mScale->dimZ() != input->dimZ() || mScale->dimB() != 1)
        {
            throw std::runtime_error("BatchNormCell_Frame::initialize(): in "
                "cell " + mName + ", wrong size for shared scale");
        }
    }

    if (mBias->empty())
        mBias->resize(1, 1, input->dimZ(), 1, 0.0);
    else {
        if (mBias->dimX() != 1 || mBias->dimY() != 1
            || mBias->dimZ() != input->dimZ() || mBias->dimB() != 1)
        {
            throw std::runtime_error("BatchNormCell_Frame::initialize(): in "
                "cell " + mName + ", wrong size for shared bias");
        }
    }

    if (mMean->empty())
        mMean->resize(1, 1, input->dimZ(), 1, 0.0);
    else {
        if (mMean->dimX() != 1 || mMean->dimY() != 1
            || mMean->dimZ() != input->dimZ() || mMean->dimB() != 1)
        {
            throw std::runtime_error("BatchNormCell_Frame::initialize(): in "
                "cell " + mName + ", wrong size for shared mean");
        }
    }

    if (mVariance->empty())
        mVariance->resize(1, 1, input->dimZ(), 1, 0.0);
    else {
        if (mVariance->dimX() != 1 || mVariance->dimY() != 1
            || mVariance->dimZ() != input->dimZ() || mVariance->dimB() != 1)
        {
            throw std::runtime_error("BatchNormCell_Frame::initialize(): in "
                "cell " + mName + ", wrong size for shared variance");
        }
    }

    mSavedMean.resize(1, 1, input->dimZ(), 1);
    mSavedVariance.resize(1, 1, input->dimZ(), 1);

    mDiffScale.resize(1, 1, input->dimZ(), 1);
    mDiffBias.resize(1, 1, input->dimZ(), 1);
    mDiffSavedMean.resize(1, 1, input->dimZ(), 1);
    mDiffSavedVariance.resize(1, 1, input->dimZ(), 1);
}

void N2D2::BatchNormCell_Frame::propagate(bool inference)
{
    const unsigned int size = mInputs.dimB() * mNbOutputs;

    if (inference) {
#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                const Float_T var = std::sqrt((*mVariance)(output) + mEpsilon);

                for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                        const Float_T normalized
                            = (mInputs(ox, oy, output, batchPos)
                               - (*mMean)(output)) / var;
                        mOutputs(ox, oy, output, batchPos)
                            = (*mScale)(output) * normalized + (*mBias)(output);
                    }
                }
            }
        }
    } else {
        const unsigned int size = mInputs[0].dimX() * mInputs[0].dimY()
                                  * mInputs.dimB();
        // Cumulative Moving Average (CMA)
        const double expAverageFactor = 1.0 / (1.0 + mNbPropagate);

#pragma omp parallel for if (mNbOutputs > 16)
        for (int output = 0; output < (int)mNbOutputs; ++output) {
            Float_T sum = 0.0;

            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox)
                        sum += mInputs(ox, oy, output, batchPos);
                }
            }

            mSavedMean(output) = sum / (Float_T)size;

            sum = 0.0;

            for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
                for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                        const Float_T zeroed = mInputs(ox, oy, output, batchPos)
                                               - mSavedMean(output);
                        sum += zeroed * zeroed;
                    }
                }
            }

            mSavedVariance(output) = sum / (Float_T)size;

            (*mMean)(output) = mSavedMean(output) * expAverageFactor
                            + (*mMean)(output) * (1.0 - expAverageFactor);
            (*mVariance)(output) = mSavedVariance(output) * expAverageFactor
                                + (*mVariance)(output) * (1.0 - expAverageFactor);
        }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                const Float_T var
                    = std::sqrt(mSavedVariance(output) + mEpsilon);

                for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                        const Float_T normalized
                            = (mInputs(ox, oy, output, batchPos)
                               - mSavedMean(output)) / var;
                        mOutputs(ox, oy, output, batchPos)
                            = (*mScale)(output) * normalized + (*mBias)(output);
                    }
                }
            }
        }

        ++mNbPropagate;
    }

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::BatchNormCell_Frame::backPropagate()
{
    Cell_Frame::backPropagate();

    const unsigned int size = mInputs[0].dimX() * mInputs[0].dimY()
                              * mInputs.dimB();
    const float betaScale = (mScaleSolver->isNewIteration()) ? 0.0f : 1.0f;
    const float betaBias = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

#pragma omp parallel for if (mNbOutputs > 16)
    for (int output = 0; output < (int)mNbOutputs; ++output) {
        const Float_T var = std::sqrt(mSavedVariance(output) + mEpsilon);

        Float_T sumScale = 0.0;
        Float_T sumBias = 0.0;
        Float_T sumMean1 = 0.0;
        Float_T sumMean2 = 0.0;
        Float_T sumVariance = 0.0;

        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                    const Float_T normalized
                        = (mInputs(ox, oy, output, batchPos)
                           - mSavedMean(output)) / var;
                    const Float_T diffNormalized
                        = mDiffInputs(ox, oy, output, batchPos)
                          * (*mScale)(output);

                    sumScale += mDiffInputs(ox, oy, output, batchPos)
                                * normalized;
                    sumBias += mDiffInputs(ox, oy, output, batchPos);

                    sumMean1 += diffNormalized;
                    sumMean2 += -2.0 * (mInputs(ox, oy, output, batchPos)
                                        - mSavedMean(output));
                    sumVariance += diffNormalized
                                   * (mInputs(ox, oy, output, batchPos)
                                      - mSavedMean(output));
                }
            }
        }

        mDiffSavedVariance(output)
            = sumVariance * (-1.0 / 2.0)
              * std::pow(mSavedVariance(output) + mEpsilon, -3.0 / 2.0);
        mDiffSavedMean(output) = sumMean1 * (-1.0 / var)
                                 + mDiffSavedVariance(output) * sumMean2
                                   / (Float_T)size;

        mDiffScale(output) = sumScale + betaScale * mDiffScale(output);
        mDiffBias(output) = sumBias + betaBias * mDiffBias(output);
    }

    if (!mDiffOutputs.empty()) {
        const unsigned int size = mInputs.dimB() * mNbOutputs;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                const bool isValid = mDiffOutputs.getTensor4d(output).isValid();
                const Float_T var
                    = std::sqrt(mSavedVariance(output) + mEpsilon);

                for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                        const Float_T diffNormalized
                            = mDiffInputs(ox, oy, output, batchPos)
                              * (*mScale)(output);
                        const Float_T gradient
                            = diffNormalized / var
                              + mDiffSavedVariance(output) * 2.0
                                * (mInputs(ox, oy, output, batchPos)
                                   - mSavedMean(output)) / (Float_T)size
                              + mDiffSavedMean(output) / (Float_T)size;

                        mDiffOutputs(ox, oy, output, batchPos)
                            = gradient
                              + isValid
                                * mDiffOutputs(ox, oy, output, batchPos);
                    }
                }
            }
        }

        mDiffOutputs.setValid();
        mDiffOutputs.synchronizeHToD();
    }
}

void N2D2::BatchNormCell_Frame::update()
{
    assert(mScale->size() == mDiffScale.size());
    assert(mBias->size() == mDiffBias.size());
    assert(mScale->size() == mBias->size());

    mScaleSolver->update(&(*mScale), &mDiffScale, mInputs.dimB());
    mBiasSolver->update(&(*mBias), &mDiffBias, mInputs.dimB());
}

void N2D2::BatchNormCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&BatchNormCell_Frame::propagate, this, false),
                  std::bind(&BatchNormCell_Frame::backPropagate, this));
    gc.check(mName + "_mDiffSavedMean", mSavedMean, mDiffSavedMean);
    gc.check(mName + "_mDiffSavedVariance", mSavedVariance, mDiffSavedVariance);
    gc.check(mName + "_mDiffScale", (*mScale), mDiffScale);
    gc.check(mName + "_mDiffBias", (*mBias), mDiffBias);

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

void N2D2::BatchNormCell_Frame::saveFreeParameters(const std::string
                                                   & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create parameter file (.SYN): "
                                 + fileName);

    for (std::vector<Float_T>::const_iterator it = mScale->begin();
         it != mScale->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::const_iterator it = mBias->begin();
         it != mBias->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::const_iterator it = mMean->begin();
         it != mMean->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::const_iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    if (!syn.good())
        throw std::runtime_error("Error writing parameter file: " + fileName);
}

void N2D2::BatchNormCell_Frame::loadFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open parameter file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open parameter file (.SYN): "
                                     + fileName);
    }

    for (std::vector<Float_T>::iterator it = mScale->begin(); it != mScale->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::iterator it = mBias->begin(); it != mBias->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::iterator it = mMean->begin(); it != mMean->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    for (std::vector<Float_T>::iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in parameter file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading parameter file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

N2D2::BatchNormCell_Frame::~BatchNormCell_Frame()
{
}
