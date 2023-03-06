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

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell_Frame.hpp"
#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_Kernels.hpp"

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<float>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::PruneQuantizerCell>
N2D2::PruneQuantizerCell_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::PruneQuantizerCell_Frame<double>::create,
    N2D2::Registrar<N2D2::PruneQuantizerCell>::Type<double>());


namespace N2D2 {

template<class T>
PruneQuantizerCell_Frame<T>::PruneQuantizerCell_Frame()
    : PruneQuantizerCell(),
      QuantizerCell_Frame<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<T> >();
}


template<class T>
void PruneQuantizerCell_Frame<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new Tensor<T>(weights.dims()));
    mMasksWeights.push_back(new Tensor<unsigned int>(weights.dims(), 1U));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new Tensor<T>(diffWeights.dims()));
}

template<class T>
void PruneQuantizerCell_Frame<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<BaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<BaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}


template<typename T>
void PruneQuantizerCell_Frame<T>::initialize()
{
    // Initialize masks if PruningFiller not None
    if (mPruningFiller != None) {
        for (unsigned int k = 0, size = mMasksWeights.size(); k < size; ++k) {
            Tensor<T> weights = tensor_cast<T>(mFullPrecisionWeights[k]);
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

            mCurrentThreshold = 0.0f;

            switch (getPruningMode()) {
            case Identity:
                // Nothing to do
                break;
            case Static:
            {
                mCurrentThreshold = mThreshold;
                break;
            }
            case Gradual:
            {
                mCurrentThreshold = mStartThreshold;
                break;
            }
            default:
                // Should never be here
                break;
            }

            if (mPruningFiller == Random) {
                PruneQuantizer_Frame_Kernels::update_masks_random(mask, mCurrentThreshold);
            } else if (mPruningFiller == IterNonStruct) {
                PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct(weights, mask, mCurrentThreshold, mDelta);
            }
            mask.synchronizeHToD();
        }
    }

    mInitialized = true;
}


template<typename T>
void PruneQuantizerCell_Frame<T>::propagate()
{
    switch (getPruningMode()) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
            Tensor<T> fpW = tensor_cast<T>(mFullPrecisionWeights[k]);
            Tensor<T> pW = tensor_cast<T>(mQuantizedWeights[k]);
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

            PruneQuantizer_Frame_Kernels::apply_pruning_with_masks(fpW, pW, mask);
        }
        break;
    }
    default:
        // Should never be here
        break;
    }

    if (mFullPrecisionBiases) {
        Tensor<T>& fullPrecBiases = dynamic_cast<Tensor<T>&>(*mFullPrecisionBiases);
        for (unsigned int i = 0; i < fullPrecBiases.size(); ++i) {
            mQuantizedBiases(i) = fullPrecBiases(i);
        }
    }
}


template<typename T>
void PruneQuantizerCell_Frame<T>::back_propagate()
{
    switch (getPruningMode()) {
    case Identity:
    case Static:
    case Gradual:
    {
        for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
            Tensor<T> diffW = tensor_cast<T>(mDiffFullPrecisionWeights[k]);
            Tensor<T> diffPrunedW = tensor_cast<T>(mDiffQuantizedWeights[k]);
            Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

            PruneQuantizer_Frame_Kernels::apply_pruning_with_masks(diffPrunedW, diffW, mask);
        }
        break;
    }
    default:
        // Should never be here
        break;
    }

    if (mDiffQuantizedBiases) {
        Tensor<T>& diffQuantBiases = dynamic_cast<Tensor<T>&>(*mDiffQuantizedBiases);
        for (unsigned int i = 0; i < diffQuantBiases.size(); ++i) {
            mDiffFullPrecisionBiases(i) = diffQuantBiases(i);
        }
    }
}


template<typename T>
void PruneQuantizerCell_Frame<T>::update(unsigned int batchSize)
{
    if (getPruningMode() == PruningMode::Gradual) {
        if (!mScheduler) {
            unsigned int stepSize = (mStepSizeThreshold > 0) ? mStepSizeThreshold : SGDSolver::mLogSteps;
            mScheduler = std::make_shared<Scheduler>(stepSize, batchSize);
        }
        if(mScheduler->step()) {
            if (mCurrentThreshold < mThreshold) {
                std::cout << "\nUpdate masks" << std::endl;

                // Update threshold
                mCurrentThreshold += mGammaThreshold;

                float maxthreshold = mThreshold;
                mCurrentThreshold = std::min(mCurrentThreshold, maxthreshold);

                for (unsigned int k = 0, size = mMasksWeights.size(); k < size; ++k) {
                    Tensor<T> weights = tensor_cast<T>(mFullPrecisionWeights[k]);
                    Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);

                    if (mPruningFiller == Random) {
                        PruneQuantizer_Frame_Kernels::update_masks_random(mask, mCurrentThreshold);
                    } else if (mPruningFiller == IterNonStruct) {
                        PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct(weights, mask, mCurrentThreshold, mDelta);
                    }

                    mask.synchronizeHToD();
                }
            }
        }
    }
}


template <class T>
PruneQuantizerCell_Frame<T>::~PruneQuantizerCell_Frame()
{
    // dtor
}


template <class T>
void PruneQuantizerCell_Frame<T>::exportFreeParameters(const std::string& fileName) const 
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string masksWFile = fileBase + "_masks" + fileExt;
    std::ofstream masksW(masksWFile.c_str());

    if (!masksW.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + masksWFile);

    for (unsigned int k = 0; k < mMasksWeights.size(); ++k) {
        Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[k]);
        for (unsigned int output = 0; output < mask.dimB(); ++output) {

            for (unsigned int i = 0; i < mask[output].size(); ++i) {
                masksW << mask[output](i) << " ";
            }
            masksW << "\n";
        }
    }
}

template <class T>
void PruneQuantizerCell_Frame<T>::importFreeParameters(const std::string& fileName, 
                                                       bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string masksWFile = fileBase + "_masks" + fileExt;
    std::ifstream masksW(masksWFile.c_str());

    if (!masksW.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << masksWFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + masksWFile);
    }

    std::vector<unsigned int> out;
    std::copy(std::istream_iterator<unsigned int>(masksW), 
              std::istream_iterator<unsigned int>(), 
              std::back_inserter(out));

    Tensor<unsigned int> mask = tensor_cast<unsigned int>(mMasksWeights[0]);
    for (unsigned int i = 0; i < mask.size(); ++i) {
        mask(i) = out[i];
    }

    mCurrentThreshold = (float)((mask.eq(0)).sum()) / mask.size();
}

}
