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

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"

/* Only float functions for now */

template<>
N2D2::Registrar<N2D2::LSQQuantizerCell>
N2D2::LSQQuantizerCell_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerCell_Frame<float>::create,
    N2D2::Registrar<N2D2::LSQQuantizerCell>::Type<float>());


namespace N2D2 {

template<>
LSQQuantizerCell_Frame<float>::LSQQuantizerCell_Frame()
    : LSQQuantizerCell(),
      QuantizerCell_Frame()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<float> >();
}


template<class T>
void LSQQuantizerCell_Frame<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new Tensor<T>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new Tensor<T>(diffWeights.dims()));

    mDiffStepSizeInterface.push_back(new Tensor<T>(diffWeights.dims()));
    mDiffStepSizeInterface.back().fill(0.0);

    //mDiffStepSizeTensor.resize(diffWeights.dims(), T(0.0));
}

template<class T>
void LSQQuantizerCell_Frame<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<BaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<BaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}


template<class T>
void LSQQuantizerCell_Frame<T>::initialize()
{
    // nothing for now
}


// ----------------------------------------------------------------------------
// ---------------------------- Forward functions -----------------------------
// ----------------------------------------------------------------------------

template<>
void LSQQuantizerCell_Frame<float>::propagate()
{
    // nothing for now
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

template<>
void LSQQuantizerCell_Frame<float>::back_propagate()
{
    // nothing for now
}


template<>
void LSQQuantizerCell_Frame<float>::update(unsigned int batchSize)
{
    mSolver->update(mStepSize, mDiffStepSize, batchSize);
}


template <class T>
LSQQuantizerCell_Frame<T>::~LSQQuantizerCell_Frame()
{
    // dtor
}


template <class T>
void LSQQuantizerCell_Frame<T>::exportFreeParameters(const std::string& fileName) const 
{
    mStepSize.synchronizeDToH();

    const std::string dirName = Utils::dirName(fileName);
    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string alphasWFile = fileBase + "_alphaW_LSQ" + fileExt;
    std::ofstream alphasW(alphasWFile.c_str());

    if (!alphasW.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + alphasWFile);
                                 
    alphasW << mStepSize(0) << " ";
    mStepSize.synchronizeHToD();
}

template <class T>
void LSQQuantizerCell_Frame<T>::importFreeParameters(const std::string& fileName, 
                                                     bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string alphasWFile = fileBase + "_alphaW_LSQ" + fileExt;
    std::ifstream alphasW(alphasWFile.c_str());

    if (!alphasW.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << alphasWFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + alphasWFile);
    }


    T valueW;
    if (!(alphasW >> valueW))
        throw std::runtime_error( "Error while reading synaptic file: "
                        + alphasWFile);

    mStepSize.resize(mStepSize.dims(), T(valueW));
    
    // Discard trailing whitespaces
    while (std::isspace(alphasW.peek()))
        alphasW.ignore();

    if (alphasW.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + alphasWFile);

    mStepSize.synchronizeHToD();

}

}
