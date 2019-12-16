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

#include "Cell/Cell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <class T>
N2D2::Cell_Frame_CUDA<T>::Cell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                       unsigned int nbOutputs,
                                       const std::shared_ptr
                                       <Activation>& activation)
    : Cell(deepNet, name, nbOutputs), Cell_Frame_Top(activation)
{
    // ctor
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::save(const std::string& dirName) const
{
    Cell::save(dirName);
    Cell_Frame_Top::save(dirName);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::load(const std::string& dirName)
{
    Cell::load(dirName);
    Cell_Frame_Top::load(dirName);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(StimuliProvider& /*sp*/,
                                     unsigned int /*channel*/,
                                     unsigned int /*x0*/,
                                     unsigned int /*y0*/,
                                     unsigned int /*width*/,
                                     unsigned int /*height*/,
                                     const Tensor<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a single "
                             "environment channel as input is not supported");
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(StimuliProvider& sp,
                                     unsigned int x0,
                                     unsigned int y0,
                                     unsigned int width,
                                     unsigned int height,
                                     const Tensor<bool>& mapping)
{
    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (x0 > 0 || y0 > 0 || width < sp.getSizeX() || height < sp.getSizeY())
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a "
                                 "cropped environment channel map as input is "
                                 "not supported");

    // Define input-output sizes
    setInputsDims(sp.getSize());
    mInputs.push_back(&sp.getData());

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(sp.getBatchSize());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    // Define input-output connections
    if (!mapping.empty() && mapping.dimY() != sp.getNbChannels()) {
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input"
                                 " channels");
    }

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), sp.getNbChannels()}, true));
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(Cell* cell, const Tensor<bool>& mapping)
{
    // Define input-output sizes
    setInputsDims(cell->getOutputsDims());

    Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(cell);

    if (cellFrame != NULL) {
        mInputs.push_back(&cellFrame->getOutputs());
        mDiffOutputs.push_back(&cellFrame->getDiffInputs());
    }
    else {
        throw std::runtime_error(
            "Cell_Frame::addInput(): cannot mix Spike and Frame models");
    }

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    // Define input-output connections
    const unsigned int cellNbOutputs = cell->getNbOutputs();

    if (!mapping.empty() && mapping.dimY() != cellNbOutputs)
        throw std::runtime_error("Cell_Frame::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), cellNbOutputs}, true));
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(Cell* cell,
                                     unsigned int x0,
                                     unsigned int y0,
                                     unsigned int width,
                                     unsigned int height)
{
    if (width == 0)
        width = cell->getOutputsWidth() - x0;
    if (height == 0)
        height = cell->getOutputsHeight() - y0;

    if (x0 > 0 || y0 > 0 || width < cell->getOutputsWidth()
        || height < cell->getOutputsHeight())
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a "
                                 "cropped output map as input is not "
                                 "supported");

    Cell_Frame_CUDA<T>::addInput(cell);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(BaseTensor& inputs,
                                     BaseTensor& diffOutputs)
{
    // Define input-output sizes
    std::vector<size_t> inputsDims = inputs.dims();
    inputsDims.pop_back();      // Remove batch

    setInputsDims(inputsDims);
    mInputs.push_back(&inputs);

    if (!diffOutputs.empty())
        mDiffOutputs.push_back(&diffOutputs);

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    mMapping.resize({getNbOutputs(), getNbChannels()}, true);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::clearInputs() {
    mInputs.clear();
    mDiffOutputs.clear();

    mInputsDims.clear();
    mMapping.clear();
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::replaceInput(BaseTensor& oldInputs,
                                            BaseTensor& newInputs,
                                            BaseTensor& newDiffOutputs)
{
    bool foundOldInputs = false;
    for (unsigned int i = 0; i < mInputs.size(); ++i) {
        if (&mInputs[i] == &oldInputs) {
            foundOldInputs = true;
            mInputs.replace(i, &newInputs);

            if (!newDiffOutputs.empty()) {
                assert(i < mDiffOutputs.size());
                mDiffOutputs.replace(i, &newDiffOutputs);
            }
        }
    }

    if(!foundOldInputs) {
        throw std::runtime_error("Can't replace input, the input has not been found.");
    }
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::propagate(bool inference)
{
    if (mActivation)
        mActivation->propagate(*this, mOutputs, inference);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::backPropagate()
{
    if (mActivation)
        mActivation->backPropagate(*this, mOutputs, mDiffInputs);
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::setOutputTarget(const Tensor<int>& targets,
                                                 double targetVal,
                                                 double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTarget(): target "
                                "and output batch sizes don't match.");

    if (targets.size() / targets.dimB() != 1)
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTarget(): require "
                                "one target per batch.");

    mOutputs.synchronizeDToH();

    const unsigned int outputSize = mOutputs.size() / mOutputs.dimB();

    double loss = 0.0;

    for (unsigned int batchPos = 0; batchPos < mOutputs.dimB(); ++batchPos) {
        if (targets(0, batchPos) >= 0) {
            if ((outputSize > 1 && targets(0, batchPos) >= (int)outputSize)
                || (outputSize == 1 && (targets(0, batchPos) < 0
                                        || targets(0, batchPos) > 1))) {
                throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTarget(): "
                                        "target not within output range.");
            }
        }

        for (unsigned int index = 0; index < outputSize; ++index) {
            if (targets(0, batchPos) >= 0) {
                const double error
                    = ((outputSize > 1 && (int)index == targets(0, batchPos))
                       || (outputSize == 1 && targets(0, batchPos) == 1))
                          ? targetVal - mOutputs(index, batchPos)
                          : defaultVal - mOutputs(index, batchPos);

                mDiffInputs(index, batchPos) = error;
                loss += error * error;
            } else
                mDiffInputs(index, batchPos) = 0.0;
        }
    }

    mDiffInputs.synchronizeHToD();

    return (loss / mOutputs.dimB());
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::setOutputTargets(const Tensor<int>& targets,
                                                  double targetVal,
                                                  double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (targets.dimX() != mOutputsDims[0] || targets.dimY() != mOutputsDims[1])
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputTargets(): wrong target "
            "matrix size. Expected " << mOutputsDims << ", got "
            << targets.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    mTargets.resize(targets.dims());
    mTargets = targets;  // TODO: could be optimized by avoiding the copy
    mTargets.synchronizeHToD();

    mNbTargetOutputs.resize(
        {(getNbOutputs() > 1) ? getNbOutputs() : 2, mOutputs.dimB()}, 0U);

    cudaPopulateNbTargetOutputs(CudaContext::getDeviceProp(),
                                mTargets.getDevicePtr(),
                                mNbTargetOutputs.getDevicePtr(),
                                getNbOutputs(),
                                mTargets.dimY(),
                                mTargets.dimX(),
                                mTargets.dimB());

    mLossMem.resize(mOutputs.dims());

    double loss = setOutputTargetsInternal(targetVal, defaultVal);

    return (loss / mOutputs.dimB());
}

namespace N2D2 {
template <>
double Cell_Frame_CUDA<half_float::half>::setOutputTargetsInternal(
    double targetVal,
    double defaultVal)
{
    return cudaHSetOutputTargets(CudaContext::getDeviceProp(),
                                 mTargets.getDevicePtr(),
                                 mNbTargetOutputs.getDevicePtr(),
                                 mLossMem.getDevicePtr(),
                                 mOutputs.getDevicePtr(),
                                 mDiffInputs.getDevicePtr(),
                                 mDiffInputs.dimZ(), // = getNbOutputs()
                                 mDiffInputs.dimY(),
                                 mDiffInputs.dimX(),
                                 mDiffInputs.dimB(),
                                 half_float::half(targetVal),
                                 half_float::half(defaultVal));
}

template <>
double Cell_Frame_CUDA<float>::setOutputTargetsInternal(
    double targetVal,
    double defaultVal)
{
    return cudaSSetOutputTargets(CudaContext::getDeviceProp(),
                                 mTargets.getDevicePtr(),
                                 mNbTargetOutputs.getDevicePtr(),
                                 mLossMem.getDevicePtr(),
                                 mOutputs.getDevicePtr(),
                                 mDiffInputs.getDevicePtr(),
                                 mDiffInputs.dimZ(), // = getNbOutputs()
                                 mDiffInputs.dimY(),
                                 mDiffInputs.dimX(),
                                 mDiffInputs.dimB(),
                                 (float)targetVal,
                                 (float)defaultVal);
}

template <>
double Cell_Frame_CUDA<double>::setOutputTargetsInternal(
    double targetVal,
    double defaultVal)
{
    return cudaDSetOutputTargets(CudaContext::getDeviceProp(),
                                 mTargets.getDevicePtr(),
                                 mNbTargetOutputs.getDevicePtr(),
                                 mLossMem.getDevicePtr(),
                                 mOutputs.getDevicePtr(),
                                 mDiffInputs.getDevicePtr(),
                                 mDiffInputs.dimZ(), // = getNbOutputs()
                                 mDiffInputs.dimY(),
                                 mDiffInputs.dimX(),
                                 mDiffInputs.dimB(),
                                 targetVal,
                                 defaultVal);
}
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::setOutputTargets(const BaseTensor& baseTargets)
{
    if (baseTargets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (baseTargets.dimX() != mOutputsDims[0]
        || baseTargets.dimY() != mOutputsDims[1]
        || baseTargets.dimZ() != getNbOutputs())
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputTargets(): wrong target "
            "matrix size. Expected " << mOutputsDims << ", got "
            << baseTargets.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    mOutputs.synchronizeDToH();

    const Tensor<T>& targets = tensor_cast<T>(baseTargets);

    double loss = 0.0;

    for (unsigned int index = 0; index < mOutputs.size(); ++index) {
        const double error = targets(index) - mOutputs(index);
        mDiffInputs(index) = error;
        loss += error * error;
    }

    mDiffInputs.synchronizeHToD();

    return (loss / mOutputs.dimB());
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setOutputErrors(const BaseTensor& baseErrors)
{
    if (baseErrors.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (baseErrors.dimX() != mOutputsDims[0]
        || baseErrors.dimY() != mOutputsDims[1]
        || baseErrors.dimZ() != getNbOutputs())
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputErrors(): wrong target "
            "matrix size. Expected " << mOutputsDims << ", got "
            << baseErrors.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    const Tensor<T>& errors = tensor_cast<T>(baseErrors);

    for (unsigned int index = 0; index < mOutputs.size(); ++index)
        mDiffInputs(index) = errors(index);

    mDiffInputs.synchronizeHToD();
}

template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getOutputs()
{
    return mOutputs;
}

template <class T>
const N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getOutputs() const
{
    return mOutputs;
}

template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getDiffInputs()
{
    return mDiffInputs;
}

template <class T>
const N2D2::BaseTensor&
N2D2::Cell_Frame_CUDA<T>::getDiffInputs() const
{
    return mDiffInputs;
}

template <class T>
unsigned int N2D2::Cell_Frame_CUDA<T>::getMaxOutput(unsigned int batchPos) const
{
    mOutputs.synchronizeDToH();
    const Tensor<T> output = mOutputs[batchPos];
    return std::distance(output.begin(),
                         std::max_element(output.begin(), output.end()));
}

template <class T>
N2D2::Cell_Frame_CUDA<T>::~Cell_Frame_CUDA()
{
    // dtor
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::discretizeSignals(unsigned int nbLevels,
                                              const Signals& signals)
{
    if (signals & In) {
        mInputs.synchronizeDBasedToH();

        for (CudaInterface<>::iterator itTensor = mInputs.begin(),
            itTensorEnd = mInputs.end(); itTensor != itTensorEnd; ++itTensor)
        {
            Tensor<T> input = tensor_cast<T>(*(*itTensor));

            //#pragma omp parallel for
            for (int index = 0; index < (int)input.size(); ++index)
                input(index) = Utils::round((nbLevels - 1) * input(index))
                                  / (nbLevels - 1);

            *(*itTensor) = input;
        }

        mInputs.synchronizeHToDBased();
    }

    if (signals & Out) {
        mOutputs.synchronizeDToH();

        //#pragma omp parallel for
        for (int index = 0; index < (int)mOutputs.size(); ++index)
            mOutputs(index) = Utils::round((nbLevels - 1) * mOutputs(index))
                              / (nbLevels - 1);

        mOutputs.synchronizeHToD();
    }
}

namespace N2D2 {
    template class Cell_Frame_CUDA<half_float::half>;
    template class Cell_Frame_CUDA<float>;
    template class Cell_Frame_CUDA<double>;
}

#endif


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
template<typename T>
void declare_Cell_Frame_CUDA(py::module &m, const std::string& typeStr) {
    const std::string pyClassName("Cell_Frame_CUDA_" + typeStr);
    py::class_<Cell_Frame_CUDA<T>, std::shared_ptr<Cell_Frame_CUDA<T>>, Cell, Cell_Frame_Top> (m, pyClassName.c_str(), py::multiple_inheritance());
}

void init_Cell_Frame_CUDA(py::module &m) {
    declare_Cell_Frame_CUDA<float>(m, "float");
    declare_Cell_Frame_CUDA<double>(m, "double");
}
}
#endif
