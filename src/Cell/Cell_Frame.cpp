/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#include "Cell/Cell_Frame.hpp"
#include "StimuliProvider.hpp"
#include "third_party/half.hpp"

template <class T>
N2D2::Cell_Frame<T>::Cell_Frame(const std::string& name,
                             unsigned int nbOutputs,
                             const std::shared_ptr
                             <Activation>& activation)
    : Cell(name, nbOutputs), Cell_Frame_Top(activation)
{
    // ctor
}

template <class T>
void N2D2::Cell_Frame<T>::save(const std::string& dirName) const
{
    Cell::save(dirName);
    Cell_Frame_Top::save(dirName);
}

template <class T>
void N2D2::Cell_Frame<T>::load(const std::string& dirName)
{
    Cell::load(dirName);
    Cell_Frame_Top::load(dirName);
}

template <class T>
void N2D2::Cell_Frame<T>::addInput(StimuliProvider& /*sp*/,
                                unsigned int /*channel*/,
                                unsigned int /*x0*/,
                                unsigned int /*y0*/,
                                unsigned int /*width*/,
                                unsigned int /*height*/,
                                const Tensor<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_Frame<T>::addInput(): adding a single "
                             "environment channel as input is not supported");
}

template <class T>
void N2D2::Cell_Frame<T>::addInput(StimuliProvider& sp,
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
        throw std::runtime_error("Cell_Frame<T>::addInput(): adding a cropped "
                                 "environment channel map as input is not "
                                 "supported");

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
    if (!mapping.empty() && mapping.dimY() != sp.getNbChannels())
        throw std::runtime_error("Cell_Frame<T>::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), sp.getNbChannels()}, true));
}

template <class T>
void N2D2::Cell_Frame<T>::addInput(Cell* cell, const Tensor<bool>& mapping)
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
            "Cell_Frame<T>::addInput(): cannot mix Spike and Frame models");
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
        throw std::runtime_error("Cell_Frame<T>::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), cellNbOutputs}, true));
}

template <class T>
void N2D2::Cell_Frame<T>::addInput(Cell* cell,
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
        throw std::runtime_error("Cell_Frame<T>::addInput(): adding a cropped "
                                 "output map as input is not supported");

    Cell_Frame<T>::addInput(cell);
}

template <class T>
void N2D2::Cell_Frame<T>::addInput(BaseTensor& inputs,
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
void N2D2::Cell_Frame<T>::replaceInput(BaseTensor& oldInputs,
                                       BaseTensor& newInputs,
                                       BaseTensor& newDiffOutputs)
{
    for (unsigned int i = 0; i < mInputs.size(); ++i) {
        if (&mInputs[i] == &oldInputs) {
            mInputs.replace(i, &newInputs);

            if (!newDiffOutputs.empty()) {
                assert(i < mDiffOutputs.size());
                mDiffOutputs.replace(i, &newDiffOutputs);
            }
        }
    }
}

template <class T>
void N2D2::Cell_Frame<T>::propagate(bool inference)
{
    if (mActivation)
        mActivation->propagate(mOutputs, inference);
}

template <class T>
void N2D2::Cell_Frame<T>::backPropagate()
{
    if (mActivation)
        mActivation->backPropagate(mOutputs, mDiffInputs);
}

template <class T>
double N2D2::Cell_Frame<T>::setOutputTarget(const Tensor<int>& targets,
                                            double targetVal,
                                            double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame::setOutputTarget(): target "
                                "and output batch sizes don't match.");

    if (targets.size() / targets.dimB() != 1)
        throw std::domain_error("Cell_Frame::setOutputTarget(): require "
                                "one target per batch.");

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

    return (loss / mOutputs.dimB());
}

template <class T>
double N2D2::Cell_Frame<T>::setOutputTargets(const Tensor<int>& targets,
                                             double targetVal,
                                             double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame<T>::setOutputTargets(): target and "
                                "output batch sizes don't match.");

    if (targets.dimX() != mOutputsDims[0] || targets.dimY() != mOutputsDims[1])
        throw std::domain_error(
            "Cell_Frame<T>::setOutputTargets(): wrong target matrix size.");

    double loss = 0.0;

#pragma omp parallel for if (mOutputs.dimB() > 4) reduction(+:loss)
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        const Tensor<int> target = targets[batchPos][0];

        std::vector<unsigned int> nbTargetOutputs(
            (getNbOutputs() > 1) ? getNbOutputs() : 2, 0);

        for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                if (target(ox, oy) >= 0) {
                    if ((getNbOutputs() > 1 && target(ox, oy) >= (int)getNbOutputs())
                        || (getNbOutputs() == 1
                            && (target(ox, oy) < 0 || target(ox, oy) > 1)))
                    {
#pragma omp critical
                        {
                            std::stringstream errorMsg;
                            errorMsg << "Cell_Frame<T>:: "
                                "setOutputTargets(): "
                                "output target (" << target(ox, oy) << ") out "
                                "of range [0," << (getNbOutputs()
                                                - (getNbOutputs() > 1)) << "].";

                            throw std::domain_error(errorMsg.str());
                        }
                    }

                    ++nbTargetOutputs[target(ox, oy)];
                }
            }
        }

        for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                for (unsigned int output = 0; output < getNbOutputs(); ++output) {
                    if (target(ox, oy) >= 0) {
                        const double error
                            = ((getNbOutputs() > 1 && target(ox, oy) == (int)output)
                               || (getNbOutputs() == 1 && target(ox, oy) == 1))
                                  ? targetVal
                                    - mOutputs(ox, oy, output, batchPos)
                                  : defaultVal
                                    - mOutputs(ox, oy, output, batchPos);

                        mDiffInputs(ox, oy, output, batchPos)
                            = error / nbTargetOutputs[target(ox, oy)];
                        loss += error * error;
                    } else
                        mDiffInputs(ox, oy, output, batchPos) = 0.0;
                }
            }
        }
    }

    return (loss / mOutputs.dimB());
}

template <class T>
double N2D2::Cell_Frame<T>::setOutputTargets(const BaseTensor& baseTargets)
{
    if (baseTargets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame<T>::setOutputTargets(): target and "
                                "output batch sizes don't match.");

    if (baseTargets.dimX() != mOutputsDims[0]
        || baseTargets.dimY() != mOutputsDims[1]
        || baseTargets.dimZ() != getNbOutputs())
    {
        throw std::domain_error(
            "Cell_Frame<T>::setOutputTargets(): wrong target matrix size.");
    }

    const Tensor<T>& targets = tensor_cast<T>(baseTargets);

    double loss = 0.0;

    for (unsigned int index = 0; index < mOutputs.size(); ++index) {
        const double error = targets(index) - mOutputs(index);
        mDiffInputs(index) = error;
        loss += error * error;
    }

    return (loss / mOutputs.dimB());
}

template <class T>
void N2D2::Cell_Frame<T>::setOutputErrors(const BaseTensor& baseErrors)
{
    if (baseErrors.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (baseErrors.dimX() != mOutputsDims[0]
        || baseErrors.dimY() != mOutputsDims[1]
        || baseErrors.dimZ() != getNbOutputs())
    {
        throw std::domain_error(
            "Cell_Frame::setOutputErrors(): wrong target matrix size.");
    }

    const Tensor<T>& errors = tensor_cast<T>(baseErrors);

    for (unsigned int index = 0; index < mOutputs.size(); ++index)
        mDiffInputs(index) = errors(index);
}

template <class T>
unsigned int N2D2::Cell_Frame<T>::getMaxOutput(unsigned int batchPos) const
{
    const Tensor<T> output = mOutputs[batchPos];
    return std::distance(output.begin(),
                         std::max_element(output.begin(), output.end()));
}

template <class T>
void N2D2::Cell_Frame<T>::discretizeSignals(unsigned int nbLevels,
                                         const Signals& signals)
{
    if (signals & In) {
        mInputs.synchronizeDToH();

        for (Interface<>::iterator itTensor = mInputs.begin(),
                                   itTensorEnd = mInputs.end();
             itTensor != itTensorEnd;
             ++itTensor)
        {
            Tensor<T> input = tensor_cast<T>(*(*itTensor));

            //#pragma omp parallel for
            for (int index = 0; index < (int)input.size(); ++index)
                input(index) = Utils::round((nbLevels - 1) * input(index))
                                  / (nbLevels - 1);

            *(*itTensor) = input;
        }

        mInputs.synchronizeHToD();
    }

    if (signals & Out) {
        //#pragma omp parallel for
        for (int index = 0; index < (int)mOutputs.size(); ++index)
            mOutputs(index) = Utils::round((nbLevels - 1) * mOutputs(index))
                              / (nbLevels - 1);
    }
}

namespace N2D2 {
    template class Cell_Frame<half_float::half>;
    template class Cell_Frame<float>;
    template class Cell_Frame<double>;
}
