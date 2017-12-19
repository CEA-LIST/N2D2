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

N2D2::Cell_Frame::Cell_Frame(const std::string& name,
                             unsigned int nbOutputs,
                             const std::shared_ptr
                             <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs), Cell_Frame_Top(activation)
{
    // ctor
}

void N2D2::Cell_Frame::addInput(StimuliProvider& /*sp*/,
                                unsigned int /*channel*/,
                                unsigned int /*x0*/,
                                unsigned int /*y0*/,
                                unsigned int /*width*/,
                                unsigned int /*height*/,
                                const std::vector<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_Frame::addInput(): adding a single "
                             "environment channel as input is not supported");
}

void N2D2::Cell_Frame::addInput(StimuliProvider& sp,
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height,
                                const Matrix<bool>& mapping)
{
    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (x0 > 0 || y0 > 0 || width < sp.getSizeX() || height < sp.getSizeY())
        throw std::runtime_error("Cell_Frame::addInput(): adding a cropped "
                                 "environment channel map as input is not "
                                 "supported");

    // Define input-output sizes
    setInputsSize(width, height);
    mNbChannels += sp.getNbChannels();

    mInputs.push_back(&sp.getData());
    setOutputsSize();

    if (mOutputs.empty()) {
        mOutputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, sp.getBatchSize());
        mDiffInputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, mInputs.dimB());
    }

    // Define input-output connections
    if (!mapping.empty() && mapping.rows() != sp.getNbChannels())
        throw std::runtime_error("Cell_Frame::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMaps.resize(mNbOutputs, mNbChannels);
    const unsigned int channelOffset = mNbChannels - sp.getNbChannels();

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < sp.getNbChannels();
             ++channel) {
            mMaps(output, channelOffset + channel)
                = (!mapping.empty()) ? mapping(channel, output) : true;
        }
    }
}

void N2D2::Cell_Frame::addInput(Cell* cell, const Matrix<bool>& mapping)
{
    // Define input-output sizes
    setInputsSize(cell->getOutputsWidth(), cell->getOutputsHeight());
    mNbChannels += cell->getNbOutputs();

    Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(cell);

    if (cellFrame != NULL) {
        mInputs.push_back(&cellFrame->getOutputs());
        mDiffOutputs.push_back(&cellFrame->getDiffInputs());
    }
    else {
        throw std::runtime_error(
            "Cell_Frame::addInput(): cannot mix Spike and Frame models");
    }

    setOutputsSize();

    if (mOutputs.empty()) {
        mOutputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, mInputs.dimB());
        mDiffInputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, mInputs.dimB());
    }

    // Define input-output connections
    if (!mapping.empty() && mapping.rows() != cell->getNbOutputs())
        throw std::runtime_error("Cell_Frame::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMaps.resize(mNbOutputs, mNbChannels);
    const unsigned int channelOffset = mNbChannels - cell->getNbOutputs();

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < cell->getNbOutputs();
             ++channel) {
            mMaps(output, channelOffset + channel)
                = (!mapping.empty()) ? mapping(channel, output) : true;
        }
    }
}

void N2D2::Cell_Frame::addInput(Cell* cell,
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
        throw std::runtime_error("Cell_Frame::addInput(): adding a cropped "
                                 "output map as input is not supported");

    Cell_Frame::addInput(cell);
}

void N2D2::Cell_Frame::addInput(Tensor4d<Float_T>& inputs,
                                Tensor4d<Float_T>& diffOutputs)
{
    // Define input-output sizes
    setInputsSize(inputs.dimX(), inputs.dimY());
    mNbChannels += inputs.dimZ();

    mInputs.push_back(&inputs);

    if (!diffOutputs.empty())
        mDiffOutputs.push_back(&diffOutputs);

    setOutputsSize();

    if (mOutputs.empty()) {
        mOutputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, mInputs.dimB());
        mDiffInputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, mInputs.dimB());
    }

    mMaps.resize(mNbOutputs, mNbChannels, true);
}

void N2D2::Cell_Frame::propagate(bool /*inference*/)
{
    if (mActivation)
        mActivation->propagate(&mOutputs);
}

void N2D2::Cell_Frame::backPropagate()
{
    if (mActivation)
        mActivation->backPropagate(&mOutputs, &mDiffInputs);
}

void N2D2::Cell_Frame::setOutputTarget(const Tensor4d<int>& targets,
                                       double targetVal,
                                       double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA::setOutputTarget(): target "
                                "and output batch sizes don't match.");

    if (targets.size() / targets.dimB() != 1)
        throw std::domain_error("Cell_Frame_CUDA::setOutputTarget(): require "
                                "one target per batch.");

    const unsigned int outputSize = mOutputs.size() / mOutputs.dimB();

    for (unsigned int batchPos = 0; batchPos < mOutputs.dimB(); ++batchPos) {
        if (targets(0, batchPos) >= 0) {
            if ((outputSize > 1 && targets(0, batchPos) >= (int)outputSize)
                || (outputSize == 1 && (targets(0, batchPos) < 0
                                        || targets(0, batchPos) > 1))) {
                throw std::domain_error("Cell_Frame_CUDA::setOutputTarget(): "
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
            } else
                mDiffInputs(index, batchPos) = 0.0;
        }
    }
}

void N2D2::Cell_Frame::setOutputTargets(const Tensor4d<int>& targets,
                                        double targetVal,
                                        double defaultVal)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame::setOutputTargets(): target and "
                                "output batch sizes don't match.");

    if (targets.dimX() != mOutputsWidth || targets.dimY() != mOutputsHeight)
        throw std::domain_error(
            "Cell_Frame::setOutputTargets(): wrong target matrix size.");

    for (unsigned int batchPos = 0; batchPos < mOutputs.dimB(); ++batchPos) {
        const Tensor2d<int> target = targets[batchPos][0];

        std::vector<unsigned int> nbTargetOutputs(
            (mNbOutputs > 1) ? mNbOutputs : 2, 0);

        for (unsigned int oy = 0; oy < mOutputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsWidth; ++ox) {
                if (target(ox, oy) >= 0) {
                    if ((mNbOutputs > 1 && target(ox, oy) >= (int)mNbOutputs)
                        || (mNbOutputs == 1
                            && (target(ox, oy) < 0 || target(ox, oy) > 1))) {
                        throw std::domain_error("Cell_Frame::setOutputTargets()"
                                                ": output target out of "
                                                "range.");
                    }

                    ++nbTargetOutputs[target(ox, oy)];
                }
            }
        }

        for (unsigned int oy = 0; oy < mOutputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsWidth; ++ox) {
                for (unsigned int output = 0; output < mNbOutputs; ++output) {
                    if (target(ox, oy) >= 0) {
                        const double error
                            = ((mNbOutputs > 1 && target(ox, oy) == (int)output)
                               || (mNbOutputs == 1 && target(ox, oy) == 1))
                                  ? targetVal
                                    - mOutputs(ox, oy, output, batchPos)
                                  : defaultVal
                                    - mOutputs(ox, oy, output, batchPos);

                        mDiffInputs(ox, oy, output, batchPos)
                            = error / nbTargetOutputs[target(ox, oy)];
                    } else
                        mDiffInputs(ox, oy, output, batchPos) = 0.0;
                }
            }
        }
    }
}

void N2D2::Cell_Frame::setOutputTargets(const Tensor4d<Float_T>& targets)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame::setOutputTargets(): target and "
                                "output batch sizes don't match.");

    if (targets.dimX() != mOutputsWidth || targets.dimY() != mOutputsHeight
        || targets.dimZ() != mNbOutputs)
        throw std::domain_error(
            "Cell_Frame::setOutputTargets(): wrong target matrix size.");

    for (unsigned int index = 0; index < mOutputs.size(); ++index)
        mDiffInputs(index) = targets(index) - mOutputs(index);
}

void N2D2::Cell_Frame::setOutputErrors(const Tensor4d<Float_T>& errors)
{
    if (errors.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (errors.dimX() != mOutputsWidth || errors.dimY() != mOutputsHeight
        || errors.dimZ() != mNbOutputs)
        throw std::domain_error(
            "Cell_Frame_CUDA::setOutputErrors(): wrong target matrix size.");

    for (unsigned int index = 0; index < mOutputs.size(); ++index)
        mDiffInputs(index) = errors(index);
}

unsigned int N2D2::Cell_Frame::getMaxOutput(unsigned int batchPos) const
{
    const Tensor3d<Float_T> output = mOutputs[batchPos];
    return std::distance(output.begin(),
                         std::max_element(output.begin(), output.end()));
}

void N2D2::Cell_Frame::discretizeSignals(unsigned int nbLevels,
                                         const Signals& signals)
{
    if (signals & In) {
        mInputs.synchronizeDToH();

        for (std::vector<Tensor4d<Float_T>*>::iterator itTensor = mInputs.begin(),
                                                       itTensorEnd = mInputs.end();
             itTensor != itTensorEnd;
             ++itTensor) {
            Tensor4d<Float_T>* input = (*itTensor);

            //#pragma omp parallel for
            for (int index = 0; index < (int)input->size(); ++index)
                (*input)(index) = Utils::round((nbLevels - 1) * (*input)(index))
                                  / (nbLevels - 1);
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
