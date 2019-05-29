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

#include "Cell/ThresholdCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Transformation/ThresholdTransformation.hpp"

N2D2::Registrar<N2D2::ThresholdCell>
N2D2::ThresholdCell_Frame::mRegistrar(
    "Frame", N2D2::ThresholdCell_Frame::create);

N2D2::ThresholdCell_Frame::ThresholdCell_Frame(
    const DeepNet& deepNet, 
    const std::string& name,
    unsigned int nbOutputs,
    double threshold)
    : Cell(deepNet, name, nbOutputs),
      ThresholdCell(deepNet, name, nbOutputs, threshold),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs)
{
    // ctor
}

void N2D2::ThresholdCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDBasedToH();

    if (mInputs.size() > 1)
        throw std::runtime_error("ThresholdCell can only have one input");

    const Tensor<Float_T>& input0 = tensor_cast<Float_T>(mInputs[0]);

    ThresholdTransformation trans(mThreshold);
    trans.setParameter<Operation>("Operation", mOperation);
    trans.setParameter<double>("MaxValue", mMaxValue);

    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos)
        mOutputs[batchPos] = trans.apply(input0[batchPos]);
}
