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

#include "Cell/TransformationCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Transformation/Transformation.hpp"

N2D2::Registrar<N2D2::TransformationCell>
N2D2::TransformationCell_Frame::mRegistrar(
    "Frame", N2D2::TransformationCell_Frame::create);

N2D2::TransformationCell_Frame::TransformationCell_Frame(
    const DeepNet& deepNet, 
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Transformation>& transformation)
    : Cell(deepNet, name, nbOutputs),
      TransformationCell(deepNet, name, nbOutputs, transformation),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs)
{
    // ctor
}

void N2D2::TransformationCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDBasedToH();

    if (mInputs.size() > 1)
        throw std::runtime_error("TransformationCell can only have one input");

    const Tensor<Float_T>& input0 = tensor_cast<Float_T>(mInputs[0]);

    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos)
        mOutputs[batchPos] = mTransformation->apply(input0[batchPos]);
}
