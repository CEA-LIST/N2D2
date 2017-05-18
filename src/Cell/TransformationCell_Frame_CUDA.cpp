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

#ifdef CUDA

#include "Cell/TransformationCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::TransformationCell>
N2D2::TransformationCell_Frame_CUDA::mRegistrar(
    "Frame_CUDA", N2D2::TransformationCell_Frame_CUDA::create);

N2D2::TransformationCell_Frame_CUDA::TransformationCell_Frame_CUDA(
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Transformation>& transformation)
    : Cell(name, nbOutputs),
      TransformationCell(name, nbOutputs, transformation),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
}

void N2D2::TransformationCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();

    if (mInputs.size() > 1)
        throw std::runtime_error("TransformationCell can only have one input");

    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos)
        mOutputs[batchPos] = mTransformation->apply(mInputs[0][batchPos]);

    mOutputs.synchronizeHToD();
}

#endif
