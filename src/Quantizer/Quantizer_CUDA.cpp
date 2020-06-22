/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)

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

#include "Quantizer/Quantizer_CUDA.hpp"


N2D2::Quantizer_CUDA::Quantizer_CUDA()
{
    // ctor
}



//void N2D2::Quantizer_CUDA::addInput(BaseTensor& inputs,
//                                     BaseTensor& /*diffOutputs*/)
void N2D2::Quantizer_CUDA::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new CudaTensor<float>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new CudaTensor<float>(weights.dims()));
}


void N2D2::Quantizer_CUDA::addActivations(BaseTensor& activations, BaseTensor& diffActivations)
{
    mFullPrecisionActivations.push_back(&activations);
    mQuantizedActivations.push_back(new CudaTensor<float>(activations.dims()));

    mDiffQuantizedActivations.push_back(&diffActivations);
    mDiffFullPrecisionActivations.push_back(new CudaTensor<float>(activations.dims()));
}


void N2D2::Quantizer_CUDA::addCell(Cell* /*cell*/)
{
    //
}




void N2D2::Quantizer_CUDA::initialize()
{
    // resize output variables etc.
}


void N2D2::Quantizer_CUDA::update()
{
    // Update all trainable parameters
}


void N2D2::Quantizer_CUDA::propagate()
{
    // 
}



void N2D2::Quantizer_CUDA::back_propagate()
{
    //
}



N2D2::Quantizer_CUDA::~Quantizer_CUDA()
{
   //dtor
}

#endif
