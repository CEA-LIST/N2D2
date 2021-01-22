/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_RPCELL_FRAME_CUDA_H
#define N2D2_RPCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "RPCell.hpp"
#include "RPCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include <thrust/sort.h>
#include <thrust/functional.h>

namespace N2D2 {
class RPCell_Frame_CUDA : public virtual RPCell, public Cell_Frame_CUDA<Float_T> {
public:
    RPCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                 unsigned int nbAnchors,
                 unsigned int nbProposals,
                 unsigned int scoreIndex = 0,
                 unsigned int IoUIndex = 5);
    static std::shared_ptr<RPCell> create(const DeepNet& deepNet, const std::string& name,
                                          unsigned int nbAnchors,
                                          unsigned int nbProposals,
                                          unsigned int scoreIndex = 0,
                                          unsigned int IoUIndex = 5)
    {
        return std::make_shared<RPCell_Frame_CUDA>(deepNet, name,
                                              nbAnchors,
                                              nbProposals,
                                              scoreIndex,
                                              IoUIndex);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon */ = 1.0e-4,
                       double /*maxError */ = 1.0e-6) {};
    virtual ~RPCell_Frame_CUDA() {};

protected:
    virtual void setOutputsDims();

    CudaTensor<int> mIndexI;
    CudaTensor<int> mSortedIndexI;

    CudaTensor<int> mIndexJ;
    CudaTensor<int> mSortedIndexJ;

    CudaTensor<int> mIndexK;
    CudaTensor<int> mSortedIndexK;

    CudaTensor<int> mIndexB;
    CudaTensor<int> mSortedIndexB;

    CudaTensor<unsigned int> mMap;
    CudaTensor<unsigned long long> mMask;

    CudaTensor<Float_T> mValues;

    CudaTensor<int> mSortedIndex;
    CudaTensor<int> mGPUAnchors;


private:
    static Registrar<RPCell> mRegistrar;
};
}

#endif // N2D2_RPCELL_FRAME_CUDA_H

