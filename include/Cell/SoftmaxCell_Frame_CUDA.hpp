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

#ifndef N2D2_SOFTMAXCELL_FRAME_CUDA_H
#define N2D2_SOFTMAXCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame_CUDA.hpp"
#include "SoftmaxCell.hpp"
#include "Solver/SGDSolver_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include "DeepNet.hpp"

namespace N2D2 {
template <class T>
class SoftmaxCell_Frame_CUDA : public virtual SoftmaxCell,
                               public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivationDesc;

    SoftmaxCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                           unsigned int nbOutputs,
                           bool withLoss = false,
                           unsigned int groupSize = 0);
    static std::shared_ptr<SoftmaxCell> create(const DeepNet& deepNet, const std::string& name,
                                               unsigned int nbOutputs,
                                               bool withLoss = false,
                                               unsigned int groupSize = 0)
    {
        return std::make_shared
            <SoftmaxCell_Frame_CUDA>(deepNet, name, nbOutputs, withLoss, groupSize);
    }

    virtual void initialize();
    virtual void initializeDataDependent();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6) {};
    virtual ~SoftmaxCell_Frame_CUDA();

private:
    virtual void backPropagateWithLoss();

    cudnnTensorDescriptor_t mGroupTensor;

    static Registrar<SoftmaxCell> mRegistrar;
};
}

#endif // N2D2_SOFTMAXCELL_FRAME_CUDA_H
