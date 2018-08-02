/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_CELL_CSPIKE_BP_CUDA_H
#define N2D2_CELL_CSPIKE_BP_CUDA_H

#include "Cell/Cell_CSpike_CUDA.hpp"
#include "CudaUtils.hpp"

namespace N2D2 {
class Cell_CSpike_BP_CUDA : public Cell_CSpike_CUDA {
public:

    Cell_CSpike_BP_CUDA(const std::string& name,
                        unsigned int nbOutputs);

    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Matrix<bool>& mapping = Matrix<bool>());

    virtual void addInput(Cell* cell,
                          const Matrix<bool>& mapping = Matrix<bool>());
    virtual void addInput(Cell* cell,
                            unsigned int x0,
                            unsigned int y0,
                            unsigned int width,
                            unsigned int height);

    virtual ~Cell_CSpike_BP_CUDA() {};

protected:
    Parameter<bool> mExampleReset;
    Parameter<unsigned int> mSubConnectIdx;

    Cell_CSpike_BP_CUDA* mLowerCell=nullptr;


};
}


#endif // N2D2_CELL_CSPIKE_BP_CUDA_H

