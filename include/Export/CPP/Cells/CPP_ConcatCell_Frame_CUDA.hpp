/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_CPP_CONCAT_CELL_FRAME_CUDA_H
#define N2D2_CPP_CONCAT_CELL_FRAME_CUDA_H

#include <string>

#include "Cell/Cell_Frame_CUDA.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"

namespace N2D2 {

class DeepNet;

class CPP_ConcatCell_Frame_CUDA final: public virtual CPP_ConcatCell, 
                                             public Cell_Frame_CUDA<Float_T> 
{
public:
    using Cell_Frame_CUDA<Float_T>::mInputs;
    using Cell_Frame_CUDA<Float_T>::mOutputs;
    using Cell_Frame_CUDA<Float_T>::mDiffInputs;
    using Cell_Frame_CUDA<Float_T>::mDiffOutputs;

    
    CPP_ConcatCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                 unsigned int nbOutputs);

    static std::shared_ptr<CPP_ConcatCell> create(const DeepNet& deepNet, 
                                                        const std::string& name,
                                                        unsigned int nbOutputs);

    void initialize() override;
    void propagate(bool inference = false) override;
    void backPropagate() override;
    void update() override;
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6) override;
    
    std::pair<double, double> getOutputsRange() const override;
};

}

#endif
