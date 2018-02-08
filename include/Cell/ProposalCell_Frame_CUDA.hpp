/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_PROPOSALCELL_FRAME_CUDA_H
#define N2D2_PROPOSALCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame_CUDA.hpp"
#include "ProposalCell.hpp"
#include "ProposalCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class ProposalCell_Frame_CUDA : public virtual ProposalCell, public Cell_Frame_CUDA {
public:

    ProposalCell_Frame_CUDA(const std::string& name,
                            StimuliProvider& sp,
                            const unsigned int nbOutputs,
                            unsigned int nbProposals,
                            unsigned int scoreIndex = 0,
                            unsigned int IoUIndex = 5,
                            bool isNms = false,
                            std::vector<double> meansFactor = { 0.0, 0.0, 0.0, 0.0},
                            std::vector<double> stdFactor = {1.0, 1.0, 1.0, 1.0},
                            std::vector<unsigned int> numParts = {},
                            std::vector<unsigned int> numTemplates = {});

    static std::shared_ptr<ProposalCell> create(const std::string& name,
                                          StimuliProvider& sp,
                                          const unsigned int nbOutputs,
                                          unsigned int nbProposals,
                                          unsigned int scoreIndex = 0,
                                          unsigned int IoUIndex = 5,
                                          bool isNms = false,
                                          std::vector<double> meansFactor = { 0.0, 0.0, 0.0, 0.0},
                                          std::vector<double> stdFactor = {1.0, 1.0, 1.0, 1.0},
                                          std::vector<unsigned int> numParts = {},
                                          std::vector<unsigned int> numTemplates = {})
    {
        return std::make_shared<ProposalCell_Frame_CUDA>(name,
                                                        sp,
                                                        nbOutputs,
                                                        nbProposals,
                                                        scoreIndex,
                                                        IoUIndex,
                                                        isNms,
                                                        meansFactor,
                                                        stdFactor,
                                                        numParts,
                                                        numTemplates);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon */ = 1.0e-4,
                       double /*maxError */ = 1.0e-6) {};
    void discretizeFreeParameters(unsigned int /*nbLevels*/) {}; // no free
    // parameter to
    // discretize
    virtual ~ProposalCell_Frame_CUDA() {};

protected:
    virtual void setOutputsSize();
  
    CudaTensor4d<Float_T> mMeansCUDA;
    CudaTensor4d<Float_T> mStdCUDA; 
    CudaTensor4d<Float_T> mNormalizeROIs; 
    CudaTensor4d<int> mMaxCls; 

    CudaTensor4d<Float_T> mPartsPrediction;
    CudaTensor4d<int> mNumPartsPerClass;
    CudaTensor4d<int> mPartsVisibilityPrediction;

    CudaTensor4d<Float_T> mTemplatesPrediction;
    CudaTensor4d<int> mNumTemplatesPerClass;
    CudaTensor4d<int> mKeepIndex;


private:
    static Registrar<ProposalCell> mRegistrar;
};
}

#endif // N2D2_PROPOSALCELL_FRAME_CUDA_H

