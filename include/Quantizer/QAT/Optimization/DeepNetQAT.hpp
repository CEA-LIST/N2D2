/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifndef N2D2_DEEPNETQAT_H
#define N2D2_DEEPNETQAT_H

#include "DeepNetQuantization.hpp"
#include "Cell/BatchNormCell.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/FcCell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
class DeepNetQAT : public DeepNetQuantization {
public:
    using DeepNetQuantization::mDeepNet;

    DeepNetQAT(DeepNet& deepNet);

    void fuseQATGraph(StimuliProvider& sp,
                      ScalingMode actScalingMode,
                      WeightsApprox wMode = WeightsApprox::NONE,
                      WeightsApprox bMode = WeightsApprox::NONE,
                      WeightsApprox cMode = WeightsApprox::NONE);
    bool QuantizeAndfuseBatchNormWithConv(std::pair <size_t, size_t>& rangeConvBN,
                                std::pair <double, double>& alphasConvBN,
                                const std::shared_ptr<ConvCell>& convCell, 
                                const std::shared_ptr<BatchNormCell>& bnCell,
                                ScalingMode actScalingMode,
                                WeightsApprox wMode = WeightsApprox::NONE,
                                WeightsApprox bMode = WeightsApprox::NONE,
                                WeightsApprox cMode = WeightsApprox::NONE);
    bool QuantizeFC(std::pair <size_t, size_t>& range,
                    std::pair <double, double>& alpha,
                    const std::shared_ptr<FcCell>& fcCell,
                    WeightsApprox wMode = WeightsApprox::NONE,
                    WeightsApprox bMode = WeightsApprox::NONE,
                    WeightsApprox cMode = WeightsApprox::NONE);
    bool QuantizeElemWise(  std::pair <size_t, size_t>& rangeElWise,
                            std::pair <double, double>& alphaElWise,
                            size_t rangeParent0,
                            double alphaParent0,
                            const std::shared_ptr<ElemWiseCell>& elemWiseCell);
    void exportOutputsLayers(StimuliProvider& sp,
                             const std::string& dirName,
                             Database::StimuliSet set,
                             int nbStimuliMax);

    virtual ~DeepNetQAT() {};

protected:

private:

    std::vector<std::string> mVarNulName;
    std::vector<int> mVarNul;

};
}

#endif // N2D2_DEEPNETQAT_H
