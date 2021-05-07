/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_PROPOSALCELL_FRAME_H
#define N2D2_PROPOSALCELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame.hpp"
#include "DeepNet.hpp"
#include "ObjectDetCell.hpp"

namespace N2D2 {
class ObjectDetCell_Frame : public virtual ObjectDetCell, public Cell_Frame<Float_T> {
public:
    ObjectDetCell_Frame(const DeepNet& deepNet, const std::string& name,
                        StimuliProvider& sp,
                        const unsigned int nbOutputs,
                        unsigned int nbAnchors,
                        const AnchorCell_Frame_Kernels::Format inputFormat,
                        const AnchorCell_Frame_Kernels::PixelFormat pixelFormat,
                        unsigned int nbProposals,
                        unsigned int nbClass,
                        Float_T nmsThreshold = 0.5,
                        std::vector<Float_T> scoreThreshold 
                                                    = std::vector<Float_T>(1, 0.5),
                        std::vector<unsigned int> numParts
                                                = std::vector<unsigned int>(),
                        std::vector<unsigned int> numTemplates
                                                = std::vector<unsigned int>(),
                        const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors 
                                                = std::vector<AnchorCell_Frame_Kernels::Anchor>());

    static std::shared_ptr<ObjectDetCell> create(const DeepNet& deepNet, const std::string& name,
                                                StimuliProvider& sp,
                                                const unsigned int nbOutputs,
                                                unsigned int nbAnchors,
                                                const AnchorCell_Frame_Kernels::Format inputFormat,
                                                const AnchorCell_Frame_Kernels::PixelFormat pixelFormat,
                                                unsigned int nbProposals,
                                                unsigned int nbClass,
                                                Float_T nmsThreshold = 0.5,
                                                std::vector<Float_T> scoreThreshold 
                                                    = std::vector<Float_T>(1, 0.5),
                                                std::vector<unsigned int> numParts
                                                        = std::vector<unsigned int>(),
                                                std::vector<unsigned int> numTemplates
                                                        = std::vector<unsigned int>(),
                                                const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors 
                                                                        = std::vector<AnchorCell_Frame_Kernels::Anchor>())
    {
        return std::make_shared<ObjectDetCell_Frame>(deepNet, name,
                                                    sp,
                                                    nbOutputs,
                                                    nbAnchors,
                                                    inputFormat,
                                                    pixelFormat,
                                                    nbProposals,
                                                    nbClass,
                                                    nmsThreshold,
                                                    scoreThreshold,
                                                    numParts,
                                                    numTemplates,
                                                    anchors);
    }
    virtual std::vector<Float_T> getAnchor(const unsigned int idx) const;

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon */ = 1.0e-4,
                       double /*maxError */ = 1.0e-6) {};
    virtual ~ObjectDetCell_Frame() {};

protected:
    virtual void setOutputsDims();
    Tensor<Float_T> mPartsPrediction;
    Tensor<Float_T> mTemplatesPrediction;
    std::vector<AnchorCell_Frame_Kernels::Anchor> mAnchors;

private:
    static Registrar<ObjectDetCell> mRegistrar;
};
}

#endif // N2D2_OBJECTDETCELL_FRAME_H
