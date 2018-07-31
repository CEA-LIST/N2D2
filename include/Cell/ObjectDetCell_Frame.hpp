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
#include "ObjectDetCell.hpp"

namespace N2D2 {
class ObjectDetCell_Frame : public virtual ObjectDetCell, public Cell_Frame {
public:
    struct BBox_T {
        float x;
        float y;
        float w;
        float h;

        BBox_T() {}
        BBox_T(float x_, float y_, float w_, float h_):
            x(x_), y(y_), w(w_), h(h_) {}
    };


    ObjectDetCell_Frame(const std::string& name,
                        StimuliProvider& sp,
                        const unsigned int nbOutputs,
                        unsigned int nbAnchors,
                        unsigned int nbProposals,
                        unsigned int nbClass,
                        Float_T nmsThreshold = 0.5,
                        Float_T scoreThreshold = 0.0);

    static std::shared_ptr<ObjectDetCell> create(const std::string& name,
                                                StimuliProvider& sp,
                                                const unsigned int nbOutputs,
                                                unsigned int nbAnchors,
                                                unsigned int nbProposals,
                                                unsigned int nbClass,
                                                Float_T nmsThreshold = 0.5,
                                                Float_T scoreThreshold = 0.0)
    {
        return std::make_shared<ObjectDetCell_Frame>(name,
                                                    sp,
                                                    nbOutputs,
                                                    nbAnchors,
                                                    nbProposals,
                                                    nbClass,
                                                    nmsThreshold,
                                                    scoreThreshold);
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
    virtual ~ObjectDetCell_Frame() {};

protected:
    virtual void setOutputsDims();

private:
    static Registrar<ObjectDetCell> mRegistrar;
};
}

#endif // N2D2_OBJECTDETCELL_FRAME_H
