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

#ifndef N2D2_RESIZECELL_FRAME_H
#define N2D2_RESIZECELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame.hpp"
#include "ResizeCell.hpp"
#include "ResizeCell_Frame_Kernels_struct.hpp"

namespace N2D2 {
class ResizeCell_Frame : public virtual ResizeCell, public Cell_Frame {
public:
    void BilinearInterpolation(const int out_size,
                                const int in_size,
                                const float scale,
                                ResizeCell_Frame_Kernels::PreComputed* interpolation);

    ResizeCell_Frame(const std::string& name,
                         unsigned int outputsWidth,
                         unsigned int outputsHeight,
                         unsigned int nbOutputs,
                         ResizeMode resizeMode);
    static std::shared_ptr<ResizeCell> create(const std::string& name,
                                                  unsigned int outputsWidth,
                                                  unsigned int outputsHeight,
                                                  unsigned int nbOutputs,
                                                  ResizeMode resizeMode)
    {
        return std::make_shared<ResizeCell_Frame>(name,
                                                      outputsWidth,
                                                      outputsHeight,
                                                      nbOutputs,
                                                      resizeMode);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    virtual ~ResizeCell_Frame() {};
protected:
    std::vector<ResizeCell_Frame_Kernels::PreComputed> mYStride;
    std::vector<ResizeCell_Frame_Kernels::PreComputed> mXStride;
    Float_T mScaleX;
    Float_T mScaleY;
private:
    static Registrar<ResizeCell> mRegistrar;
};
}

#endif // N2D2_RESIZECELL_FRAME_H