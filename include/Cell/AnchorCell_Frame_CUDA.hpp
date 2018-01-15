/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_ANCHORCELL_FRAME_CUDA_H
#define N2D2_ANCHORCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame_CUDA.hpp"
#include "AnchorCell.hpp"
#include "AnchorCell_Frame_CUDA_Kernels.hpp"

namespace N2D2 {
class AnchorCell_Frame_CUDA : public virtual AnchorCell, public Cell_Frame_CUDA {
public:
    AnchorCell_Frame_CUDA(const std::string& name,
                 StimuliProvider& sp,
                 const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
                 unsigned int scoresCls = 1);
    static std::shared_ptr<AnchorCell> create(const std::string& name,
                  StimuliProvider& sp,
                  const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
                  unsigned int scoresCls)
    {
        return std::make_shared<AnchorCell_Frame_CUDA>(name,
                                                       sp,
                                                       anchors,
                                                       scoresCls);
    }

    virtual const std::vector<AnchorCell_Frame_Kernels::BBox_T>&
        getGT(unsigned int batchPos) const;
    virtual std::shared_ptr<ROI> getAnchorROI(const Tensor4d<int>::Index& index)
        const;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorBBox(
        const Tensor4d<int>::Index& index) const;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorGT(
        const Tensor4d<int>::Index& index) const;
    virtual Float_T getAnchorIoU(const Tensor4d<int>::Index& index) const;
    virtual int getAnchorArgMaxIoU(const Tensor4d<int>::Index& index) const;
    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    virtual int getNbAnchors() const;
    virtual std::vector<Float_T> getAnchor(const unsigned int idx) const;
    void checkGradient(double /*epsilon */ = 1.0e-4,
                       double /*maxError */ = 1.0e-6) {};
    virtual ~AnchorCell_Frame_CUDA();

protected:
    CudaTensor4d<AnchorCell_Frame_Kernels::Anchor> mAnchors;
    std::vector<std::vector<AnchorCell_Frame_Kernels::BBox_T> > mGT;
    AnchorCell_Frame_Kernels::BBox_T** mCudaGT;
    unsigned int mNbLabelsMax;
    CudaTensor4d<unsigned int> mNbLabels;
    CudaTensor4d<int> mArgMaxIoU;
    CudaTensor4d<Float_T> mMaxIoU;

private:
    static Registrar<AnchorCell> mRegistrar;
};
}

#endif // N2D2_ANCHORCELL_FRAME_CUDA_H
