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

#include "AnchorCell.hpp"
#include "AnchorCell_Frame_CUDA_Kernels.hpp"
#include "Cell_Frame_CUDA.hpp"
#include "DeepNet.hpp"

namespace N2D2 {
class ROI;
class StimuliProvider;

class AnchorCell_Frame_CUDA : public virtual AnchorCell, public Cell_Frame_CUDA<Float_T> {
public:
    AnchorCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                 StimuliProvider& sp,
                 const AnchorCell_Frame_Kernels::DetectorType detectorType,
                 const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
                 unsigned int scoresCls = 1);
    static std::shared_ptr<AnchorCell> create(const DeepNet& deepNet, const std::string& name,
                  StimuliProvider& sp,
                  const AnchorCell_Frame_Kernels::DetectorType detectorType,
                  const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
                  unsigned int scoresCls)
    {
        return std::make_shared<AnchorCell_Frame_CUDA>(deepNet, name,
                                                       sp,
                                                       detectorType,
                                                       anchors,
                                                       scoresCls);
    }

    virtual const std::vector<AnchorCell_Frame_Kernels::BBox_T>&
        getGT(unsigned int batchPos) const;
    virtual std::shared_ptr<ROI> getAnchorROI(const Tensor<int>::Index& index)
        const;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorBBox(
        const Tensor<int>::Index& index) const;
    virtual AnchorCell_Frame_Kernels::BBox_T getAnchorGT(
        const Tensor<int>::Index& index) const;
    virtual Float_T getAnchorIoU(const Tensor<int>::Index& index) const;
    virtual int getAnchorArgMaxIoU(const Tensor<int>::Index& index) const;
    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    virtual int getNbAnchors() const;
    virtual std::vector<Float_T> getAnchor(const unsigned int idx) const;
    void checkGradient(double /*epsilon */ = 1.0e-4,
                       double /*maxError */ = 1.0e-6) {};
                       
    virtual void setAnchors(const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors);

    virtual ~AnchorCell_Frame_CUDA();

protected:
    std::vector<std::vector<AnchorCell_Frame_Kernels::BBox_T> > mGT;

    CudaTensor<AnchorCell_Frame_Kernels::Anchor> mAnchors;
    unsigned int mNbLabelsMax;
    CudaTensor<int> mArgMaxIoU;
    //FasterRCNN mode
    AnchorCell_Frame_Kernels::BBox_T** mCudaGT;
    CudaTensor<unsigned int> mNbLabels;
    CudaTensor<Float_T> mMaxIoU;
    //SingleShot mode
    CudaTensor<AnchorCell_Frame_Kernels::BBox_T> mGTClass;
    std::vector<std::vector<std::vector< AnchorCell_Frame_Kernels::BBox_T> > > mHostGTClass;

    CudaTensor<unsigned int> mNbLabelsClass;
    CudaTensor<Float_T> mMaxIoUClass;

    CudaTensor<int> mKeyNegSamples;
    CudaTensor<int> mKeyNegSamplesSorted;
    CudaTensor<Float_T> mConfNegSamples;
    CudaTensor<Float_T> mConfNegSamplesFiltered;

    CudaTensor<int> mKeyPosSamples;
    CudaTensor<int> mKeyPosSamplesSorted;
    CudaTensor<Float_T> mConfPosSamples;
    CudaTensor<Float_T> mConfPosSamplesFiltered;


    std::vector<dim3> GPU_BLOCK_GRID;
    std::vector<dim3> GPU_THREAD_GRID;

private:
    static Registrar<AnchorCell> mRegistrar;
};
}

#endif // N2D2_ANCHORCELL_FRAME_CUDA_H
