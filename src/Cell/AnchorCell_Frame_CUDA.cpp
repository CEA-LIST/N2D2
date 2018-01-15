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

#ifdef CUDA

#include "Cell/AnchorCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::AnchorCell>
N2D2::AnchorCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                        N2D2::AnchorCell_Frame_CUDA::create);

N2D2::AnchorCell_Frame_CUDA::AnchorCell_Frame_CUDA(
    const std::string& name,
    StimuliProvider& sp,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
    unsigned int scoresCls)
    : Cell(name, 6*anchors.size()),
      AnchorCell(name, sp, anchors, scoresCls),
      Cell_Frame_CUDA(name, 6*anchors.size()),
      mCudaGT(NULL),
      mNbLabelsMax(16)
{
    // ctor
    mAnchors.push_back(Tensor3d<AnchorCell_Frame_Kernels::Anchor>(
        1, 1, anchors.size(), anchors.begin(), anchors.end()));
}
int N2D2::AnchorCell_Frame_CUDA::getNbAnchors() const { return mAnchors.size(); }

std::vector<N2D2::Float_T> N2D2::AnchorCell_Frame_CUDA::getAnchor(unsigned int idx) const
{
    std::vector<Float_T> vect_anchor;
    vect_anchor.push_back(mAnchors(idx).x0);
    vect_anchor.push_back(mAnchors(idx).y0);
    vect_anchor.push_back(mAnchors(idx).x1);
    vect_anchor.push_back(mAnchors(idx).y1);
    return vect_anchor;
}
const std::vector<N2D2::AnchorCell_Frame_Kernels::BBox_T>&
N2D2::AnchorCell_Frame_CUDA::getGT(unsigned int batchPos) const
{
    assert(batchPos < mGT.size());

    return mGT[batchPos];
}

std::shared_ptr<N2D2::ROI>
N2D2::AnchorCell_Frame_CUDA::getAnchorROI(const Tensor4d<int>::Index& index) const
{
    mArgMaxIoU.synchronizeDToH();
    const int argMaxIoU = mArgMaxIoU(index);

    if (argMaxIoU >= 0) {
        std::vector<std::shared_ptr<ROI> > labelROIs
            = mStimuliProvider.getLabelsROIs(index.b);
        assert(argMaxIoU < (int)labelROIs.size());
        return labelROIs[argMaxIoU];
    }
    else
        return std::shared_ptr<ROI>();
}

N2D2::AnchorCell_Frame_Kernels::BBox_T
N2D2::AnchorCell_Frame_CUDA::getAnchorBBox(const Tensor4d<int>::Index& index) const
{
    assert(index.i < mArgMaxIoU.dimX());
    assert(index.j < mArgMaxIoU.dimY());
    assert(index.k < mArgMaxIoU.dimZ());
    assert(index.b < mArgMaxIoU.dimB());

    const unsigned int nbAnchors = mAnchors.size();
    mOutputs.synchronizeDToH();
    const Float_T xa = mOutputs(index.i, index.j,
                                index.k + 1 * nbAnchors, index.b);
    const Float_T ya = mOutputs(index.i, index.j,
                                index.k + 2 * nbAnchors, index.b);
    const Float_T wa = mOutputs(index.i, index.j,
                                index.k + 3 * nbAnchors, index.b);
    const Float_T ha = mOutputs(index.i, index.j,
                                index.k + 4 * nbAnchors, index.b);
    return AnchorCell_Frame_Kernels::BBox_T(xa, ya, wa, ha);
}

N2D2::AnchorCell_Frame_Kernels::BBox_T
N2D2::AnchorCell_Frame_CUDA::getAnchorGT(const Tensor4d<int>::Index& index) const
{
    assert(index.b < mGT.size());

    mArgMaxIoU.synchronizeDToH();
    const int argMaxIoU = mArgMaxIoU(index);

    assert(argMaxIoU < 0 || argMaxIoU < (int)mGT[index.b].size());

    return ((argMaxIoU >= 0)
        ? mGT[index.b][argMaxIoU]
        : AnchorCell_Frame_Kernels::BBox_T(0.0, 0.0, 0.0, 0.0));
}

N2D2::Float_T
N2D2::AnchorCell_Frame_CUDA::getAnchorIoU(const Tensor4d<int>::Index& index) const
{
    assert(index.i < mArgMaxIoU.dimX());
    assert(index.j < mArgMaxIoU.dimY());
    assert(index.k < mArgMaxIoU.dimZ());
    assert(index.b < mArgMaxIoU.dimB());

    mOutputs.synchronizeDToH();
    return mOutputs(index.i, index.j, index.k + 5 * mAnchors.size(), index.b);
}

int
N2D2::AnchorCell_Frame_CUDA::getAnchorArgMaxIoU(const Tensor4d<int>::Index& index)
    const
{
    mArgMaxIoU.synchronizeDToH();
    return mArgMaxIoU(index);
}

void N2D2::AnchorCell_Frame_CUDA::initialize()
{
    const unsigned int nbAnchors = mAnchors.size();

    if (mInputs.dimZ() != (mScoresCls + 4) * nbAnchors) {
        throw std::domain_error("AnchorCell_Frame_CUDA::initialize():"
                                " the number of input channels must be equal to"
                                " (scoresCls + 4) times the number of"
                                " anchors.");
    }

    if (mInputs.size() > 1 && mInputs[0].dimZ() != mScoresCls * nbAnchors) {
        throw std::domain_error("AnchorCell_Frame_CUDA::initialize():"
                                " the first input number of channels must be"
                                " equal to scoresCls times the number of"
                                " anchors.");
    }

    if (mFlip) {
        const double xRatio = std::ceil(mStimuliProvider.getSizeX()
                                        / (double)mOutputs.dimX());
        const double yRatio = std::ceil(mStimuliProvider.getSizeY()
                                        / (double)mOutputs.dimY());

        const double xOffset = mStimuliProvider.getSizeX() - 1
                                - (mOutputsWidth - 1) * xRatio;
        const double yOffset = mStimuliProvider.getSizeY() - 1
                                - (mOutputsHeight - 1) * yRatio;

        for (unsigned int k = 0; k < nbAnchors; ++k) {
            AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors(k);
            anchor.x0 += xOffset;
            anchor.y0 += yOffset;
            anchor.x1 += xOffset;
            anchor.y1 += yOffset;
        }
    }

    mGT.resize(mOutputs.dimB());

    CHECK_CUDA_STATUS(
        cudaMalloc(&mCudaGT, mOutputs.dimB() * sizeof(*mCudaGT)));

    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        CHECK_CUDA_STATUS(
            cudaMalloc(&mCudaGT + batchPos, mNbLabelsMax * sizeof(**mCudaGT)));
    }

    mNbLabels.resize(mOutputs.dimB(), 1, 1, 1, 0);

    mArgMaxIoU.resize(mOutputsWidth,
                      mOutputsHeight,
                      mAnchors.size(),
                      mOutputs.dimB());
    mMaxIoU.resize(mOutputs.dimB(), 1, 1, 1, 0.0);
}

void N2D2::AnchorCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

#pragma omp parallel for if (mOutputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        std::vector<AnchorCell_Frame_Kernels::BBox_T>& GT = mGT[batchPos];

        // Extract ground true ROIs
        std::vector<std::shared_ptr<ROI> > labelROIs
            = mStimuliProvider.getLabelsROIs(batchPos);

        GT.resize(labelROIs.size());

        mNbLabels(batchPos) = labelROIs.size();

        if (labelROIs.size() > mNbLabelsMax) {
            CHECK_CUDA_STATUS(cudaFree(mCudaGT + batchPos));
            CHECK_CUDA_STATUS(
                cudaMalloc(&mCudaGT + batchPos,
                           labelROIs.size() * sizeof(**mCudaGT)));

            std::cout << Utils::cwarning
                << "AnchorCell_Frame_CUDA::propagate(): reallocating mCudaGT "
                "with size " << labelROIs.size() << " (initial: "
                << mNbLabelsMax << ")" << Utils::cdef << std::endl;
        }

        for (unsigned int l = 0; l < labelROIs.size(); ++l) {
            cv::Rect labelRect = labelROIs[l]->getBoundingRect();

            // Crop labelRect to the slice for correct overlap area calculation
            if (labelRect.tl().x < 0) {
                labelRect.width+= labelRect.tl().x;
                labelRect.x = 0;
            }
            if (labelRect.tl().y < 0) {
                labelRect.height+= labelRect.tl().y;
                labelRect.y = 0;
            }
            if (labelRect.br().x > (int)mStimuliProvider.getSizeX())
                labelRect.width = mStimuliProvider.getSizeX() - labelRect.x;
            if (labelRect.br().y > (int)mStimuliProvider.getSizeY())
                labelRect.height = mStimuliProvider.getSizeY() - labelRect.y;

            GT[l] = AnchorCell_Frame_Kernels::BBox_T(labelRect.tl().x,
                                                     labelRect.tl().y,
                                                     labelRect.width,
                                                     labelRect.height);
        }

        CHECK_CUDA_STATUS(cudaMemcpy(&mCudaGT[batchPos],
                                     &GT,
                                     labelROIs.size() * sizeof(**mCudaGT),
                                     cudaMemcpyHostToDevice));
    }

    mNbLabels.synchronizeHToD();
    mMaxIoU.fill(0.0);
    mMaxIoU.synchronizeHToD();
    mAnchors.synchronizeHToD();

    cudaSAnchorPropagate(mStimuliProvider.getSizeX(),
                         mStimuliProvider.getSizeY(),
                         mFlip,
                         inference,
                         mInputs[0].getDevicePtr(),
                         (mInputs.size() > 1)
                            ? mInputs[1].getDevicePtr()
                            : mInputs[0].getDevicePtr(),
                         (mInputs.size() > 1)
                            ? 0
                            : mScoresCls,
                         mAnchors.getDevicePtr(),
                         mCudaGT,
                         mNbLabels.getDevicePtr(),
                         mOutputs.getDevicePtr(),
                         mArgMaxIoU.getDevicePtr(),
                         mMaxIoU.getDevicePtr(),
                         mAnchors.size(),
                         mOutputs.dimY(),
                         mOutputs.dimX(),
                         mInputs.dimB(),
                         mScoresCls,
                         mInputs.size());
    mOutputs.synchronizeDToH();

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::AnchorCell_Frame_CUDA::backPropagate()
{
    throw std::runtime_error(
        "AnchorCell_Frame_CUDA::backPropagate(): not implemented.");
}

void N2D2::AnchorCell_Frame_CUDA::update()
{
    // Nothing to update
}

N2D2::AnchorCell_Frame_CUDA::~AnchorCell_Frame_CUDA()
{
    if (mCudaGT != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mCudaGT));
        
        mCudaGT = NULL;
    }
}

#endif
