/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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
#ifndef KERNELS_CPU_HPP
#define KERNELS_CPU_HPP

#include "dnn_utils.hpp"

void anchor_cpu(unsigned int batchSize,
                unsigned int nbOutputs,
                unsigned int outputHeight,
                unsigned int outputWidth,
                unsigned int stimuliHeight,
                unsigned int stimuliWidth,
                unsigned int scoreCls,
                bool isFlip,
                unsigned int nbAnchors,
                double xRatio,
                double yRatio,
                std::vector<Anchor> anchors,
                const float* inputs,
                float* outputs/*,
                float* maxIoU,
                float* ArgMaxIoU*/);

void region_proposal_cpu(unsigned int batchSize,
                         unsigned int nbOutputs,
                         unsigned int outputHeight,
                         unsigned int outputWidth,
                         unsigned int nbAnchors,
                         unsigned int channelHeight,
                         unsigned int channelWidth,
                         unsigned int nbProposals,
                         unsigned int preNmsTopN,
                         double nmsIoU,
                         double minHeight,
                         double minWidth,
                         unsigned int scoreIndex,
                         unsigned int iouIndex,
                         const float* inputs,
                         float* outputs);

void ROIPooling_bilinear_cpu(unsigned int batchSize,
                             unsigned int nbOutputs,
                             unsigned int outputHeight,
                             unsigned int outputWidth,
                             unsigned int stimuliHeight,
                             unsigned int stimuliWidth,
                             std::vector<trt_Dims3> featureDims,
                             unsigned int nbProposals,
                             const float* inputs,
                             float* outputs);

void object_det_cpu(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int channelHeight,
                    unsigned int channelWidth,
                    unsigned int nbAnchors,
                    unsigned int nbProposals,
                    unsigned int nbClass,
                    double nmsIoU,
                    const float* scoreThreshold,
                    const float* inputs,
                    float* outputs);

#endif
