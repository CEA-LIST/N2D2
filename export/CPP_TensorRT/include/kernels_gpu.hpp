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
#ifndef KERNELS_GPU_H
#define KERNELS_GPU_H

#include "dnn_utils.hpp"
#include <cfloat>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

extern "C" void cuda_batchnormcell_propagate(unsigned int nbChannels,
                                            unsigned int channelsHeight,
                                            unsigned int channelsWidth,
                                            const float* inputs,
                                            unsigned int nbOutputs_,
                                            unsigned int outputOffset,
                                            float* outputs,
                                            const float* bias,
                                            const float* variance,
                                            const float* mean,
                                            const float* scale,
                                            const float epsilon,
                                            dim3 threadsPerBlocks,
                                            dim3 blocksPerGrid,
                                            cudaStream_t stream);

extern "C" void cuda_resize_bilinearTF_propagate(   unsigned int outputSizeX,
                                                    unsigned int outputSizeY,
                                                    unsigned int outputNbChannels,
                                                    unsigned int batchSize,
                                                    unsigned int inputSizeX,
                                                    unsigned int inputSizeY,
                                                    unsigned int* yLowIdx,
                                                    unsigned int* yHighIdx,
                                                    float* yInter,
                                                    unsigned int* xLowIdx,
                                                    unsigned int* xHighIdx,
                                                    float* xInter,
                                                    const float* input,
                                                    float* outputs,
                                                    const dim3 blocksPerGrid,
                                                    const dim3 threadsPerBlock,
                                                    cudaStream_t stream);

extern "C" void cuda_roipooling_bilinear_propagate(const float alpha,
                                                 const float* proposals,
                                                 unsigned int proposalIdx,
                                                 unsigned int nbProposals,
                                                 unsigned int inputSizeY,
                                                 unsigned int inputSizeX,
                                                 const float* inputs,
                                                 unsigned int nbChannels,
                                                 unsigned int channelsHeight,
                                                 unsigned int channelsWidth,
                                                 unsigned int batchSize,
                                                 unsigned int channelOffset,
                                                 const float beta,
                                                 float* outputs,
                                                 unsigned int nbOutputs,
                                                 unsigned int outputsHeight,
                                                 unsigned int outputsWidth,
                                                 unsigned int outputOffset,
                                                 bool bilinearTF,
                                                 bool isFlip,
                                                 bool ignorePadding,
                                                 dim3 threadsPerBlocks,
                                                 dim3 blocksPerGrid,
                                                 cudaStream_t stream);


extern "C" void cuda_proposal_normalize( unsigned int inputSizeX,
                                        unsigned int inputSizeY,
                                        unsigned int nbProposals,
                                        unsigned int batchSize,
                                        unsigned int scoreIdx,
                                        unsigned int nbCls,
                                        unsigned int maxParts,
                                        unsigned int maxTemplates,
                                        bool keepMax,
                                        bool generateParts,
                                        bool generateTemplates,
                                        const float normX,
                                        const float normY,
                                        const float* means,
                                        const float* std,
                                        const unsigned int* numPartsPerClass,
                                        const unsigned int* numTemplatesPerClass,
                                        const float* ROIRef,
                                        const float* ROIEst,
                                        const float* ValuesEst,
                                        const float* partsEst,
                                        const float* partsVisibilityEst,
                                        const float* templatesEst,
                                        float* outputs,
                                        int* argMax,
                                        float* partsPrediction,
                                        float* partsVisibilityPrediction,
                                        float* templatesPrediction,
                                        float scoreThreshold,
                                        const dim3 threadsPerBlock,
                                        const dim3 blocksPerGrid);

extern "C" void cuda_proposal_to_output( const unsigned int nbProposals,
                                         const unsigned int scoreIdx,
                                         const unsigned int nbCls,
                                         const unsigned int nbOutputs,
                                         unsigned int maxParts,
                                         unsigned int maxTemplates,
                                         bool generateParts,
                                         bool generateTemplates,
                                         const unsigned int* numPartsPerClass,
                                         const unsigned int* numTemplatesPerClass,
                                         const int* maxCls,
                                         const float* input,
                                         const int* predictionIndex,
                                         const float* partsPrediction,
                                         const float* partsVisibilityPrediction,
                                         const float* templatesPrediction,
                                         float* outputs,
                                         const dim3 threadsPerBlock,
                                         const dim3 blocksPerGrid);

extern "C" void cuda_anchor_propagate(unsigned int batchSize,
                                        unsigned int nbOutputs,
                                        unsigned int outputHeight,
                                        unsigned int outputWidth,
                                        unsigned int stimuliHeight,
                                        unsigned int stimuliWidth,
                                        unsigned int scoreCls,
                                        bool isCoordinatesAnchors,
                                        bool isFlip,
                                        unsigned int nbAnchors,
                                        const float* anchors,
                                        const float* inputs,
                                        float* outputs,
                                        dim3 threadsPerBlocks,
                                        dim3 blocksPerGrid,
                                        cudaStream_t stream);

extern "C" void cuda_region_proposal_split_indexes(  unsigned int inputSizeX,
                                                    unsigned int inputSizeY,
                                                    unsigned int nbAnchors,
                                                    unsigned int batchSize,
                                                    unsigned int nbBlocks,
                                                    const float* inputs,
                                                    float* values,
                                                    float* indexI,
                                                    float* indexJ,
                                                    float* indexK,
                                                    float* indexB,
                                                    unsigned int* map,
                                                    float minWidth,
                                                    float minHeight,
                                                    unsigned int scoreIndex);

extern "C" void cuda_region_proposal_nms(   unsigned int inputSizeX,
                                            unsigned int inputSizeY,
                                            unsigned int nbAnchors,
                                            unsigned int batchSize,
                                            const float* inputs,
                                            float* i,
                                            float* j,
                                            float* k,
                                            float* b,
                                            const unsigned int indexOffset,
                                            unsigned long long* mask,
                                            const unsigned int outputOffset,
                                            const float nms_iou_thresh,
                                            const unsigned int max_nbBoxes,
                                            const dim3 threadsPerBlock,
                                            const dim3 blocksPerGrid);

extern "C" void cuda_region_proposal_gathering( unsigned int inputSizeX,
                                                unsigned int inputSizeY,
                                                unsigned int nbAnchors,
                                                unsigned int batchSize,
                                                const float* inputs,
                                                const float* i,
                                                const float* j,
                                                const float* k,
                                                const float* b,
                                                const int* mask,
                                                float* outputs,
                                                const unsigned int topN,
                                                const unsigned int nbProposals,
                                                const unsigned int nbBlocks);

extern "C" void cuda_spatial_outputs(unsigned int nbClass,
                                    unsigned int targetHeight,
                                    unsigned int targetWidth,
                                    unsigned int batchSize,
                                    float threshold,
                                    float* targetData,
                                    uint32_t* outputEstimated,
                                    dim3 threadsPerBlocks,
                                    dim3 blocksPerGrid,
                                    cudaStream_t stream);

extern "C" void cudaS_SSD_output_gathering(     unsigned int batchSize,
                                            unsigned int nbClass,
                                            unsigned int nbAnchors,
                                            unsigned int channelWidth,
                                            unsigned int channelHeight,
                                            unsigned int nbProposals,
                                            unsigned int* nbValidROIs,
                                            unsigned int cls,
                                            unsigned int totalParts,
                                            unsigned int totalTemplates,
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            unsigned int cumulParts,
                                            unsigned int cumulTemplates,
                                            unsigned int nbParts,
                                            unsigned int nbTemplates,
                                            bool isCoordinatesAnchors,
                                            bool isPixelFormatXY,
                                            float xRatio,
                                            float yRatio,
                                            float xOutputRatio,
                                            float yOutputRatio,
                                            const float* roi_bbox,
                                            const float* roi_anchors,
                                            const float* anchors,
                                            const float* inputs_parts,
                                            const float* inputs_templates,
                                            float* outputs,
                                            const dim3 blocksPerGrid,
                                            const dim3 threadsPerBlock);


                                 
extern "C" void cudaSReduceIndex(    const unsigned int inputSize,
                                    const unsigned int inputBatchOffset,
                                    const unsigned int outputBatchOffset,
                                    const unsigned int channelWidth,
                                    const unsigned int channelHeight,
                                    const unsigned int nbAnchors,
                                    const float* valueThreshold,
                                    const float* inputs,
                                    int* outputMap,
                                    float* scores,
                                    const dim3 blocksPerGrid,
                                    const dim3 threadsPerBlock);

extern "C" void cuda_gather_int2int_indices( const int* keys,
                                                    const int* indicesX,
                                                    const int* indicesY,
                                                    const int* indicesK,
                                                    int* outX,
                                                    int* outY,
                                                    int* outK,
                                                    unsigned int nbElements,
                                                    const dim3 blocksPerGrid,
                                                    const dim3 threadsPerBlock);

extern "C" void cuda_add_weighted(unsigned int batchSize,
                                    unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      float* estimated_labels,
                                      unsigned int nbChannels,
                                      unsigned int image_height,
                                      unsigned int image_width,
                                      float* input_image,
                                      unsigned char* workspace,
                                      float alpha,
                                      const dim3 threadsPerBlock,
                                      const dim3 blocksPerGrid,
                                        cudaStream_t stream);

extern "C" void thrust_sort(float* inputs, unsigned int nbElements);

extern "C" void thrust_sort_keys(float* inputs,
                                 unsigned int* keys,
                                 unsigned int nbElements,
                                 unsigned int offset);

extern "C" void thrust_sort_keys_int(float* inputs,
                                     int* keys,
                                     unsigned int nbElements,
                                     unsigned int offset);


extern "C" void thrust_gather(const unsigned int* keys,
                              const float* inputs,
                              float* outputs,
                              unsigned int nbElements,
                              unsigned int inputOffset,
                              unsigned int outputOffset);

extern "C" void thrust_gather_int(const int* keys,
                                    const float* inputs,
                                    float* outputs,
                                    unsigned int nbElements,
                                    unsigned int inputOffset,
                                    unsigned int outputOffset);


extern "C" void thrust_gather_int2int(  const int* keys,
                                        const int* inputs,
                                        int* outputs,
                                        unsigned int nbElements,
                                        unsigned int inputOffset,
                                        unsigned int outputOffset);

extern "C" void thrust_scatter_int2int(  const int* keys,
                                        const int* inputs,
                                        int* outputs,
                                        unsigned int nbElements);


extern "C" int copy_if_int(const int* inputs,
                            int* outputs, 
                            unsigned int nbElements);
extern "C" int copy_if_float(const float* inputs,
                            float* outputs, 
                            unsigned int nbElements);



#endif