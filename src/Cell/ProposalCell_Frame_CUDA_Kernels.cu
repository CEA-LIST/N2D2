
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND(david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Cell/ProposalCell_Frame_CUDA_Kernels.hpp"

__global__ void cudaSNormalizeROIs_kernel( unsigned int inputSizeX,
                                            unsigned int inputSizeY,
                                            unsigned int nbProposals,
                                            unsigned int batchSize,
                                            unsigned int scoreIdx,
                                            unsigned int nbCls,
                                            bool keepMax,
                                            const float normX,
                                            const float normY,
                                            const float* means,
                                            const float* std,
                                            const float* ROIRef,
                                            float* ROIEst,
                                            float* ValueEst,
                                            float* outputs,
                                            float* argMax,
                                            float scoreThreshold)
{
    const int batchPos = blockIdx.z*nbProposals;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbProposals)
    {
        unsigned int indexMin = scoreIdx;
        unsigned int indexMax = nbCls;

        if(keepMax)
        {
            unsigned int cls = scoreIdx;
            float maxVal = 0.0;

            for(unsigned int i = indexMin; i < indexMax; ++i)
            {
                unsigned int inputIdx = i + index*nbCls + batchPos*nbCls;

                if (ValueEst[inputIdx] >= maxVal)
                {
                    maxVal = ValueEst[inputIdx];
                    cls = i;
                }

            }
            argMax[index + batchPos] = cls;

            indexMin = cls;
            indexMax = cls + 1;
        }

        for(unsigned int clsIdx = indexMin; clsIdx < indexMax; ++clsIdx)
        {

            unsigned int bboxRefIdx = index*4 + batchPos*4;
            unsigned int bboxEstIdx = clsIdx*4 + index*4*nbCls + batchPos*4*nbCls;
            unsigned int valEstIdx = clsIdx + index*nbCls + batchPos*nbCls;
            //unsigned int outputIdx = keepMax ? index*4 + batchPos*4 : 
            //                            (clsIdx - scoreIdx)*4 + index*4*nbCls + batchPos*4*nbCls;
            unsigned int outputIdx = keepMax ? index*4*(nbCls - scoreIdx) + batchPos*4*(nbCls - scoreIdx)
                                        : (clsIdx - scoreIdx)*4 + index*4*(nbCls - scoreIdx) + batchPos*4*(nbCls - scoreIdx);


            const float xbbRef = ROIRef[0 + bboxRefIdx]*normX;
            const float ybbRef = ROIRef[1 + bboxRefIdx]*normY;
            const float wbbRef = ROIRef[2 + bboxRefIdx]*normX;
            const float hbbRef = ROIRef[3 + bboxRefIdx]*normY;


            const float xbbEst = ROIEst[0 + bboxEstIdx]*std[0] + means[0];

            const float ybbEst = ROIEst[1 + bboxEstIdx]*std[1] + means[1];

            const float wbbEst = ROIEst[2 + bboxEstIdx]*std[2] + means[2];

            const float hbbEst = ROIEst[3 + bboxEstIdx]*std[3] + means[3];


            float x = xbbEst*wbbRef + xbbRef + wbbRef/2.0 - (wbbRef/2.0)*exp(wbbEst);
            float y = ybbEst*hbbRef + ybbRef + hbbRef/2.0 - (hbbRef/2.0)*exp(hbbEst);
            float w = wbbRef*exp(wbbEst);
            float h = hbbRef*exp(hbbEst);

            /**Clip values**/
            if(x < 0.0)
            {
                w += x;
                x = 0.0;
            }

            if(y < 0.0)
            {
                h += y;
                y = 0.0;
            }

            w = ((w + x) > 1.0) ? (1.0 - x) / normX : w / normX;
            h = ((h + y) > 1.0) ? (1.0 - y) / normY : h / normY;

            x /= normX;
            y /= normY;
            
            if(ValueEst[valEstIdx] >= scoreThreshold)
            {
                outputs[0 + outputIdx] = x;
                outputs[1 + outputIdx] = y;
                outputs[2 + outputIdx] = w;
                outputs[3 + outputIdx] = h;

            }
            else
            {
                outputs[0 + outputIdx] = 0.0;
                outputs[1 + outputIdx] = 0.0;
                outputs[2 + outputIdx] = 0.0;
                outputs[3 + outputIdx] = 0.0;
            }    
        }
    }

}


__global__ void cudaSToOutput_kernel( const unsigned int nbProposals,
                                      const unsigned int scoreIdx,
                                      const unsigned int nbCls,
                                      const float* ROIEst,
                                      float* outputs)
{
    const int batchPos = blockIdx.z*nbProposals;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbProposals)
    {
        const unsigned int inputIdx = index*4*(nbCls - scoreIdx) 
                                            + batchPos*4*(nbCls - scoreIdx);

        outputs[0 + index*4 + batchPos*4] = ROIEst[0 + inputIdx];
        outputs[1 + index*4 + batchPos*4] = ROIEst[1 + inputIdx];
        outputs[2 + index*4 + batchPos*4] = ROIEst[2 + inputIdx];
        outputs[3 + index*4 + batchPos*4] = ROIEst[3 + inputIdx];        
    }

}
void N2D2::cudaSNormalizeROIs(unsigned int inputSizeX,
                        unsigned int inputSizeY,
                        unsigned int nbProposals,
                        unsigned int batchSize,
                        unsigned int scoreIdx,
                        unsigned int nbCls,
                        bool keepMax,
                        const float normX,
                        const float normY,
                        const float* means,
                        const float* std,
                        const float* ROIRef,
                        float* ROIEst,
                        float* ValueEst,
                        float* outputs,
                        float* argMax,
                        float scoreThreshold,
                        const dim3 threadsPerBlock,
                        const dim3 blocksPerGrid)
{

    cudaSNormalizeROIs_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSizeX,
                                                                    inputSizeY, 
                                                                    nbProposals,
                                                                    batchSize, 
                                                                    scoreIdx,
                                                                    nbCls,
                                                                    keepMax,
                                                                    normX, 
                                                                    normY, 
                                                                    means, 
                                                                    std, 
                                                                    ROIRef, 
                                                                    ROIEst,
                                                                    ValueEst,
                                                                    outputs,
                                                                    argMax,
                                                                    scoreThreshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}


void N2D2::cudaSToOutputROIs(const unsigned int nbProposals,
                             const unsigned int scoreIdx,
                             const unsigned int nbCls,
                             const float* ROIEst,
                             float* outputs,
                             const dim3 threadsPerBlock,
                             const dim3 blocksPerGrid)
{

    cudaSToOutput_kernel<<<blocksPerGrid, threadsPerBlock>>>( nbProposals,
                                                              scoreIdx,
                                                              nbCls,
                                                              ROIEst, 
                                                              outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}
