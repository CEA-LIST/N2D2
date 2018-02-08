
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
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            bool keepMax,
                                            bool generateParts,
                                            bool generateTemplates,
                                            const float normX,
                                            const float normY,
                                            const float* means,
                                            const float* std,
                                            const int* numPartsPerClass,
                                            const int* numTemplatesPerClass,
                                            const float* ROIRef,
                                            const float* ROIEst,
                                            const float* ValueEst,
                                            const float* partsEst,
                                            const float* partsVisibilityEst,
                                            const float* templatesEst,
                                            float* outputs,
                                            int* argMax,
                                            float* partsPrediction,
                                            float* templatesPrediction,
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
        else
           argMax[index + batchPos] = -1; 

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
                if(generateParts)
                {
                    unsigned int partsIdx = 0;
                    unsigned int proposalPartIdx = 0;
                    for(int idxCls = 0; idxCls < clsIdx; ++idxCls)
                        partsIdx += numPartsPerClass[idxCls];

                    proposalPartIdx = partsIdx;

                    for(int idxCls = clsIdx; idxCls < nbCls; ++idxCls)
                        proposalPartIdx += numPartsPerClass[idxCls];

                    for(unsigned int part = 0; part < numPartsPerClass[clsIdx];
                            ++part)
                    {
                        //const unsigned int partIdx = (partsIdx + part)*2;
                        const unsigned int inPartIdx = batchPos*2*proposalPartIdx 
                                                        + index*2*proposalPartIdx 
                                                        + partsIdx*2
                                                        + part*2;

                        const unsigned int outPartIdx = batchPos*2*maxParts*nbCls 
                                                        + index*2*maxParts*nbCls
                                                        + clsIdx*2*maxParts
                                                        + part*2;

                        const float partY = partsEst[0 + inPartIdx];
                        const float partX = partsEst[1 + inPartIdx];
                        partsPrediction[0 + outPartIdx] = ((partY + 0.5) * hbbRef + ybbRef) / normY;
                        partsPrediction[1 + outPartIdx] = ((partX + 0.5) * wbbRef + xbbRef) / normX;

                    }
                }
                if(generateTemplates)
                {
                    unsigned int templatesIdx = 0;
                    unsigned int proposalTemplateIdx = 0;
                    for(int idxCls = 0; idxCls < clsIdx; ++idxCls)
                        templatesIdx += numTemplatesPerClass[idxCls];

                    proposalTemplateIdx = templatesIdx;

                    for(int idxCls = clsIdx; idxCls < nbCls; ++idxCls)
                        proposalTemplateIdx += numTemplatesPerClass[idxCls];

                    for(unsigned int tpl = 0; tpl < numTemplatesPerClass[clsIdx];
                            ++tpl)
                    {
                        const unsigned int inTemplateIdx = batchPos*3*proposalTemplateIdx 
                                                        + index*3*proposalTemplateIdx 
                                                        + templatesIdx*3 
                                                        + tpl*3;

                        const unsigned int outTemplateIdx = batchPos*3*maxTemplates*nbCls 
                                                        + index*3*maxTemplates*nbCls
                                                        + clsIdx*3*maxTemplates
                                                        + tpl*3;

                        const float templateA = expf(templatesEst[0 + inTemplateIdx]);
                        const float templateB = expf(templatesEst[1 + inTemplateIdx]);
                        const float templateC = expf(templatesEst[2 + inTemplateIdx]);

                        templatesPrediction[0 + outTemplateIdx] = templateA;
                        templatesPrediction[1 + outTemplateIdx] = templateB;
                        templatesPrediction[2 + outTemplateIdx] = templateC;
                    }
                }
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
                                      const unsigned int nbOutputs,
                                      const unsigned int maxParts,
                                      const unsigned int maxTemplates,
                                      bool generateParts,
                                      bool generateTemplates,
                                      const int* numPartsPerClass,
                                      const int* numTemplatesPerClass,
                                      const int* maxCls,
                                      const float* ROIEst,
                                      const int* predictionIndex,
                                      const float* partsPrediction,
                                      const float* templatesPrediction,
                                      float* outputs)
{
    const int batchPos = blockIdx.z*nbProposals;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbProposals)
    {
        const unsigned int inputIdx = index*4*(nbCls - scoreIdx) 
                                        + batchPos*4*(nbCls - scoreIdx);
        unsigned int outputIdx = 0;
        unsigned offset = 0;

        if((nbOutputs == 4))
            outputIdx = index*4 + batchPos*4;
        else if((nbOutputs == 5))
            outputIdx = index*5 + batchPos*5;
        else if(generateParts && generateTemplates)
            outputIdx = (index + batchPos)*(5 + maxParts*2 + maxTemplates*3);
        else if(generateTemplates)
            outputIdx = (index + batchPos)*(5 + maxTemplates*3);
        else if(generateParts)
            outputIdx = (index + batchPos)*(5 + maxParts*2);

        outputs[0 + outputIdx] = ROIEst[0 + inputIdx];
        outputs[1 + outputIdx] = ROIEst[1 + inputIdx];
        outputs[2 + outputIdx] = ROIEst[2 + inputIdx];
        outputs[3 + outputIdx] = ROIEst[3 + inputIdx];

        offset += 4;

        if(nbOutputs > 4)    
        {
            int cls = maxCls[index + batchPos];
            outputs[4 + outputIdx] = cls > -1 ? 
                                    (float) cls
                                    : 0.0;
            offset += 1;
        }
        
        if(generateParts)
        {
            const int predProp = predictionIndex[(index + batchPos)*2 + 0];
            const int predCls = predictionIndex[(index + batchPos)*2 + 1];

            if(predCls > -1)
            {
                for(unsigned int part = 0; part < numPartsPerClass[predCls];
                     ++part)
                {
                    const unsigned int partIdx = batchPos*maxParts*2*nbCls 
                                            + predProp*maxParts*2*nbCls
                                            + predCls*maxParts*2
                                            + part*2;
                    outputs[0 + offset + part*2 + outputIdx] = partsPrediction[0 + partIdx];
                    outputs[1 + offset + part*2 + outputIdx] = partsPrediction[1 + partIdx];

                }
                for(int idx = numPartsPerClass[predCls]; idx < maxParts; ++idx)
                {
                    outputs[0 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
                    outputs[1 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
                }
            }
            offset += maxParts*2;
        }

        if(generateTemplates)
        {
            
            const int predProp = predictionIndex[(index + batchPos)*2 + 0];
            const int predCls = predictionIndex[(index + batchPos)*2 + 1];

            if(predCls > -1)
            {
                for(unsigned int tpl = 0; tpl < numTemplatesPerClass[predCls]; ++tpl)
                {
                    unsigned int templateIdx = batchPos*maxTemplates*3*nbCls 
                                                + predProp*maxTemplates*3*nbCls
                                                + predCls*maxTemplates*3
                                                + tpl*3;

                    outputs[0 + offset + tpl*3 + outputIdx] = templatesPrediction[0 + templateIdx];
                    outputs[1 + offset + tpl*3 + outputIdx] = templatesPrediction[1 + templateIdx];
                    outputs[2 + offset + tpl*3 + outputIdx] = templatesPrediction[2 + templateIdx];

                }
                for(int idx = numTemplatesPerClass[predCls]; idx < maxParts; ++idx)
                {
                    outputs[0 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
                    outputs[1 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
                    outputs[2 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;

                }
            
            }
        }
    }

}
void N2D2::cudaSNormalizeROIs(unsigned int inputSizeX,
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
                        const int* numPartsPerClass,
                        const int* numTemplatesPerClass,
                        const float* ROIRef,
                        const float* ROIEst,
                        const float* ValueEst,
                        const float* partsEst,
                        const float* partsVisibilityEst,
                        const float* templatesEst,
                        float* outputs,
                        int* maxCls,
                        float* partsPrediction,
                        float* templatesPrediction,
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
                                                                    maxParts,
                                                                    maxTemplates,
                                                                    keepMax,
                                                                    generateParts,
                                                                    generateTemplates,
                                                                    normX, 
                                                                    normY, 
                                                                    means, 
                                                                    std, 
                                                                    numPartsPerClass,
                                                                    numTemplatesPerClass,
                                                                    ROIRef, 
                                                                    ROIEst,
                                                                    ValueEst,
                                                                    partsEst,
                                                                    partsVisibilityEst,
                                                                    templatesEst,
                                                                    outputs,
                                                                    maxCls,
                                                                    partsPrediction,
                                                                    templatesPrediction,
                                                                    scoreThreshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}


void N2D2::cudaSToOutputROIs(const unsigned int nbProposals,
                            const unsigned int scoreIdx,
                            const unsigned int nbCls,
                            const unsigned int nbOutputs,
                            unsigned int maxParts,
                            unsigned int maxTemplates,
                            bool generateParts,
                            bool generateTemplates,
                            const int* numPartsPerClass,
                            const int* numTemplatesPerClass,
                            const int* maxCls,
                            const float* ROIEst,
                            const int* predictionIndex,
                            const float* partsPrediction,
                            const float* templatesPrediction,
                            float* outputs,
                            const dim3 threadsPerBlock,
                            const dim3 blocksPerGrid)
{

    cudaSToOutput_kernel<<<blocksPerGrid, threadsPerBlock>>>( nbProposals,
                                                              scoreIdx,
                                                              nbCls,
                                                              nbOutputs,
                                                              maxParts,
                                                              maxTemplates,
                                                              generateParts,
                                                              generateTemplates,
                                                              numPartsPerClass,
                                                              numTemplatesPerClass,                                                              
                                                              maxCls,
                                                              ROIEst, 
                                                              predictionIndex,
                                                              partsPrediction,
                                                              templatesPrediction,
                                                              outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}