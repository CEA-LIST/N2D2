/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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

#ifndef PLUGINLAYERS_HPP
#define PLUGINLAYERS_HPP
#include "plugins/anchor_cpu.hpp"
#include "plugins/anchor_gpu.hpp"
#include "plugins/batchnorm_cudnn.hpp"
#include "plugins/batchnorm_gpu.hpp"
#include "plugins/objectdetection_cpu.hpp"
#include "plugins/objectdetection_gpu.hpp"
#include "plugins/proposal_gpu.hpp"
#include "plugins/regionproposal_cpu.hpp"
#include "plugins/resize_gpu.hpp"
#include "plugins/roipooling_cpu.hpp"
#include "plugins/roipooling_gpu.hpp"

class PluginFactory : public nvinfer1::IPluginFactory
{


public:

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int stimuliHeight,
                                    unsigned int stimuliWidth,
                                    unsigned int nbFeature,
                                    unsigned int* featureChannels,
                                    unsigned int* featureHeight,
                                    unsigned int* featureWidth,
                                    Pooling_T poolType,
                                    unsigned int nbProposals,
                                    bool ignorePadding,
                                    bool isFlip)
    {
        if(!strncmp(layerName, "ROIPooling_CPU", 14))
        {
            mROIPoolingCPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 stimuliHeight,
                                 stimuliWidth,
                                 nbFeature,
                                 featureChannels,
                                 featureHeight,
                                 featureWidth,
                                 poolType,
                                 nbProposals,
                                 ignorePadding,
                                 isFlip);

            return mROIPoolingCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_GPU", 14))
        {
            mROIPoolingGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 stimuliHeight,
                                 stimuliWidth,
                                 nbFeature,
                                 featureChannels,
                                 featureHeight,
                                 featureWidth,
                                 poolType,
                                 nbProposals,
                                 ignorePadding,
                                 isFlip);

            return mROIPoolingGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of ROIPooling layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int featureHeight,
                                    unsigned int featureWidth,
                                    Pooling_T resizeType,
                                    bool alignCorner)
    {

        if(!strncmp(layerName, "Resize_GPU", 10))
        {
            mResizeGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 featureHeight,
                                 featureWidth,
                                 resizeType,
                                 alignCorner);

            return mResizeGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of resize layer is not implemented");
    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int nbAnchors,
                                    unsigned int channelHeight,
                                    unsigned int channelWidth,
                                    unsigned int nbProposals,
                                    unsigned int preNMsTopN,
                                    double nmsIoU,
                                    double minHeight,
                                    double minWidth,
                                    unsigned int scoreIndex,
                                    unsigned int iouIndex)
    {
        if(!strncmp(layerName, "RegionProposal_CPU", 18))
        {
            mRegionProposalCPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 nbAnchors,
                                 channelHeight,
                                 channelWidth,
                                 nbProposals,
                                 preNMsTopN,
                                 nmsIoU,
                                 minHeight,
                                 minWidth,
                                 scoreIndex,
                                 iouIndex);
            return mRegionProposalCPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of RegionProposal layer is not implemented");

    }



    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int nbProposals,
                                    unsigned int mNbCls,
                                    double nmsIoU,
                                    unsigned int scoreIndex,
                                    double scoreThreshold,
                                    unsigned int maxParts,
                                    unsigned int maxTemplates,
                                    const unsigned int* numPartsPerClass,
                                    const unsigned int* numTemplatesPerClass,
                                    const float* means,
                                    const float* std,
                                    bool applyNMS,
                                    bool keepMax,
                                    double normX,
                                    double normY)

    {
        if(!strncmp(layerName, "Proposals_GPU", 13))
        {
            mProposalGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 nbProposals,
                                 mNbCls,
                                 nmsIoU,
                                 scoreIndex,
                                 scoreThreshold,
                                 maxParts,
                                 maxTemplates,
                                 numPartsPerClass,
                                 numTemplatesPerClass,
                                 means,
                                 std,
                                 applyNMS,
                                 keepMax,
                                 normX,
                                 normY);
            return mProposalGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Proposal layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int channelHeight,
                                    unsigned int channelWidth,
                                    unsigned int stimuliWidth,
                                    unsigned int stimuliHeight,
                                    unsigned int featureMapWidth,
                                    unsigned int featureMapHeight,
                                    unsigned int nbProposals,
                                    unsigned int nbCls,
                                    unsigned int nbAnchors,
                                    double nmsIoU,
                                    const float* scoreThreshold,
                                    unsigned int maxParts,
                                    unsigned int maxTemplates,
                                    const unsigned int* numPartsPerClass,
                                    const unsigned int* numTemplatesPerClass,
                                    const float* anchor)

    {
        if(!strncmp(layerName, "ObjectDet_CPU", 13))
        {
            mObjectDetCPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 channelHeight,
                                 channelWidth,
                                 nbProposals,
                                 nbCls,
                                 nbAnchors,
                                 nmsIoU,
                                 scoreThreshold);

            return mObjectDetCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_GPU", 13))
        {
            mObjectDetGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 channelHeight,
                                 channelWidth,
                                 stimuliWidth,
                                 stimuliHeight,
                                 featureMapWidth,
                                 featureMapHeight,
                                 nbProposals,
                                 nbCls,
                                 nbAnchors,
                                 nmsIoU,
                                 scoreThreshold,
                                 maxParts,
                                 maxTemplates,
                                 numPartsPerClass,
                                 numTemplatesPerClass,
                                 anchor);

            return mObjectDetGPUPlugin.get();
        }

        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of ObjectDetect layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int stimuliHeight,
                                    unsigned int stimuliWidth,
                                    unsigned int featureMapWidth,
                                    unsigned int featureMapHeight,
                                    unsigned int scoreCls,
                                    bool isFlip,
                                    unsigned int nbAnchors,
                                    const float* anchors)
    {
        if(!strncmp(layerName, "Anchor_CPU", 10))
        {
            mAnchorCPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 stimuliHeight,
                                 stimuliWidth,
                                 featureMapWidth,
                                 featureMapHeight,
                                 scoreCls,
                                 isFlip,
                                 nbAnchors,
                                 anchors);
            return mAnchorCPUPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_GPU", 10))
        {
            mAnchorGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 stimuliHeight,
                                 stimuliWidth,
                                 featureMapWidth,
                                 featureMapHeight,
                                 scoreCls,
                                 isFlip,
                                 nbAnchors,
                                 anchors);
            return mAnchorGPUPlugin.get();
        }

        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Anchor layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    float* scales,
                                    float* biases,
                                    float* means,
                                    float* variances,
                                    float epsilon)
    {
        if(!strncmp(layerName, "BatchNorm_CUDA", 14))
        {
            mBatchNormCUDAPlugin.add(batchSize,
                                           nbOutputs,
                                        outputHeight,
                                        outputWidth,
                                        scales,
                                        biases,
                                        means,
                                        variances,
                                        epsilon);
            return mBatchNormCUDAPlugin.get();
        }
        else if (!strncmp(layerName, "BatchNorm_CUDNN", 15))
        {
            mBatchNormCUDNNPlugin.add(batchSize,
                                      nbOutputs,
                                      outputHeight,
                                      outputWidth,
                                      scales,
                                      biases,
                                      means,
                                      variances,
                                      epsilon);

            return mBatchNormCUDNNPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Batchnorm layer is not implemented");


    }

	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
        if(!strncmp(layerName, "BatchNorm_CUDA", 14))
        {
	    	mBatchNormCUDAPlugin.add(serialData, serialLength);
            return mBatchNormCUDAPlugin.get();
        }
        else if(!strncmp(layerName, "BatchNorm_CUDNN", 15))
        {
            mBatchNormCUDNNPlugin.add(serialData, serialLength);
            return mBatchNormCUDNNPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_CPU", 10))
        {
	    	mAnchorCPUPlugin.add(serialData, serialLength);
            return mAnchorCPUPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_GPU", 10))
        {
	    	mAnchorGPUPlugin.add(serialData, serialLength);
            return mAnchorGPUPlugin.get();
        }
        else if(!strncmp(layerName, "RegionProposal_CPU", 18))
        {
	    	mRegionProposalCPUPlugin.add(serialData, serialLength);
            return mRegionProposalCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_CPU", 14))
        {
	    	mROIPoolingCPUPlugin.add(serialData, serialLength);
            return mROIPoolingCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_GPU", 14))
        {
	    	mROIPoolingGPUPlugin.add(serialData, serialLength);
            return mROIPoolingGPUPlugin.get();
        }
        else if(!strncmp(layerName, "Proposals_GPU", 13))
        {
	    	mProposalGPUPlugin.add(serialData, serialLength);
            return mProposalGPUPlugin.get();
        }
        else if(!strncmp(layerName, "Resize_GPU", 10))
        {
	    	mResizeGPUPlugin.add(serialData, serialLength);
            return mResizeGPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_CPU", 13))
        {
	    	mObjectDetCPUPlugin.add(serialData, serialLength);
            return mObjectDetCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_GPU", 13))
        {
	    	mObjectDetGPUPlugin.add(serialData, serialLength);
            return mObjectDetGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin(const char*, const void*, size_t): this kind of layer is not implemented");

    }
    void destroyPlugin()
    {
        //BatchNormPlugin models destroy
        mBatchNormCUDAPlugin.destroy();
        mBatchNormCUDNNPlugin.destroy();
        //AnchorPlugin models destroy
        mAnchorCPUPlugin.destroy();
        mAnchorGPUPlugin.destroy();
        //Proposal models destroy
        mProposalGPUPlugin.destroy();
        //Region Proposal models destroy
        mRegionProposalCPUPlugin.destroy();
        //ROI Pooling models destroy
        mROIPoolingCPUPlugin.destroy();
        mROIPoolingGPUPlugin.destroy();

        mResizeGPUPlugin.destroy();

        mObjectDetCPUPlugin.destroy();
        mObjectDetGPUPlugin.destroy();

    }

    pluginAnchor_CPU mAnchorCPUPlugin;
    pluginAnchor_GPU mAnchorGPUPlugin;

    pluginBatchnorm_CUDA mBatchNormCUDAPlugin;
    pluginBatchnorm_CUDNN mBatchNormCUDNNPlugin;
    pluginRegionProposal_CPU mRegionProposalCPUPlugin;

    pluginProposal_GPU mProposalGPUPlugin;

    pluginROIPooling_CPU mROIPoolingCPUPlugin;
    pluginROIPooling_GPU mROIPoolingGPUPlugin;

    pluginResize_GPU mResizeGPUPlugin;

    pluginObjDet_CPU mObjectDetCPUPlugin;
    pluginObjDet_GPU mObjectDetGPUPlugin;

};
#endif // PLUGINLAYERS_HPP