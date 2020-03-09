/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "N2D2.hpp"

#include "Cell/ResizeCell_Frame.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Random.hpp"

#include <limits>
#include <string>
#include <tuple>
#include <vector>

using namespace N2D2;

class ResizeCell_Frame_Test: public ResizeCell_Frame {
public:
    ResizeCell_Frame_Test(const DeepNet& deepNet,
                          const std::string& name,
                          unsigned int outputWidth,
                          unsigned int outputHeight,
                          unsigned int nbOutputs,
                          ResizeMode resizeMode):
        Cell(deepNet, name, nbOutputs),
        ResizeCell(deepNet, name, outputWidth, outputHeight, nbOutputs, resizeMode),
        ResizeCell_Frame(deepNet, name, outputWidth, outputHeight, nbOutputs, resizeMode) 
    {                                
    }

    friend class UnitTest_ResizeCell_Frame_propagate_bilinearTF_aligned_checkGradient;
    friend class UnitTest_ResizeCell_Frame_nearestNeighbor;
};

template<typename Iterator>
Tensor<Float_T> createBatchTensor(Iterator beginImg, Iterator endImg) {
    Tensor<Float_T> tensor;
    while(beginImg != endImg) {
        tensor.push_back(Tensor<Float_T>(*beginImg));
        ++beginImg;
    }
    
    return tensor;
}

TEST_DATASET(ResizeCell_Frame,
     propagate_bilinearTF_aligned_checkGradient,
     (unsigned int outputWidth,
     unsigned int outputHeight,
     unsigned int inputWidth,
     unsigned int inputHeight,
     unsigned int batchSize),
     std::make_tuple(24, 24, 12, 12, 1),
     std::make_tuple(92, 92, 53, 53, 1),
     std::make_tuple(32, 32, 96, 96, 1),
     std::make_tuple(32, 32, 96, 96, 4))
{
    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const unsigned int nbOutputs = 3;
    ResizeCell::ResizeMode mode = ResizeCell::ResizeMode::BilinearTF;
    ResizeCell_Frame_Test resize(dn, "resize",
                                  outputWidth,
                                  outputHeight,
                                  nbOutputs,
                                  mode);

    ASSERT_EQUALS(resize.getName(), "resize");
    ASSERT_EQUALS(resize.getNbOutputs(), nbOutputs);
    Tensor<Float_T> inputs({inputWidth, inputHeight, nbOutputs, batchSize});
    Tensor<Float_T> diffOutputs({inputWidth, inputHeight, nbOutputs, batchSize});

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);


    resize.addInput(inputs, diffOutputs);
    resize.setOutputsDims();
    resize.initialize();

    resize.propagate();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(resize.getOutputs());
    //ASSERT_NOTHROW_ANY(resize.checkGradient(1.0e-3, 1.0e-3));

    ASSERT_EQUALS(outputs.dimB(), batchSize);
    ASSERT_EQUALS(outputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(outputs.dimX(), outputWidth);
    ASSERT_EQUALS(outputs.dimY(), outputHeight);

}

TEST_DATASET(ResizeCell_Frame,
             nearestNeighbor,
            (unsigned int outputWidth,
             unsigned int outputHeight,
             unsigned int inputWidth,
             unsigned int inputHeight,
             unsigned int nbChannels,
             unsigned int batchSize),
            std::make_tuple(4, 4, 2, 2, 3, 1),
            std::make_tuple(24, 24, 6, 6, 7, 4),
            std::make_tuple(92, 27, 53, 53, 2, 7),
            std::make_tuple(32, 32, 96, 96, 4, 3),
            std::make_tuple(49, 31, 73, 85, 9, 5))
{
    Network net;
    DeepNet dn(net);

    cv::theRNG().state = std::numeric_limits<std::uint64_t>::max();

    /** 
     * Resize with OpenCV
     */
    std::vector<cv::Mat> inputs(batchSize, cv::Mat(inputHeight, inputWidth, CV_32FC(nbChannels)));
    std::vector<cv::Mat> outputsProp(batchSize, cv::Mat(outputHeight, outputWidth, CV_32FC(nbChannels)));
    std::vector<cv::Mat> outputsBackprop(batchSize, cv::Mat(inputHeight, inputWidth, CV_32FC(nbChannels)));
    
    for(std::size_t i = 0; i < batchSize; i++) {
        cv::randu(inputs[i], -10, 10);
        cv::resize(inputs[i], outputsProp[i], outputsProp[i].size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(outputsProp[i], outputsBackprop[i], outputsBackprop[i].size(), 0, 0, cv::INTER_NEAREST);
    }

    /** 
     * Resize input to outputsProp with N2D2 through propagate
     */
    ResizeCell_Frame_Test resize(dn, "r", outputWidth, outputHeight, nbChannels, 
                                      ResizeCell::ResizeMode::NearestNeighbor);

    Tensor<Float_T> inputTensor = createBatchTensor(inputs.begin(), inputs.end());
    Tensor<Float_T> diffOutputTensor(inputTensor.dims());

    resize.addInput(inputTensor, diffOutputTensor);
    resize.initialize();
    resize.propagate();
    
    ASSERT_EQUALS(tensor_cast_nocopy<Float_T>(resize.getOutputs()), createBatchTensor(outputsProp.begin(), outputsProp.end()));

    /** 
     * Resize outputsProp to outputsBackprop with N2D2 through backpropagate
     */
    resize.mDiffInputs = resize.getOutputs();
    resize.mDiffInputs.setValid();
    resize.mDiffInputs.synchronizeHToD();
    resize.backPropagate();
    
    ASSERT_EQUALS(diffOutputTensor, createBatchTensor(outputsBackprop.begin(), outputsBackprop.end()));
}

RUN_TESTS()
