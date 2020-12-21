/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <string>
#include <unordered_map>

#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "N2D2.hpp"
#include "Histogram.hpp"
#include "Xnet/Network.hpp"
#include "RangeStats.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CellExport.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Target/TargetScore.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

static const unsigned int SEED = 7;

void fillRangeAndHistorgramNetwork(DeepNet& deepNet, std::size_t nbTestStimuli, 
                                   std::unordered_map<std::string, RangeStats>& outputsRange,
                                   std::unordered_map<std::string, Histogram>& outputsHistogram,
                                   std::size_t nbBits,
                                   ClippingMode actClippingMode) 
{
    const std::shared_ptr<StimuliProvider>& sp = deepNet.getStimuliProvider();
    DeepNetQuantization dnQuantization(deepNet);

    const std::size_t nbBatches = std::ceil(1.0*nbTestStimuli/sp->getBatchSize());
    for(std::size_t batch = 0; batch < nbBatches; batch++) {
        sp->readBatch(Database::Test, batch*sp->getBatchSize());
        deepNet.test(Database::Test);

        dnQuantization.reportOutputsRange(outputsRange);
        dnQuantization.reportOutputsHistogram(outputsHistogram, outputsRange, 
                                              nbBits, actClippingMode);
    }

    deepNet.clear(Database::Test);
}

double testNetwork(DeepNet& deepNet, std::size_t nbTestStimuli) {
    const std::shared_ptr<StimuliProvider>& sp = deepNet.getStimuliProvider();

    const std::size_t nbBatches = std::ceil(1.0*nbTestStimuli/sp->getBatchSize());
    for(std::size_t batch = 0; batch < nbBatches; batch++) {
        sp->readBatch(Database::Test, batch*sp->getBatchSize());
        deepNet.test(Database::Test);
    }


    if(deepNet.getTargets().size() != 1)  {
        throw std::runtime_error("Only one target is supported.");
    }

    std::shared_ptr<TargetScore> targetScore = std::dynamic_pointer_cast<TargetScore>(
                                                   deepNet.getTargets().front()
                                               );

    const double score = 100.0*targetScore->getAverageSuccess(Database::Test);
    deepNet.clear(Database::Test);

    return score;
}

bool isNormalized(DeepNet& deepNet, std::size_t nbTestStimuli) {
    const double delta = 0.05;

    const std::vector<std::vector<std::string>>& layers = deepNet.getLayers();
    const std::shared_ptr<StimuliProvider>& sp = deepNet.getStimuliProvider();

    // Check weights
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            const auto& cell = deepNet.getCell(*itCell);

            auto range = cell->getFreeParametersRange(false);
            if(Utils::max_abs(range.first, range.second) > 1.0 + delta) {
                return false;
            }
        }
    }

    // Check activations
    const std::size_t nbBatches = std::ceil(1.0*nbTestStimuli/sp->getBatchSize());
    for(std::size_t batch = 0; batch < nbBatches; batch++) {
        sp->readBatch(Database::Test, batch*sp->getBatchSize());

        sp->getData().synchronizeDBasedToH();
        const Tensor<Float_T>& envInput = sp->getData();
        for(Float_T val: envInput) {
            if(val < -1.0 - delta || val > 1.0 + delta) {
                return false;
            }
        }

        deepNet.test(Database::Test);

        for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
            for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
                const auto& cell = deepNet.getCell(*itCell);
                const auto cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
                
                cellFrame->getOutputs().synchronizeDBasedToH();
                const Tensor<Float_T>& outputs = tensor_cast<Float_T>(cellFrame->getOutputs());
                for(Float_T val : outputs) {
                    if(val < -1.0 - delta || val > 1.0 + delta) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


TEST_DATASET(DeepNetQuantization, quantization,
        (const std::string& model, const std::string& weightsDir,  
         bool unsignedEnv, bool rescalePerOutput, std::size_t nbTestStimuli, std::size_t nbBits, 
         ClippingMode actClippingMode, ScalingMode actScalingMode,
         double expectedScore, double expectedQuantizedScore),

    std::make_tuple("tests_data/mnist_model/model.ini", "tests_data/mnist_model/weights",
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6667),
    std::make_tuple("tests_data/mnist_model/model.ini", "tests_data/mnist_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::MSE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.7334),
    std::make_tuple("tests_data/mnist_model/model.ini", "tests_data/mnist_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::KL_DIVERGENCE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6667),

    std::make_tuple("tests_data/mnist_model/model.ini", "tests_data/mnist_model/weights", 
                    true, true, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6667),

    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FIXED_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::SINGLE_SHIFT,
                    98.5999, 98.4667),
    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::DOUBLE_SHIFT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::MSE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::KL_DIVERGENCE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.6000),

    std::make_tuple("tests_data/mnist_multibranch_model/model.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, true, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.6000)
)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    CellExport::mPrecision = static_cast<CellExport::Precision>(nbBits);
    DeepNetExport::mEnvDataUnsigned = unsignedEnv;

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, model);

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(weightsDir);

    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.01);



    std::unordered_map<std::string, Histogram> outputsHistogram;
    std::unordered_map<std::string, RangeStats> outputsRange;

    fillRangeAndHistorgramNetwork(*deepNet, nbTestStimuli, 
                                  outputsRange, outputsHistogram, nbBits, actClippingMode);


    DeepNetQuantization dnQuantization(*deepNet);
    dnQuantization.quantizeNetwork(outputsHistogram, outputsRange, nbBits, 
                                   actClippingMode, actScalingMode,
                                   rescalePerOutput);

    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0), 
        Database::All
    );
    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedQuantizedScore, 0.01);
}


#ifdef CUDA
TEST_DATASET(DeepNetQuantization, quantization_CUDA,
        (const std::string& model, const std::string& weightsDir,  
         bool unsignedEnv, bool rescalePerOutput, std::size_t nbTestStimuli, std::size_t nbBits, 
         ClippingMode actClippingMode, ScalingMode actScalingMode,
         double expectedScore, double expectedQuantizedScore),

    std::make_tuple("tests_data/mnist_model/model_CUDA.ini", "tests_data/mnist_model/weights",
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6667),
    std::make_tuple("tests_data/mnist_model/model_CUDA.ini", "tests_data/mnist_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::MSE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.7334),
    std::make_tuple("tests_data/mnist_model/model_CUDA.ini", "tests_data/mnist_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::KL_DIVERGENCE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6667),
    
    std::make_tuple("tests_data/mnist_model/model_CUDA.ini", "tests_data/mnist_model/weights", 
                    true, true, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    96.5999, 96.6000),

    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FIXED_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::SINGLE_SHIFT,
                    98.5999, 98.4667),
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::NONE, ScalingMode::DOUBLE_SHIFT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::MSE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.5334),
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, false, 500, 8, 
                    ClippingMode::KL_DIVERGENCE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.6000),
    
    std::make_tuple("tests_data/mnist_multibranch_model/model_CUDA.ini", "tests_data/mnist_multibranch_model/weights", 
                    true, true, 500, 8, 
                    ClippingMode::NONE, ScalingMode::FLOAT_MULT,
                    98.5999, 98.6000)
)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    CellExport::mPrecision = static_cast<CellExport::Precision>(nbBits);
    DeepNetExport::mEnvDataUnsigned = unsignedEnv;

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, model);

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(weightsDir);

    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.01);



    std::unordered_map<std::string, Histogram> outputsHistogram;
    std::unordered_map<std::string, RangeStats> outputsRange;

    fillRangeAndHistorgramNetwork(*deepNet, nbTestStimuli, 
                                  outputsRange, outputsHistogram, nbBits, actClippingMode);


    DeepNetQuantization dnQuantization(*deepNet);
    dnQuantization.quantizeNetwork(outputsHistogram, outputsRange, nbBits, 
                                   actClippingMode, actScalingMode,
                                   rescalePerOutput);

    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0), 
        Database::All
    );
    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedQuantizedScore, 0.01);
}
#endif

RUN_TESTS()