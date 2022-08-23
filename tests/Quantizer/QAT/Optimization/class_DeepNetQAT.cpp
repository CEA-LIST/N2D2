/**
 * (C) Copyright 2021 CEA LIST. All Rights Reserved.
 *  Contributor(s): David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#include <string>
#include <unordered_map>

#include "DeepNet.hpp"
#include "Quantizer/QAT/Optimization/DeepNetQAT.hpp"
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
#include "DrawNet.hpp"

using namespace N2D2;

static const unsigned int SEED = 7;

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

    std::cout << "    Score = " << score << std::endl;
    std::cout << "    Sensitivity: " << (100.0
                    * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Sensitivity))
                << "% / Specificity: " << (100.0
                * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Specificity))
                << "% / Precision: " << (100.0
                * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Precision))
                << "%\n"
                "    Accuracy: " << (100.0
                * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Accuracy))
                << "% / F1-score: " << (100.0
                * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::F1Score))
                << "% / Informedness: " << (100.0
                * targetScore->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Informedness))
                << "%\n" << std::endl;

    deepNet.clear(Database::Test);

    return score;
}

#ifdef CUDA
TEST_DATASET(DeepNetQAT, QAT_inference_SAT_CUDA,
        (const std::string& model, const std::string& weightsDir,  
         bool unsignedEnv, std::size_t nbTestStimuli, 
         ScalingMode actScalingMode,
         WeightsApprox weightsApproximationMode,
         double expectedScore),
    std::make_tuple("tests_data/mnist_model/LeNet-bn-SAT-8b.ini", "tests_data/mnist_model/weights_8b_SAT",
                    true, 10000,
                    ScalingMode::FLOAT_MULT, WeightsApprox::NONE, 99.32),
    std::make_tuple("tests_data/mnist_model/LeNet-bn-SAT-8b.ini", "tests_data/mnist_model/weights_8b_SAT",
                    true, 10000,
                    ScalingMode::FLOAT_MULT, WeightsApprox::RINTF, 99.32),    
    std::make_tuple("tests_data/mnist_model/LeNet-bn-SAT.ini", "tests_data/mnist_model/weights_test_SAT",
                    true, 10000,
                    ScalingMode::FLOAT_MULT, WeightsApprox::NONE, 99.25),
    std::make_tuple("tests_data/mnist_model/LeNet-bn-SAT.ini", "tests_data/mnist_model/weights_test_SAT",
                    true, 10000,
                    ScalingMode::FLOAT_MULT,  WeightsApprox::RINTF, 99.25))
{
    CudaContext::setDevice(0);

    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));
    DeepNetExport::mEnvDataUnsigned = unsignedEnv;

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, model);

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(weightsDir);
    deepNet->removeDropout();
    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_full_precision.png"));
    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0),
        Database::All
    );
    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.5);
    DeepNetQAT dnQAT(*deepNet);
    dnQAT.fuseQATGraph(*deepNet->getStimuliProvider(), actScalingMode, weightsApproximationMode, weightsApproximationMode, weightsApproximationMode);

    deepNet->exportNetworkFreeParameters("./w_q_mnist");

    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0), 
        Database::All
    );

    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_quant.png"));
    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.5);
}
#endif


#ifdef CUDA
TEST_DATASET(DeepNetQAT, QAT_multibranch_inference_SAT_CUDA,
        (const std::string& model, const std::string& weightsDir,
         bool unsignedEnv, std::size_t nbTestStimuli,
         ScalingMode actScalingMode,
         WeightsApprox weightsApproximationMode,
         double expectedScore),
        std::make_tuple("tests_data/mnist_multibranch_model/MobileNet_v2_mini.ini",
                        "tests_data/mnist_multibranch_model/mobilenet-v2-weights-8b",
                        true, 10000,
                        ScalingMode::FLOAT_MULT, WeightsApprox::NONE, 98.67),
        std::make_tuple("tests_data/mnist_multibranch_model/MobileNet_v2_mini.ini",
                        "tests_data/mnist_multibranch_model/mobilenet-v2-weights-8b",
                        true, 10000,
                        ScalingMode::FLOAT_MULT,  WeightsApprox::RINTF, 98.67),

        std::make_tuple("tests_data/mnist_multibranch_model/MobileNet_v2_mini_wSym.ini",
                        "tests_data/mnist_multibranch_model/mobilenet-v2-weights-sym-8b",
                        true, 10000,
                        ScalingMode::FLOAT_MULT, WeightsApprox::NONE, 98.55),
        std::make_tuple("tests_data/mnist_multibranch_model/MobileNet_v2_mini_wSym.ini",
                        "tests_data/mnist_multibranch_model/mobilenet-v2-weights-sym-8b",
                        true, 10000,
                        ScalingMode::FLOAT_MULT,  WeightsApprox::RINTF, 98.55)
                    )
{
    CudaContext::setDevice(0);

    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));
    DeepNetExport::mEnvDataUnsigned = unsignedEnv;

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, model);

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(weightsDir);
    deepNet->removeDropout();
    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_multibranch_full_precision.png"));

    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0),
        Database::All
    );

    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.001);
    DeepNetQAT dnQAT(*deepNet);
    std::cout << "fusing graph" << std::endl;
    dnQAT.fuseQATGraph(*deepNet->getStimuliProvider(), actScalingMode, weightsApproximationMode, weightsApproximationMode, weightsApproximationMode);
    deepNet->exportNetworkFreeParameters("./w_multibranch_q_mnist");

    // TODO Add a tranformation to avoid the automatic scaling to the [0;1] range of N2D2
    deepNet->getStimuliProvider()->addTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, 1.0),
        Database::All
    );

    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_multibranch_quant.png"));

    ASSERT_EQUALS_DELTA(testNetwork(*deepNet, nbTestStimuli), expectedScore, 0.5);

    //when weights rounded:
    /*
    SAT default:
    98.56525139664804 != 98.67
    SAT with symmetric weights:
    98.53503591380687 != 98.55
    */

}
#endif



RUN_TESTS()