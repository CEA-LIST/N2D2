/**
 * (C) Copyright 2019 CEA LIST. All Rights Reserved.
 *  Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
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

#include <cstdlib>

#include "DeepNet.hpp"
#include "Quantizer/QAT/Optimization/DeepNetQAT.hpp"
#include "DeepNetQuantization.hpp"
#include "Histogram.hpp"
#include "N2D2.hpp"
#include "Xnet/Network.hpp"
#include "RangeStats.hpp"
#include "ScalingMode.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "utils/UnitTest.hpp"
#include "DrawNet.hpp"

using namespace N2D2;

static const unsigned int SEED = 7;

float readSuccessRateFile(const std::string& succesRateFile) {
    std::ifstream sucess_file(succesRateFile);
    float success_rate = 0.0;
    if (!sucess_file.good()) {
        throw std::runtime_error("Could not open success file: " + succesRateFile);
    }
    sucess_file >> success_rate;

    return success_rate;
}

TEST_DATASET(CPP_Export_QAT, generate,
        (const std::string& model, const std::string& weightsDir,
         const std::string& exportDir,
         bool unsignedEnv, std::size_t nbTestStimuli,
         const std::string& exportType,
         ScalingMode actScalingMode,
         WeightsApprox weightsApproximationMode,
         double expectedScore),
    std::make_tuple("tests_data/mnist_model/LeNet_4b.ini", "tests_data/mnist_model/weights_LeNet_4b",
                    "export_CPP_Quantization_LeNet4b_FloatPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FLOAT_MULT, WeightsApprox::RINTF, 99.23),
    std::make_tuple("tests_data/mnist_model/LeNet_4b.ini", "tests_data/mnist_model/weights_LeNet_4b",
                    "export_CPP_Quantization_LeNet_4b_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 99.20),
    std::make_tuple("tests_data/mnist_model/LeNet_w4_a8.ini", "tests_data/mnist_model/weights_LeNet_w4_a8",
                    "export_CPP_Quantization_LeNet_w4_a8_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 99.32),
    std::make_tuple("tests_data/mnist_model/LeNet_w5_a8.ini", "tests_data/mnist_model/weights_LeNet_w5_a8",
                    "export_CPP_Quantization_LeNet_w5_a8_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 99.35),
    std::make_tuple("tests_data/mnist_model/LeNet_w3_a4.ini", "tests_data/mnist_model/weights_LeNet_w3_a4",
                    "export_CPP_Quantization_LeNet_w3_a4_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 99.36),
    std::make_tuple("tests_data/mnist_model/LeNet_w1_a8.ini", "tests_data/mnist_model/weights_LeNet_w1_a8",
                    "export_CPP_Quantization_LeNet_w1_a8_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 98.94),
    std::make_tuple("tests_data/mnist_model/LeNet_w1_a4.ini", "tests_data/mnist_model/weights_LeNet_w1_a4",
                    "export_CPP_Quantization_LeNet_w1_a4_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 98.76),
    std::make_tuple("tests_data/mnist_model/LeNet_wMix_a4.ini", "tests_data/mnist_model/weights_LeNet_wMix_a4",
                    "export_CPP_Quantization_LeNet_wMix_a4_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 99.16),
    std::make_tuple("tests_data/mnist_model/LeNet_w1_a2.ini", "tests_data/mnist_model/weights_LeNet_w1_a2",
                    "export_CPP_Quantization_LeNet_w1_a2_FixedPoint/", true, 10000, "CPP_Quantization",
                    ScalingMode::FIXED_MULT32, WeightsApprox::RINTF, 98.56)
                    )
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    // Initialize
    DeepNetExport::mEnvDataUnsigned = unsignedEnv;
    CellExport::mPrecision = static_cast<CellExport::Precision>(8);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, model);

    deepNet->initialize();

    deepNet->importNetworkFreeParameters(weightsDir);
    deepNet->removeDropout();
    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_full_precision.png"));

    deepNet->initialize();

    DeepNetQAT dnQAT(*deepNet);
    dnQAT.fuseQATGraph(*deepNet->getStimuliProvider(), actScalingMode, weightsApproximationMode, weightsApproximationMode, weightsApproximationMode);

    DrawNet::drawGraph(*deepNet, Utils::baseName("graph_fused.png"));

    DeepNetExport::generate(*deepNet, exportDir, exportType);

    deepNet->exportNetworkFreeParameters("weights_export");

#ifndef WIN32
    ASSERT_EQUALS(system(("rm -f " + exportDir + "stimuli/*pgm").c_str()), 0);

    StimuliProviderExport::generate(*deepNet, *deepNet->getStimuliProvider(),
                                    exportDir + "stimuli", exportType, Database::Test,
                                    DeepNetExport::mEnvDataUnsigned, CellExport::mPrecision,
                                    nbTestStimuli);



    ASSERT_EQUALS(system(("rm -rf " + exportDir + "stimuli/stimuli_0").c_str()), 0);

    ASSERT_EQUALS(system(("cd " + exportDir + " && CXXFLAGS=\"-DOUTPUTFILE\" make && "
                          "./bin/run_export").c_str()), 0);

    // Check success rate
    ASSERT_EQUALS_DELTA(readSuccessRateFile(exportDir + "/success_rate.txt"), expectedScore, 0.01);
#endif

}

RUN_TESTS()
