/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <cstdlib>

#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "Histogram.hpp"
#include "N2D2.hpp"
#include "Xnet/Network.hpp"
#include "RangeStats.hpp"
#include "ScalingMode.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

static const unsigned int SEED = 7;

float readSuccessRateFile(const std::string& succesRateFile) {
    std::ifstream sucess_file(succesRateFile);
    float success_rate = 0.0;
    if (!sucess_file.good()) {
        throw std::runtime_error("Could not open success file success_rate.txt:");
    }
    sucess_file >> success_rate;

    return success_rate;
}

TEST(C_Export32f, generate) {
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string testDataDir = "tests_data/mnist_model/";
    const std::string exportDir = "export_C_float32/";
    const std::string exportType = "C";
    const std::size_t nbTestStimuli = 200;


    // Initialize
    DeepNetExport::mEnvDataUnsigned = true;
    CellExport::mPrecision = static_cast<CellExport::Precision>(-32);

    Network net(SEED,false);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, testDataDir + "model.ini");

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(testDataDir + "weights");


    // Export
    DeepNetExport::generate(*deepNet, exportDir, exportType);

#ifndef WIN32
    ASSERT_EQUALS(system(("rm -f " + exportDir + "stimuli/*pgm").c_str()), 0);
    StimuliProviderExport::generate(*deepNet, *deepNet->getStimuliProvider(), 
                                    exportDir + "stimuli", exportType, Database::Test, 
                                    DeepNetExport::mEnvDataUnsigned, CellExport::mPrecision, 
                                    nbTestStimuli);

    ASSERT_EQUALS(system(("cd " + exportDir + " && make OUTPUTFILE=1 && ./bin/n2d2_test").c_str()), 0);


    // Check success rate
    ASSERT_EQUALS_DELTA(readSuccessRateFile(exportDir + "success_rate.txt"), 96.00, 0.01);
#endif
}

TEST(C_Export8i, generate) {
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string testDataDir = "tests_data/mnist_model/";
    const std::string exportDir = "export_C_int8/";
    const std::string exportType = "C";
    const std::size_t nbTestStimuli = 200;


    // Initialize
    DeepNetExport::mEnvDataUnsigned = true;
    CellExport::mPrecision = static_cast<CellExport::Precision>(8);

    Network net(SEED,false);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, testDataDir + "model.ini");

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(testDataDir + "weights");


    // Quantize
    std::unordered_map<std::string, Histogram> emptyOutputsHistogram;
    std::unordered_map<std::string, RangeStats> outputsRange;
    RangeStats::loadOutputsRange(testDataDir + "outputs_range.bin", outputsRange);


    DeepNetQuantization dnQuantization(*deepNet);
    dnQuantization.quantizeNetwork(emptyOutputsHistogram, outputsRange,
                                   CellExport::mPrecision, ClippingMode::NONE, 
                                   ScalingMode::SINGLE_SHIFT, false);


    // Export
    DeepNetExport::generate(*deepNet, exportDir, exportType);

#ifndef WIN32
    ASSERT_EQUALS(system(("rm -f " + exportDir + "stimuli/*pgm").c_str()), 0);
    StimuliProviderExport::generate(*deepNet, *deepNet->getStimuliProvider(), 
                                    exportDir + "stimuli", exportType, Database::Test, 
                                    DeepNetExport::mEnvDataUnsigned, CellExport::mPrecision, 
                                    nbTestStimuli);

    ASSERT_EQUALS(system(("cd " + exportDir + " && make OUTPUTFILE=1 && ./bin/n2d2_test").c_str()), 0);


    // Check success rate
    ASSERT_EQUALS_DELTA(readSuccessRateFile(exportDir + "success_rate.txt"), 96.00, 0.01);
#endif
}

RUN_TESTS()


