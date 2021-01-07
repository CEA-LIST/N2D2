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
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "utils/UnitTest.hpp"

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

TEST(CPP_Export, generateMemory) {
    const std::string data = "DefaultModel=Frame\n"
                             "\n"
                             "[env]\n"
                             "SizeX=32\n"
                             "SizeY=32\n"
                             "\n"
                             "[conv1.1]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=8\n"
                             "\n"
                             "[conv1.2]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=8\n"
                             "\n"
                             "[conv2]\n"
                             "Input=conv1.1,conv1.2\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=16\n"
                             "\n"
                             "[conv2.Target]\n";

    UnitTest::FileWriteContent("net_test.ini", data);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "net_test.ini");

    deepNet->initialize();

    CPP_DeepNetExport::addBranchesCells(*deepNet);

    bool wrapAroundBuffer = false;
    bool noBranchConcatOpt = false;
    bool includeInputInBuffer = true;
    int memoryAlignment = 1;

    MemoryManager memManager = CPP_DeepNetExport::generateMemory(*deepNet,
        wrapAroundBuffer, noBranchConcatOpt, includeInputInBuffer,
        memoryAlignment);

    ASSERT_EQUALS(memManager.getPeakUsage(), 32*32 + 32*32*16 + 32*32*16);

    const std::shared_ptr<Cell> conv11 = deepNet->getCell("conv1.1");
    ASSERT_EQUALS(memManager.getPlanes(conv11).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->offset, 32*32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->size, 32*32*8);
    ASSERT_TRUE(memManager.getPlanes(conv11).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().size, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().getLimit(), 32*32*8);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().stride, 8);

    const std::shared_ptr<Cell> conv12 = deepNet->getCell("conv1.2");
    ASSERT_EQUALS(memManager.getPlanes(conv12).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->offset, 32*32 + 32*32*8);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->size, 32*32*8);
    ASSERT_TRUE(memManager.getPlanes(conv12).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().size, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().getLimit(), 32*32*8);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().stride, 8);

    ASSERT_TRUE(deepNet->hasCell("conv2_concat"));

    const std::shared_ptr<Cell> conv2_concat = deepNet->getCell("conv2_concat");
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().memSpace->allocated, 2);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().memSpace->offset, 32*32 + 32*32*16);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().memSpace->size, 32*32*16);
    ASSERT_TRUE(memManager.getPlanes(conv2_concat).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().size, 16);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().getLimit(), 32*32*16);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2_concat).back().stride, 16);

    const std::shared_ptr<Cell> conv2 = deepNet->getCell("conv2");
    ASSERT_EQUALS(memManager.getPlanes(conv2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->released, 4);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->size, 32*32*16);
    ASSERT_TRUE(memManager.getPlanes(conv2).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().size, 16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().getLimit(), 32*32*16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().stride, 16);

    memManager.log("memory_mapping.log");
}

TEST_DATASET(CPP_Export,
             generateMemory_noBranchConcatOpt,
             (unsigned int nbOutputs, int memoryAlignment),
             std::make_tuple(8U, 1),
             std::make_tuple(12U, 1),
             std::make_tuple(8U, 8),
             std::make_tuple(12U, 8))
{
    const std::string data = "DefaultModel=Frame\n"
                             "\n"
                             "[env]\n"
                             "SizeX=32\n"
                             "SizeY=32\n"
                             "\n"
                             "[conv1.1]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=8\n"
                             "\n"
                             "[conv1.2]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=" + std::to_string(nbOutputs) + "\n"
                             "\n"
                             "[conv2]\n"
                             "Input=conv1.1,conv1.2\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=16\n"
                             "\n"
                             "[conv2.Target]\n";

    UnitTest::FileWriteContent("net_test.ini", data);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "net_test.ini");

    deepNet->initialize();

    CPP_DeepNetExport::addBranchesCells(*deepNet);

    bool wrapAroundBuffer = false;
    bool noBranchConcatOpt = true;
    bool includeInputInBuffer = true;

    MemoryManager memManager = CPP_DeepNetExport::generateMemory(*deepNet,
        wrapAroundBuffer, noBranchConcatOpt, includeInputInBuffer,
        memoryAlignment);

    const unsigned int aligned = memoryAlignment
        * (unsigned int)std::ceil((8 + nbOutputs) / (double)memoryAlignment);

    ASSERT_EQUALS(memManager.getPeakUsage(), 32*32 + 32*32*aligned + 8 + 32*32*16);

    const std::shared_ptr<Cell> conv11 = deepNet->getCell("conv1.1");
    ASSERT_EQUALS(memManager.getPlanes(conv11).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->offset, 32*32);
    // +8 because offset of 8 for conv12. Those bytes are wasted, but not sure
    // it is worth optimizing... 
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->size, 32*32*aligned + 8);
    ASSERT_TRUE(memManager.getPlanes(conv11).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().size, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().getLimit(), 32*32*aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().stride, aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().getSize(), 32*32*aligned);

    const std::shared_ptr<Cell> conv12 = deepNet->getCell("conv1.2");
    ASSERT_EQUALS(memManager.getPlanes(conv12).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->offset, 32*32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->size, 32*32*aligned + 8);
    ASSERT_TRUE(memManager.getPlanes(conv12).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().offset, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().size, nbOutputs);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().getLimit(), 32*32*aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().stride, aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().getSize(), 32*32*aligned);

    ASSERT_TRUE(!deepNet->hasCell("conv2_concat"));

    const std::shared_ptr<Cell> conv2 = deepNet->getCell("conv2");
    ASSERT_EQUALS(memManager.getPlanes(conv2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->released, 4);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->offset, 32*32 + 32*32*aligned + 8);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->size, 32*32*16);
    ASSERT_TRUE(memManager.getPlanes(conv2).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().size, 16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().getLimit(), 32*32*16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().stride, 16);

    memManager.log("memory_mapping_noBranchConcatOpt.log");
}

TEST_DATASET(CPP_Export,
             generateMemory_wrapAroundBuffer_noBranchConcatOpt,
             (unsigned int nbOutputs, int memoryAlignment),
             std::make_tuple(8U, 1),
             std::make_tuple(12U, 1),
             std::make_tuple(8U, 8),
             std::make_tuple(12U, 8))
{
    const std::string data = "DefaultModel=Frame\n"
                             "\n"
                             "[env]\n"
                             "SizeX=32\n"
                             "SizeY=32\n"
                             "\n"
                             "[conv1.1]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=8\n"
                             "\n"
                             "[conv1.2]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=" + std::to_string(nbOutputs) + "\n"
                             "\n"
                             "[conv2]\n"
                             "Input=conv1.1,conv1.2\n"
                             "Type=Conv\n"
                             "KernelDims=1 1\n"
                             "NbOutputs=16\n"
                             "\n"
                             "[conv2.Target]\n";

    UnitTest::FileWriteContent("net_test.ini", data);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "net_test.ini");

    deepNet->initialize();

    CPP_DeepNetExport::addBranchesCells(*deepNet);

    bool wrapAroundBuffer = true;
    bool noBranchConcatOpt = true;
    bool includeInputInBuffer = true;

    MemoryManager memManager = CPP_DeepNetExport::generateMemory(*deepNet,
        wrapAroundBuffer, noBranchConcatOpt, includeInputInBuffer,
        memoryAlignment);

    const unsigned int aligned = memoryAlignment
        * (unsigned int)std::ceil((8 + nbOutputs) / (double)memoryAlignment);
    const unsigned int marginCorrection
        = (nbOutputs == 12 && memoryAlignment == 1) ? 32*4 :
          (nbOutputs == 12 && memoryAlignment == 8) ? 32*8 : 0;

    ASSERT_EQUALS(memManager.getPeakUsage(), 32*32 + 32*32*aligned + 32*16 + marginCorrection);

    const std::shared_ptr<Cell> conv11 = deepNet->getCell("conv1.1");
    ASSERT_EQUALS(memManager.getPlanes(conv11).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->offset, 32*32);
    // +32*16 because of wrapping
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().memSpace->size, 32*32*aligned + 32*16 + marginCorrection);
    ASSERT_TRUE(memManager.getPlanes(conv11).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().size, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().getLimit(), 32*32*aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().stride, aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv11).back().getSize(), 32*32*aligned);

    const std::shared_ptr<Cell> conv12 = deepNet->getCell("conv1.2");
    ASSERT_EQUALS(memManager.getPlanes(conv12).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->offset, 32*32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().memSpace->size, 32*32*aligned + 32*16 + marginCorrection);
    ASSERT_TRUE(memManager.getPlanes(conv12).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().offset, 8);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().size, nbOutputs);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().getLimit(), 32*32*aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().stride, aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv12).back().getSize(), 32*32*aligned);

    ASSERT_TRUE(!deepNet->hasCell("conv2_concat"));

    const std::shared_ptr<Cell> conv2 = deepNet->getCell("conv2");
    ASSERT_EQUALS(memManager.getPlanes(conv2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->released, 3);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->offset, 32*32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().memSpace->size, 32*32*aligned + 32*16 + marginCorrection);
    ASSERT_TRUE(memManager.getPlanes(conv2).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().offset, 32*32*aligned);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().size, 16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().getLimit(), 32*16);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().length, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().count, 32);
    ASSERT_EQUALS(memManager.getPlanes(conv2).back().stride, 16);

    memManager.log("memory_mapping_wrapAroundBuffer_noBranchConcatOpt.log");
}

TEST_DATASET(CPP_Export,
             generateMemory_ResNet18,
             (bool wrapAroundBuffer, MemoryManager::OptimizeStrategy strategy, size_t peakUsage),
             std::make_tuple(false, MemoryManager::None, 224*224*3 + 112*112*64 + 55*55*64),
             std::make_tuple(true, MemoryManager::None, 112*112*64 + 112*64*2 + 55*55*64 + 55*64 + 55*55*64),
             std::make_tuple(false, MemoryManager::OptimizeMaxHoleMaxLifetimeFirst, 224*224*3 + 112*112*64 + 55*55*64),
             std::make_tuple(true, MemoryManager::OptimizeMaxHoleMaxLifetimeFirst, 112*112*64 + 112*64*2))
{
    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "tests_data/ResNet-18.ini");

    deepNet->initialize();

    CPP_DeepNetExport::addBranchesCells(*deepNet);

    bool includeInputInBuffer = true;
    bool noBranchConcatOpt = true;  // not useful for ResNet18
    int memoryAlignment = 1;

    MemoryManager memManager = CPP_DeepNetExport::generateMemory(*deepNet,
        wrapAroundBuffer, noBranchConcatOpt, includeInputInBuffer,
        memoryAlignment);

    memManager.optimize(strategy);

    ASSERT_EQUALS(memManager.getPeakUsage(), peakUsage);

    std::ostringstream logFile;
    logFile << "memory_mapping_ResNet18_"
        << ((wrapAroundBuffer) ? "wrapAround_" : "") << strategy << ".log";

    memManager.log(logFile.str());
}

TEST(CPP_Export_32f, generate) {
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string testDataDir = "tests_data/mnist_model/";
    const std::string exportDir = "export_CPP_float32/";
    const std::string exportType = "CPP";
    const std::size_t nbTestStimuli = 200;


    // Initialize
    DeepNetExport::mEnvDataUnsigned = true;
    CellExport::mPrecision = static_cast<CellExport::Precision>(-32);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, testDataDir + "model_wo_softmax.ini");

    deepNet->initialize();
    deepNet->importNetworkFreeParameters(testDataDir + "weights");


    // Export
    DeepNetExport::generate(*deepNet, exportDir, exportType);

    ASSERT_EQUALS(system(("rm -f " + exportDir + "stimuli/*pgm").c_str()), 0);
    StimuliProviderExport::generate(*deepNet, *deepNet->getStimuliProvider(), 
                                    exportDir + "stimuli", exportType, Database::Test, 
                                    DeepNetExport::mEnvDataUnsigned, CellExport::mPrecision, 
                                    nbTestStimuli);

    ASSERT_EQUALS(system(("cd " + exportDir + " && CXXFLAGS=\"-DOUTPUTFILE\" make && ./run_export").c_str()), 0);


    // Check success rate
    ASSERT_EQUALS_DELTA(readSuccessRateFile(exportDir + "/success_rate.txt"), 96.00, 0.01);
}

TEST(CPP_Export_8i, generate) {
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string testDataDir = "tests_data/mnist_model/";
    const std::string exportDir = "export_CPP_int8/";
    const std::string exportType = "CPP";
    const std::size_t nbTestStimuli = 200;


    // Initialize
    DeepNetExport::mEnvDataUnsigned = true;
    CellExport::mPrecision = static_cast<CellExport::Precision>(8);

    Network net(SEED);
    std::shared_ptr<DeepNet> deepNet = DeepNetGenerator::generate(net, testDataDir + "model_wo_softmax.ini");

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

    ASSERT_EQUALS(system(("cd " + exportDir + " && CXXFLAGS=\"-DOUTPUTFILE\" make && "
                          "./run_export").c_str()), 0);


    // Check success rate
    ASSERT_EQUALS_DELTA(readSuccessRateFile(exportDir + "/success_rate.txt"), 96.00, 0.01);
#endif
}

RUN_TESTS()
