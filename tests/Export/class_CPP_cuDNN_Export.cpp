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

#if defined(CUDA) && !defined(WIN32)

#include <cstdlib>

#include "N2D2.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Transformation/NormalizeTransformation.hpp"
#include "Transformation/ChannelExtractionTransformation.hpp"
#include "Target/Target.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Export/DeepNetExport.hpp"
#include "Cell/ConvCell_Frame.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "Cell/PoolCell_Frame.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "Environment.hpp"
#include "Export/CPP_cuDNN/CPP_cuDNN_PoolCellExport.hpp"
#include "Export/CPP_cuDNN/CPP_cuDNN_FcCellExport.hpp"
#include "Export/CPP_cuDNN/CPP_cuDNN_ConvCellExport.hpp"
#include "Export/CPP_cuDNN/CPP_cuDNN_DeepNetExport.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(CPP_cuDNN_Export, generate)
{
    //REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string data = "DefaultModel=Frame\n"
                             "\n"
                             "[env]\n"
                             "SizeX=48\n"
                             "SizeY=48\n"
                             "BatchSize=1\n"
                             "\n"
                             "[env.Transformation-1]\n"
                             "Type=ChannelExtractionTransformation\n"
                             "CSChannel=Gray\n"
                             "\n"
                             "[env.Transformation-2]\n"
                             "Type=RescaleTransformation\n"
                             "Width=48\n"
                             "Height=48\n"
                             "[env.Transformation-3]\n"
                             "Type=NormalizeTransformation\n"
                             "\n"
                             "[conv1_3x3]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelWidth=3\n"
                             "KernelHeight=3\n"
                             "NbChannels=2\n"
                             "Stride=1\n"
                             "ConfigSection=common.config\n"
                             "\n"
                             "[pool1_3x3]\n"
                             "Input=conv1_3x3\n"
                             "Type=Pool\n"
                             "PoolWidth=3\n"
                             "PoolHeight=3\n"
                             "NbChannels=2\n"
                             "Stride=3\n"
                             "Pooling=Max\n"
                             "Mapping.Size=1\n"
                             "\n"
                             "[conv1_5x5]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelWidth=5\n"
                             "KernelHeight=5\n"
                             "NbChannels=2\n"
                             "Stride=1\n"
                             "Padding=1\n"
                             "ConfigSection=common.config\n"
                             "\n"
                             "[pool1_5x5]\n"
                             "Input=conv1_5x5\n"
                             "Type=Pool\n"
                             "PoolWidth=3\n"
                             "PoolHeight=3\n"
                             "NbChannels=2\n"
                             "Stride=3\n"
                             "Pooling=Max\n"
                             "Mapping.Size=1\n"
                             "\n"
                             "[fc1]\n"
                             "Input=pool1_3x3,pool1_5x5\n"
                             "Type=Fc\n"
                             "NbOutputs=60\n"
                             "ConfigSection=common.config\n"
                             "\n"
                             "[fc2]\n"
                             "Input=fc1\n"
                             "Type=Fc\n"
                             "NbOutputs=4\n"
                             "ConfigSection=common.config\n"
                             "\n"
                             "[fc2.Target]\n"
                             "TargetValue=1.0\n"
                             "DefaultValue=-1.0\n"
                             "\n"
                             "[common.config]\n"
                             "NoBias=0\n"
                             "WeightsSolver.LearningRate=0.01\n"
                             "Solvers.LearningRatePolicy=StepDecay\n"
                             "Solvers.LearningRateStepSize=20000\n"
                             "Solvers.LearningRateDecay=0.996\n"
                             "Solvers.Clamping=1\n";

    UnitTest::FileWriteContent("net_test.ini", data);

    Network net;
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "net_test.ini");

    deepNet->initialize();
    deepNet->importNetworkFreeParameters("tests_data/weights_test");

    std::stringstream exportDir;
    exportDir << "export_CPP_cuDNN_float32";

    DeepNetExport::mEnvDataUnsigned = false;
    CellExport::mPrecision = static_cast<CellExport::Precision>(-32);

    std::string cmd = "rm -rf export_CPP_cuDNN_float32";
    ASSERT_EQUALS(system(cmd.c_str()), 0);

    DeepNetExport::generate(*deepNet, exportDir.str(), "CPP_cuDNN");

    cmd = "mkdir export_CPP_cuDNN_float32/stimuli ";
    ASSERT_EQUALS(system(cmd.c_str()), 0);

    cmd = "cp -r tests_data/stimuli_32f/* export_CPP_cuDNN_float32/stimuli/";
    ASSERT_EQUALS(system(cmd.c_str()), 0);

    cmd =
    "cd export_CPP_cuDNN_float32/ && make OUTPUTFILE=1 NRET=1";
    ASSERT_EQUALS(system(cmd.c_str()), 0);

    REQUIRED(UnitTest::CudaDeviceExists(3));

    cmd = "cd export_CPP_cuDNN_float32/ && ./bin/n2d2_cudnn_test -batch 1";
    ASSERT_EQUALS(system(cmd.c_str()), 0);

    std::ifstream sucess_file("export_CPP_cuDNN_float32/success_rate.txt");
    float success_rate = 0.0;
    if (!sucess_file.good()) {
            throw std::runtime_error
            ("Could not open success file success_rate.txt:");
    }
    sucess_file >> success_rate;
    ASSERT_EQUALS_DELTA(success_rate, 96.3517, 1.e-3);

    cmd = "cd export_CPP_cuDNN_float32/ && ./bin/n2d2_cudnn_test -batch 100";
    ASSERT_EQUALS(system(cmd.c_str()), 0);
    sucess_file >> success_rate;
    ASSERT_EQUALS_DELTA(success_rate, 96.3517, 1.e-3);
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
