/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Cell/ConvCell_Frame.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "DeepNet.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "N2D2.hpp"
#include "Target/Target.hpp"
#include "Transformation/FlipTransformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "utils/IniParser.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(DeepNetGenerator, DeepNetGenerator)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const std::string data = "DefaultModel=Frame\n"
                             "ProgramMethod(Spike_RRAM)=Ideal\n"
                             "\n"
                             "[database]\n"
                             "Type=MNIST_IDX_Database\n"
                             "\n"
                             "[env]\n"
                             "SizeX=24\n"
                             "SizeY=24\n"
                             "BatchSize=12\n"
                             "CompositeStimuli=0\n"
                             "ConfigSection=env.config\n"
                             "\n"
                             "[env.config]\n"
                             "StimulusType=JitteredPeriodic\n"
                             "PeriodMin=1,000,000\n"
                             "PeriodMeanMin=10,000,000\n"
                             "PeriodMeanMax=100,000,000,000\n"
                             "PeriodRelStdDev=0.0\n"
                             "\n"
                             "[env.Transformation]\n"
                             "Type=PadCropTransformation\n"
                             "Width=24\n"
                             "Height=24\n"
                             "\n"
                             "[env.OnTheFlyTransformation]\n"
                             "Type=FlipTransformation\n"
                             "RandomHorizontalFlip=1\n"
                             "\n"
                             "[conv1]\n"
                             "Input=env\n"
                             "Type=Conv\n"
                             "KernelWidth=4\n"
                             "KernelHeight=4\n"
                             "NbChannels=16\n"
                             "Stride=2\n"
                             "ActivationFunction=TanhLeCun\n"
                             "\n"
                             "[conv1.Target]\n"
                             "TargetValue=0.9\n"
                             "DefaultValue=-0.9\n"
                             "TopN=4\n";

    UnitTest::FileWriteContent("DeepNetGenerator.in", data);

    Network net;
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, "DeepNetGenerator.in");

    std::shared_ptr<MNIST_IDX_Database> database = std::dynamic_pointer_cast
        <MNIST_IDX_Database>(deepNet->getDatabase());
    ASSERT_EQUALS((bool)database, true);

    const std::shared_ptr<StimuliProvider> env = deepNet->getStimuliProvider();
    ASSERT_EQUALS(env->getSizeX(), 24U);
    ASSERT_EQUALS(env->getSizeY(), 24U);
    ASSERT_EQUALS(env->getBatchSize(), 12U);
    ASSERT_EQUALS(env->getParameter<Environment::StimulusType>("StimulusType"),
                  Environment::JitteredPeriodic);
    ASSERT_EQUALS(env->getParameter<Time_T>("PeriodMin"), 1000000U);
    ASSERT_EQUALS(env->getParameter<Time_T>("PeriodMeanMin"), 10000000U);
    ASSERT_EQUALS(env->getParameter<Time_T>("PeriodMeanMax"), 100000000000U);
    ASSERT_EQUALS(env->getParameter<double>("PeriodRelStdDev"), 0.0);
    ASSERT_EQUALS(env->getNbChannels(), 1U);
    ASSERT_EQUALS(env->getTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(env->getTransformation(Database::Validation).size(), 1U);
    ASSERT_EQUALS(env->getTransformation(Database::Test).size(), 1U);
    ASSERT_EQUALS(env->getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(env->getTransformation(Database::Validation).empty(), false);
    ASSERT_EQUALS(env->getTransformation(Database::Test).empty(), false);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Validation).size(),
                  1U);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Test).size(), 1U);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Learn).empty(),
                  false);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Validation).empty(),
                  false);
    ASSERT_EQUALS(env->getOnTheFlyTransformation(Database::Test).empty(),
                  false);

    const std::shared_ptr<ConvCell_Frame> conv1 = deepNet->getCell
                                                  <ConvCell_Frame>("conv1");
    ASSERT_EQUALS((bool)conv1, true);
    ASSERT_EQUALS(conv1->getKernelWidth(), 4U);
    ASSERT_EQUALS(conv1->getKernelHeight(), 4U);
    ASSERT_EQUALS(conv1->getNbChannels(), 1U);
    ASSERT_EQUALS(conv1->getChannelsWidth(), 24U);
    ASSERT_EQUALS(conv1->getChannelsHeight(), 24U);
    ASSERT_EQUALS(conv1->getNbOutputs(), 16U);
    ASSERT_EQUALS(conv1->getOutputsWidth(), 11U);
    ASSERT_EQUALS(conv1->getOutputsHeight(), 11U);
    ASSERT_EQUALS(conv1->getStrideX(), 2U);
    ASSERT_EQUALS(conv1->getStrideY(), 2U);
    ASSERT_TRUE(conv1->getActivation()->getType() == std::string("Tanh"));

    std::shared_ptr<Target> target = deepNet->getTarget<Target>();
    ASSERT_EQUALS(target->getTargetTopN(), 4U);

    for (unsigned int i = 0; i < 10; ++i) {
        deepNet->getStimuliProvider()->readBatch(Database::Learn, 0);
        Tensor4d<Float_T>& data = deepNet->getStimuliProvider()->getData();

        ASSERT_EQUALS(data.dimX(), 24U);
        ASSERT_EQUALS(data.dimY(), 24U);
        ASSERT_EQUALS(data.dimZ(), 1U);
        ASSERT_EQUALS(data.dimB(), 12U);

        cv::Mat img = data[0];

        std::ostringstream fileName;
        fileName << "DeepNetGenerator_getData(" << i << ").png";
        cv::imwrite(fileName.str(), 255 * img);
    }
}

RUN_TESTS()
