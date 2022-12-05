/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

/** @file
 * This program simulates the learning of static images with a convolutional NN.
 * Do not forget to run ./tools/install/install_dataset.py to automatically download and
 *install the stimuli databases.
 *
 * The 101_ObjectCategories database
 * ---------------------------------
 * ./n2d2 $N2D2_MODELS/caltech101-Scherer2010.ini 101_ObjectCategories -pc 30 -test-pc 0
 *-learn 1000000 -log 10000
 *
 * The German Traffic Sign Recognition Benchmark (GTSRB)
 * -----------------------------------------------------
 * ./n2d2 $N2D2_MODELS/GTSRB-Ciresan2012.ini GTSRB -pc 0 -test-pc 0 -learn 1000000 -log
 *10000
 *
 * MNIST
 * -----
 * ./n2d2 $N2D2_MODELS/mnist-29x29-6(5x5)-12(5x5)-100-10.ini mnist -learn 6000000 -log
 *10000
 * or
 * ./n2d2 $N2D2_MODELS/mnist-Scherer2010.ini mnist -learn 6000000 -log 10000
 *
 * MNIST RBF
 * ---------
 * ./n2d2 $N2D2_MODELS/mnist-28x28-rbf.ini mnist -learn 6000000 -log 100000
*/
#include <future>

#include "N2D2.hpp"
#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "Quantizer/QAT/Optimization/DeepNetQAT.hpp"
#include "DrawNet.hpp"
#include "CEnvironment.hpp"
#include "Xnet/Environment.hpp"
#include "Histogram.hpp"
#include "Xnet/NodeEnv.hpp"
#include "RangeStats.hpp"
#include "ScalingMode.hpp"
#include "StimuliProvider.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "Cell/FcCell_Spike.hpp"
#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Solver/SGDSolver.hpp"
#include "Target/TargetROIs.hpp"
#include "Target/TargetBBox.hpp"
#include "Target/TargetScore.hpp"
#include "Target/TargetMatching.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "utils/ProgramOptions.hpp"
#include "Adversarial.hpp"
#include "utils/Helper.hpp"

#ifdef CUDA
#include <cudnn.h>

#include "CudaContext.hpp"
#endif

using namespace N2D2;
using namespace N2D2_HELPER;

int main(int argc, char* argv[]) try
{
#if defined(__GNUC__) && !defined(NDEBUG) && defined(GPROF_INTERRUPT)
    signal(SIGINT, sigUsr1Handler);
#endif

    const Options opt(argc, argv);

#ifdef CUDA
    CudaContext::setDevice(cudaDevice);
#endif

    Network net(opt.seed);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, opt.iniConfig);
    deepNet->initialize();

    if (opt.genConfig) {
        deepNet->saveNetworkParameters();
        std::exit(0);
    }

#ifdef CUDA
#ifdef NVML
    if (opt.banMultiDevice && opt.learnEpoch > 0)
        deepNet->setBanAllowed(opt.banMultiDevice);
#endif
#endif

    Database& database = *deepNet->getDatabase();
    std::cout << "Learning database size: "
              << database.getNbStimuli(Database::Learn) << " images"
              << std::endl;
    std::cout << "Validation database size: "
              << database.getNbStimuli(Database::Validation) << " images"
              << std::endl;
    std::cout << "Testing database size: "
              << database.getNbStimuli(Database::Test) << " images"
              << std::endl;

    // Network topology construction
    SGDSolver::mMaxSteps = (opt.learnEpoch > 0) 
                            ? opt.learnEpoch * database.getNbStimuli(Database::Learn)
                            : opt.learn;
    SGDSolver::mLogSteps = (opt.logEpoch > 0)
                            ? opt.logEpoch * database.getNbStimuli(Database::Learn)
                            : opt.log;

    if (opt.logDbStats) {
        // Log stats
        Utils::createDirectories("dbstats");

        // Stimuli stats
        database.logStats("dbstats/database-size.dat",
                          "dbstats/database-label.dat");
        database.logStats("dbstats/learnset-size.dat",
                          "dbstats/learnset-label.dat", Database::LearnOnly);
        database.logStats("dbstats/validationset-size.dat",
                          "dbstats/validationset-label.dat",
                          Database::ValidationOnly);
        database.logStats("dbstats/testset-size.dat",
                          "dbstats/testset-label.dat", Database::TestOnly);

        // ROIs stats
        database.logROIsStats("dbstats/database-roi-size.dat",
                              "dbstats/database-roi-label.dat");
        database.logROIsStats("dbstats/learnset-roi-size.dat",
                              "dbstats/learnset-roi-label.dat",
                              Database::LearnOnly);
        database.logROIsStats("dbstats/validationset-roi-size.dat",
                              "dbstats/validationset-roi-label.dat",
                              Database::ValidationOnly);
        database.logROIsStats("dbstats/testset-roi-size.dat",
                              "dbstats/testset-roi-label.dat",
                              Database::TestOnly);

        // Multi-channels
        if (!database.getParameter<std::string>("MultiChannelMatch").empty()) {
            database.logMultiChannelStats("dbstats/database-multi-stats.dat");
            database.logMultiChannelStats("dbstats/learnset-multi-stats.dat",
                                          Database::LearnOnly);
            database.logMultiChannelStats("dbstats/validationset-multi-stats.dat",
                                          Database::ValidationOnly);
            database.logMultiChannelStats("dbstats/testset-multi-stats.dat",
                                          Database::TestOnly);
        }
    }

    /**
     * Historically N2D2 normalized integers stimuli in the [0.0;1.0] or [-1.0;1.0] range, 
     * depending on the signess, when loading integer stimuli inside a floating-point Tensor.
     * 
     * Keep this implicit conversion for backward compatibility. 
     * TODO Make plans to eventually remove it.
     */
    if(deepNet->getDatabase() != nullptr && !deepNet->getDatabase()->empty()) {
        const int stimuliCvDepth = deepNet->getDatabase()->getStimuliDepth();
        if(deepNet->getStimuliProvider()->normalizeIntegersStimuli(stimuliCvDepth)) {
            std::cout << Utils::cnotice << "Notice: normalizing the stimuli in the [0;1] range." 
                      << Utils::cdef << std::endl;
        }
    }

    if (!opt.saveTestSet.empty()) {
        StimuliProvider& sp = *deepNet->getStimuliProvider();

        CompositeTransformation trans;
        trans.push_back(sp.getTransformation(Database::Test));
        trans.push_back(sp.getOnTheFlyTransformation(Database::Test));

        database.save(opt.saveTestSet, Database::TestOnly, trans);
    }

    if (!opt.load.empty())
        deepNet->load(opt.load);


    bool afterCalibration = false;
    if (!opt.genExport.empty()) {
        afterCalibration = generateExport(opt, deepNet);
        if (!afterCalibration)
            std::exit(0);
    }

    if(opt.calibOnly>0){
        afterCalibration = calibNetwork(opt, deepNet);
        if (!afterCalibration)
            std::exit(0);
    }

    if (!afterCalibration) {
        logStats(opt, deepNet);
    }

    if (opt.findLr > 0) {
        findLearningRate(opt, deepNet);
        std::exit(0);
    }

    if (opt.learnEpoch > 0) {
        learn_epoch(opt, deepNet);
    }
    else if (opt.learn > 0) {
        learn(opt, deepNet);
    }

    // Pruning testing section
    if (!opt.pruningMethod.empty()) {
        Pruning prune = Pruning(opt.pruningMethod);

        // Options for pruning
        std::vector<float> pruneOpt;
        // Add threshold
        pruneOpt.push_back(0.5f);
        // Apply pruning algorithm to deepNet
        prune.apply(deepNet, pruneOpt);

        // Save network parameters
        deepNet->exportNetworkFreeParameters("weights_pruned");
    }

    if (!afterCalibration) {
        if (opt.learn > 0) {
            // Reload best state after learning
            if (database.getNbStimuli(Database::Validation) > 0)
                deepNet->load("net_state_validation");
            else
                deepNet->load("net_state");
        }
        else if (opt.learnEpoch > 0)
        {
            if (database.getNbStimuli(Database::Validation) > 0){
                deepNet->importNetworkFreeParameters("weights_validation", opt.ignoreNoExist);
            }
            else
                deepNet->importNetworkFreeParameters("weights", opt.ignoreNoExist);
        }
        else if (opt.learnStdp == 0 && opt.load.empty() && opt.weights.empty())
        {
            if (database.getNbStimuli(Database::Validation) > 0){
                if(!opt.pruningMethod.empty()){
                    deepNet->importNetworkFreeParameters("weights_pruned", opt.ignoreNoExist);
                }
                else{
                    deepNet->importNetworkFreeParameters("weights_validation", opt.ignoreNoExist);
                }
            }
            else
                deepNet->importNetworkFreeParameters("weights", opt.ignoreNoExist);
        }
 
        if (opt.fuse)
            deepNet->fuseBatchNorm();
    }
    else if (opt.nbBits > 0) {
        // afterCalibration means that we are trying to simulate export result.
        // In this case, if nbBits > 0, set EstimatedLabelsValueDisplay to false
        // for non-softmax, non-logistic targets
        for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
             = deepNet->getTargets().begin(),
             itTargetsEnd = deepNet->getTargets().end();
             itTargets != itTargetsEnd;
             ++itTargets)
        {
            std::shared_ptr<Cell> cell = (*itTargets)->getCell();
            std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

            if (cell->getType() != SoftmaxCell::Type
                && (!cellFrame || !cellFrame->getActivation()
                    || cellFrame->getActivation()->getType()
                            != std::string("Logistic")))
            {
                (*itTargets)->setParameter<bool>("EstimatedLabelsValueDisplay",
                                                 false);
            }
        }
    }

    if (opt.testIndex >= 0 || opt.testId >= 0) {
        const int label = (opt.testId >= 0)
            ? database.getStimulusLabel(opt.testId)
            : database.getStimulusLabel(Database::Test, opt.testIndex);

        std::cout << "Pattern label ID = " << label << std::endl;

        for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
             = deepNet->getTargets().begin(),
             itTargetsEnd = deepNet->getTargets().end();
             itTargets != itTargetsEnd;
             ++itTargets) {
            std::cout << "Output target = "
                      << (*itTargets)->getLabelTarget(label) << std::endl;
        }
    }

    try
    {
        std::shared_ptr<Cell_Frame_Top> cellFrame = deepNet->getTargetCell<Cell_Frame_Top>();
        if (cellFrame && (opt.learn > 0 || opt.test || opt.learnEpoch > 0)) {
           test(opt, deepNet, afterCalibration);
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error for testing: " << e.what() << std::endl;
        std::cout << "Continue..." << std::endl;
    }

    std::shared_ptr<CEnvironment> cEnv = std::dynamic_pointer_cast
        <CEnvironment>(deepNet->getStimuliProvider());

    if (cEnv) {
        testCStdp(opt, deepNet);
    }

    // Adversararial testing section
    if (!opt.testAdv.empty()) {
        std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();

        if (sp->getAdversarialAttack()->getAttackName() == Adversarial::Attack_T::None) {
            std::stringstream msgStr;
            msgStr << "Please precise the name of your attack "
                   << "in the [sp.Adversarial] section of the ini file";

            throw std::runtime_error(msgStr.str());
        }

        std::ostringstream dirName;
        dirName << "testAdversarial";
        Utils::createDirectories(dirName.str());

        if (opt.testAdv == "Multi")
            sp->getAdversarialAttack()->multiTestAdv(deepNet, dirName.str());
        else if (opt.testAdv == "Solo")
            sp->getAdversarialAttack()->singleTestAdv(deepNet, dirName.str());
        else
            throw std::runtime_error("Unknown adversarial option");
    }

    std::shared_ptr<Environment> env = std::dynamic_pointer_cast
        <Environment>(deepNet->getStimuliProvider());

    if (!env) 
        return 0;

    // Spike-based testing
    Monitor monitorEnv(net);
    monitorEnv.add(env->getNodes());
    Monitor& monitorOut =
        *deepNet->getMonitor(deepNet->getTargetCell()->getName());

    if (opt.learnStdp > 0) {
        learnStdp(opt, deepNet, env, net, monitorEnv, monitorOut);
    }

    //testStdp(opt, deepNet, env, net, monitorEnv, monitorOut);

    return 0;
}
catch (const std::exception& e)
{
    std::cout << "Error: " << e.what() << std::endl;
    return 0;
}
