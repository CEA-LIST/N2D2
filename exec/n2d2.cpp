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
 * Do not forget to run ./tools/install_stimuli.py to automatically download and
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

#include "N2D2.hpp"

#include "DeepNet.hpp"
#include "DrawNet.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "Cell/FcCell_Spike.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Target/TargetROIs.hpp"
#include "Target/TargetScore.hpp"

#ifdef CUDA
#include "CudaContext.hpp"
#endif

using namespace N2D2;

#ifdef CUDA
unsigned int cudaDevice = 0;
#endif

void learnThreadWrapper(const std::shared_ptr<DeepNet>& deepNet,
                        std::vector<std::pair<std::string, double> >* timings
                            = NULL)
{
#ifdef CUDA
    CudaContext::setDevice(cudaDevice);
#endif

    deepNet->learn(timings);
}

void validationThreadWrapper(const std::shared_ptr<DeepNet>& deepNet)
{
#ifdef CUDA
    CudaContext::setDevice(cudaDevice);
#endif

    deepNet->test(Database::Validation);
}

int main(int argc, char* argv[]) try
{
    // Program command line options
    ProgramOptions opts(argc, argv);
    unsigned int seed
        = opts.parse("-seed", 0U, "N2D2 random seed (0 = time based)");
    const unsigned int log
        = opts.parse("-log", 1000U, "number of steps between logs");
    const unsigned int report
        = opts.parse("-report", 100U, "number of steps between reportings");
    const unsigned int learn
        = opts.parse("-learn", 0U, "number of backprop learning steps");
    const unsigned int stopValid = opts.parse(
        "-stop-valid", 0U, "max. number of successive lower score validation");
    const bool test = opts.parse("-test", "perform testing");
    const bool bench = opts.parse("-bench", "learning speed benchmarking");
    const unsigned int learnStdp
        = opts.parse("-learn-stdp", 0U, "number of STDP learning steps");
    const unsigned int avgWindow
        = opts.parse("-ws",
                     10000U,
                     "average window to compute success rate during learning");
    const int testIdx
        = opts.parse("-test-idx", -1, "test a single specific frame index");
    const bool check
        = opts.parse("-check", "enable gradient computation checking");
    const bool logOutputs
        = opts.parse("-log-outputs", "log layers outputs for the first "
                     "stimulus");
    const bool genConfig
        = opts.parse("-cfg", "save base configuration and exit");
    const std::string genExport
        = opts.parse<std::string>("-export", "", "generate an export and exit");
    const int nbBits
        = opts.parse("-nbbits", 8, "number of bits per weight for exports");
    const bool envDataUnsigned
        = opts.parse("-uenv", "unsigned env data for exports");
    const double timeStep
        = opts.parse("-ts", 0.1, "timestep for clock-based simulations (ns)");
    const std::string weights = opts.parse<std::string>(
        "-w",
        "",
        "import initial weights from a specific location for the learning");
    const bool noDB = opts.parse("-no-db-export", "disable database export");
    N2D2::DeepNetExport::mExportParameters = opts.parse<std::string>(
        "-export-parameters", "", "parameters for export");
#ifdef CUDA
    cudaDevice = opts.parse("-dev", 0, "CUDA device ID");
#endif

    const std::string iniConfig
        = opts.grab<std::string>("<net>", "network config file (INI)");
    opts.done();

#ifdef CUDA
    CudaContext::setDevice(cudaDevice);
#endif

    // Ensures that the seed is the same for the test than for the learning (to
    // avoid including learned stimuli in the test set)
    if (seed == 0 && learn == 0)
        seed = Network::readSeed("seed.dat");

    // Network topology construction
    Network net(seed);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, iniConfig);
    deepNet->initialize();

    if (genConfig) {
        deepNet->saveNetworkParameters();
        std::exit(0);
    }

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
    database.logROIsStats("database-roi-size.dat", "database-roi-label.dat");
    database.logROIsStats(
        "testset-roi-size.dat", "testset-roi-label.dat", Database::TestOnly);

    StimuliProvider& sp = *deepNet->getStimuliProvider();

    if (!genExport.empty()) {
        if (!weights.empty())
            deepNet->importNetworkFreeParameters(weights);
        else if (database.getNbStimuli(Database::Validation) > 0)
            deepNet->importNetworkFreeParameters("weights_validation");
        else
            deepNet->importNetworkFreeParameters("weights");

        std::stringstream exportDir;
        exportDir << "export_" << genExport << "_"
                  << ((nbBits > 0) ? "int" : "float") << std::abs(nbBits);

        DeepNetExport::mEnvDataUnsigned = envDataUnsigned;
        CellExport::mPrecision = static_cast<CellExport::Precision>(nbBits);

        DeepNetExport::generate(*deepNet, exportDir.str(), genExport);

        if (!noDB)
            StimuliProviderExport::generate(*deepNet->getStimuliProvider(),
                                            exportDir.str() + "/stimuli",
                                            Database::Test,
                                            deepNet.get());
        std::exit(0);
    }

    DrawNet::draw(*deepNet, Utils::baseName(iniConfig) + ".svg");
    deepNet->logStats("stats");
    deepNet->logLabelsMapping("labels_mapping.log");
    deepNet->logLabelsLegend("labels_legend.png");

    if (!weights.empty())
        deepNet->importNetworkFreeParameters(weights, true);

    if (check) {
        std::cout << "Checking gradient computation..." << std::endl;
        deepNet->checkGradient(1.0e-3, 1.0e-3);
    }

    // Reconstruct some frames to see the pre-processing
    Utils::createDirectories("frames");

    for (unsigned int i = 0,
                      size
                      = std::min(10U, database.getNbStimuli(Database::Learn));
         i < size;
         ++i) {
        std::ostringstream fileName;
        fileName << "frames/frame_" << i << ".dat";

        sp.readStimulus(Database::Learn, i);
        StimuliProvider::logData(fileName.str(), sp.getData()[0]);

        const Tensor3d<int> labelsData = sp.getLabelsData()[0];

        if (labelsData.dimX() > 1 || labelsData.dimY() > 1) {
            fileName.str(std::string());
            fileName << "frames/frame_" << i << "_label.dat";

            Tensor3d<Float_T> displayLabelsData(labelsData.dimX(),
                                                labelsData.dimY(),
                                                labelsData.dimZ());

            for (unsigned int index = 0; index < labelsData.size(); ++index)
                displayLabelsData(index) = labelsData(index);

            StimuliProvider::logData(fileName.str(), displayLabelsData);
        }
    }

    for (unsigned int i = 0,
                      size
                      = std::min(10U, database.getNbStimuli(Database::Test));
         i < size;
         ++i) {
        std::ostringstream fileName;
        fileName << "frames/test_frame_" << i << ".dat";

        sp.readStimulus(Database::Test, i);
        StimuliProvider::logData(fileName.str(), sp.getData()[0]);
    }

    if (learn > 0) {
        deepNet->exportNetworkFreeParameters("weights_init");

        std::chrono::high_resolution_clock::time_point startTime
            = std::chrono::high_resolution_clock::now();
        double minTimeElapsed = 0.0;
        unsigned int nextLog = log;
        unsigned int nextReport = report;
        unsigned int nbNoValid = 0;

        const unsigned int batchSize = sp.getBatchSize();
        const unsigned int nbBatch = std::ceil(learn / (double)batchSize);
        const unsigned int avgBatchWindow = avgWindow / (double)batchSize;

        sp.readRandomBatch(Database::Learn);

        std::vector<std::pair<std::string, double> > timings, cumTimings;

        for (unsigned int b = 0; b < nbBatch; ++b) {
            const unsigned int i = b * batchSize;

            sp.synchronize();
            std::thread learnThread(learnThreadWrapper,
                                    deepNet,
                                    (bench) ? &timings : NULL);

            sp.future();
            sp.readRandomBatch(Database::Learn);
            learnThread.join();

            if (logOutputs && i == 0) {
                std::cout << "First stimulus ID: " << sp.getBatch()[0]
                          << std::endl;
                std::cout << "First stimulus label: "
                          << sp.getLabelsData()[0](0) << std::endl;
                deepNet->logOutputs("outputs_init");
                deepNet->logDiffInputs("diffinputs_init");
            }

            if (bench) {
                if (!cumTimings.empty()) {
                    std::transform(timings.begin(),
                                   timings.end(),
                                   cumTimings.begin(),
                                   cumTimings.begin(),
                                   Utils::PairOp<std::string,
                                                 double,
                                                 Utils::Left<std::string>,
                                                 std::plus<double> >());
                } else
                    cumTimings = timings;
            }

            if (i >= nextReport || b == nbBatch - 1) {
                nextReport += report;

                std::chrono::high_resolution_clock::time_point curTime
                    = std::chrono::high_resolution_clock::now();

                std::ios::fmtflags f(std::cout.flags());

                std::cout << "\rLearning #" << std::setw(8) << std::left << i
                          << "   ";
                std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                          << std::right;

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                         itTargets = deepNet->getTargets().begin(),
                         itTargetsEnd = deepNet->getTargets().end();
                     itTargets != itTargetsEnd;
                     ++itTargets) {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                  Database::Learn,
                                                  avgBatchWindow)) << "% ";
                }

                const double timeElapsed = std::chrono::duration_cast
                                           <std::chrono::duration<double> >(
                                               curTime - startTime).count();

                if (minTimeElapsed == 0.0 || minTimeElapsed < timeElapsed)
                    minTimeElapsed = timeElapsed;

                std::cout << "at " << std::setw(7) << std::fixed
                          << std::setprecision(2) << (report / timeElapsed)
                          << " p./s"
                             " (" << std::setw(7) << std::fixed
                          << std::setprecision(0)
                          << 60.0 * (report / timeElapsed)
                          << " p./min)         " << std::setprecision(4)
                          << std::flush;

                std::cout.flags(f);

                startTime = std::chrono::high_resolution_clock::now();
            }

            if (i >= nextLog || b == nbBatch - 1) {
                nextLog += log;

                std::cout << std::endl;

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                         itTargets = deepNet->getTargets().begin(),
                         itTargetsEnd = deepNet->getTargets().end();
                     itTargets != itTargetsEnd;
                     ++itTargets) {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target) {
                        target->logSuccess(
                            "learning", Database::Learn, avgBatchWindow);
                        // target->logTopNSuccess("learning", Database::Learn,
                        // avgBatchWindow);
                    }
                }

                if (bench) {
                    for (std::vector<std::pair<std::string, double> >::iterator
                         it = cumTimings.begin(),
                         itEnd = cumTimings.end();
                         it != itEnd;
                         ++it) {
                        (*it).second /= (i + batchSize);
                    }
                    Utils::createDirectories("timings");

                    deepNet->logTimings("timings/learning_timings.dat", cumTimings);
                }

                deepNet->logEstimatedLabels("learning");
                deepNet->log("learning", Database::Learn);
                deepNet->clear(Database::Learn);

                if (database.getNbStimuli(Database::Validation) > 0) {
                    const unsigned int nbValid
                        = database.getNbStimuli(Database::Validation);
                    const unsigned int nbBatchValid
                        = std::ceil(nbValid / (double)batchSize);

                    std::cout << "Validation" << std::flush;
                    unsigned int progress = 0, progressPrev = 0;

                    sp.readBatch(Database::Validation, 0);

                    for (unsigned int bv = 0; bv < nbBatchValid; ++bv) {
                        const unsigned int k = bv * batchSize;

                        sp.synchronize();
                        std::thread validationThread(validationThreadWrapper,
                                                     deepNet);

                        sp.future();
                        sp.readBatch(Database::Validation, k);
                        validationThread.join();

                        // Progress bar
                        progress
                            = (unsigned int)(20.0 * bv / (double)nbBatchValid);

                        if (progress > progressPrev) {
                            std::cout << std::string(progress - progressPrev,
                                                     '.') << std::flush;
                            progressPrev = progress;
                        }
                    }

                    std::cout << std::endl;

                    sp.readRandomBatch(Database::Learn);

                    for (std::vector<std::shared_ptr<Target> >::const_iterator
                             itTargets = deepNet->getTargets().begin(),
                             itTargetsEnd = deepNet->getTargets().end();
                         itTargets != itTargetsEnd;
                         ++itTargets) {
                        std::shared_ptr<TargetScore> target
                            = std::dynamic_pointer_cast
                            <TargetScore>(*itTargets);

                        if (!target)
                            continue;

                        if (target->newValidationScore(
                                target->getAverageSuccess(
                                    Database::Validation))) {
                            nbNoValid = 0;

                            std::cout << "\n+++ BEST validation score: "
                                      << (100.0
                                          * target->getMaxValidationScore())
                                      << "%\n" << std::endl;

                            deepNet->log("validation", Database::Validation);
                            deepNet->exportNetworkFreeParameters(
                                "weights_validation");
                        } else {
                            std::cout << "\n--- LOWER validation score: "
                                      << (100.0
                                          * target->getLastValidationScore())
                                      << "% (best was "
                                      << (100.0
                                          * target->getMaxValidationScore())
                                      << "%)\n" << std::endl;

                            ++nbNoValid;

                            if (stopValid > 0 && nbNoValid >= stopValid) {
                                std::cout
                                    << "\n--- Validation did not improve after "
                                    << stopValid << " steps\n" << std::endl;
                                std::cout << "\n--- STOPPING THE LEARNING\n"
                                          << std::endl;
                                break;
                            }
                        }

                        target->newValidationTopNScore(
                            target->getAverageTopNSuccess(
                                Database::Validation)); // Top-N accuracy
                        target->logSuccess(
                            "validation", Database::Validation, avgBatchWindow);
                        target->logTopNSuccess(
                            "validation",
                            Database::Validation,
                            avgBatchWindow); // Top-N accuracy
                        target->clearSuccess(Database::Validation);
                    }

                    deepNet->clear(Database::Validation);
                } else
                    deepNet->exportNetworkFreeParameters("weights");
            }
        }

        deepNet->logFreeParameters("kernels");

        // We are still in future batch, need to synchronize for the following
        sp.synchronize();
    }

    if (weights.empty() || learn > 0) {
        if (database.getNbStimuli(Database::Validation) > 0)
            deepNet->importNetworkFreeParameters("weights_validation");
        else
            deepNet->importNetworkFreeParameters("weights");
    }

    if (testIdx >= 0) {
        const int label = database.getStimulusLabel(Database::Test, testIdx);

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
        std::shared_ptr<Cell_Frame_Top> cellFrame = deepNet->getTargetCell
                                                    <Cell_Frame_Top>();

        std::map<std::string, DeepNet::RangeStats> outputsRange;

        if (cellFrame && (learn > 0 || test)) {
            std::vector<std::pair<std::string, double> > timings, cumTimings;

            // Static testing
            unsigned int nextLog = log;
            unsigned int nextReport = report;

            const unsigned int nbTest
                = (testIdx >= 0) ? 1 : database.getNbStimuli(Database::Test);
            const unsigned int batchSize = sp.getBatchSize();
            const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

            for (unsigned int b = 0; b < nbBatch; ++b) {
                const unsigned int i = b * batchSize;
                const unsigned int idx = (testIdx >= 0) ? testIdx : i;

                sp.readBatch(Database::Test, idx);
                deepNet->test(Database::Test, &timings);
                deepNet->reportOutputsRange(outputsRange);
                deepNet->logEstimatedLabels("test");

                if (logOutputs && i == 0) {
                    std::cout << "First stimulus ID: " << sp.getBatch()[0]
                              << std::endl;
                    std::cout << "First stimulus label: "
                              << sp.getLabelsData()[0](0) << std::endl;
                    deepNet->logOutputs("outputs_test");
                }

                if (!cumTimings.empty()) {
                    std::transform(timings.begin(),
                                   timings.end(),
                                   cumTimings.begin(),
                                   cumTimings.begin(),
                                   Utils::PairOp<std::string,
                                                 double,
                                                 Utils::Left<std::string>,
                                                 std::plus<double> >());
                } else
                    cumTimings = timings;

                if (i >= nextReport || b == nbBatch - 1) {
                    nextReport += report;
                    std::cout << "Testing #" << idx << "   ";

                    for (std::vector<std::shared_ptr<Target> >::const_iterator
                             itTargets = deepNet->getTargets().begin(),
                             itTargetsEnd = deepNet->getTargets().end();
                         itTargets != itTargetsEnd;
                         ++itTargets) {
                        std::shared_ptr<TargetScore> target
                            = std::dynamic_pointer_cast
                            <TargetScore>(*itTargets);

                        if (target)
                            std::cout << (100.0 * target->getAverageSuccess(
                                                      Database::Test)) << "% ";
                    }

                    std::cout << std::endl;
                }

                if (i >= nextLog || b == nbBatch - 1) {
                    nextLog += report;

                    for (std::vector<std::shared_ptr<Target> >::const_iterator
                             itTargets = deepNet->getTargets().begin(),
                             itTargetsEnd = deepNet->getTargets().end();
                         itTargets != itTargetsEnd;
                         ++itTargets) {
                        std::shared_ptr<TargetScore> target
                            = std::dynamic_pointer_cast
                            <TargetScore>(*itTargets);

                        if (target) {
                            target->logSuccess("test", Database::Test);
                            target->logTopNSuccess("test", Database::Test);
                        }
                    }
                }
            }

            if (nbTest > 0) {
                deepNet->log("test", Database::Test);
                for (std::vector<std::pair<std::string, double> >::iterator it
                     = cumTimings.begin(),
                     itEnd = cumTimings.end();
                     it != itEnd;
                     ++it) {
                    (*it).second /= nbTest;
                }

                Utils::createDirectories("timings");

                deepNet->logTimings("timings/inference_timings.dat", cumTimings);

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                         itTargets = deepNet->getTargets().begin(),
                         itTargetsEnd = deepNet->getTargets().end();
                     itTargets != itTargetsEnd;
                     ++itTargets) {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target) {
                        std::cout << "Final recognition rate: "
                                  << (100.0 * target->getAverageSuccess(
                                                  Database::Test))
                                  << "%"
                                     "    (error rate: "
                                  << 100.0 * (1.0 - target->getAverageSuccess(
                                                        Database::Test)) << "%)"
                                  << std::endl;
                    }
                }
            }

            deepNet->logOutputsRange("test_outputs_range.dat", outputsRange);
            deepNet->normalizeOutputsRange(outputsRange, 0.25);
            deepNet->exportNetworkFreeParameters("weights_normalized");
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
        unsigned int nextLog = log;
        unsigned int nextReport = report;

        const unsigned int nbTest
            = (testIdx >= 0) ? 1 : database.getNbStimuli(Database::Test);
        const unsigned int batchSize = sp.getBatchSize();
        const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

        for (unsigned int b = 0; b < nbBatch; ++b) {
            const unsigned int i = b * batchSize;
            const unsigned int idx = (testIdx >= 0) ? testIdx : i;

            sp.readBatch(Database::Test, idx);
            deepNet->cTicks(0, 1 * TimeUs, (Time_T)(timeStep * TimeNs));
            deepNet->cTargetsProcess(Database::Test);

            if (i >= nextReport || b == nbBatch - 1) {
                nextReport += report;
                std::cout << "Testing #" << idx << "   ";

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                         itTargets = deepNet->getTargets().begin(),
                         itTargetsEnd = deepNet->getTargets().end();
                     itTargets != itTargetsEnd;
                     ++itTargets) {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                  Database::Test)) << "% ";
                }

                std::cout << std::endl;
            }

            if (i >= nextLog || b == nbBatch - 1) {
                nextLog += report;

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                         itTargets = deepNet->getTargets().begin(),
                         itTargetsEnd = deepNet->getTargets().end();
                     itTargets != itTargetsEnd;
                     ++itTargets) {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target) {
                        target->logSuccess("test", Database::Test);
                        target->logTopNSuccess("test", Database::Test);
                    }
                }
            }

            deepNet->cReset();
        }

        if (nbTest > 0) {
            deepNet->log("test", Database::Test);

            for (std::vector<std::shared_ptr<Target> >::const_iterator itTargets
                 = deepNet->getTargets().begin(),
                 itTargetsEnd = deepNet->getTargets().end();
                 itTargets != itTargetsEnd;
                 ++itTargets) {
                std::shared_ptr<TargetScore> target = std::dynamic_pointer_cast
                    <TargetScore>(*itTargets);

                if (target) {
                    std::cout << "Final recognition rate: "
                              << (100.0
                                  * target->getAverageSuccess(Database::Test))
                              << "%"
                                 "    (error rate: "
                              << 100.0 * (1.0 - target->getAverageSuccess(
                                                    Database::Test)) << "%)"
                              << std::endl;
                }
            }
        }
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

    if (learnStdp > 0) {
        Utils::createDirectories("weights_stdp");

        for (unsigned int i = 0; i < learnStdp; ++i) {
            const bool logStep = (i + 1) % log == 0 || i == learnStdp - 1;

            // Run
            std::cout << "Learn from " << i* TimeUs / (double)TimeS << "s to "
                      << (i + 1) * TimeUs / (double)TimeS << "s..."
                      << std::endl;

            const Database::StimulusID id
                = env->readRandomStimulus(Database::Learn);
            env->propagate(i * TimeUs, (i + 1) * TimeUs);
            net.run((i + 1) * TimeUs);

            // Check activity
            std::vector<std::pair<std::string, unsigned int> > activity
                = deepNet->update(
                    logStep, i * TimeUs, (i + 1) * TimeUs, logStep);

            monitorEnv.update();

            unsigned int sumActivity = 0;

            for (std::vector
                 <std::pair<std::string, unsigned int> >::const_iterator it
                 = activity.begin(),
                 itEnd = activity.end();
                 it != itEnd;
                 ++it) {
                std::cout << "   " << (*it).first << ": " << (*it).second;
                sumActivity += (*it).second;
            }

            std::cout << "   (total: " << sumActivity << ")" << std::endl;

            // Check success
            std::shared_ptr<FcCell_Spike> fcCell = deepNet->getTargetCell
                                                   <FcCell_Spike>();
            const unsigned int outputTarget
                = deepNet->getTarget()->getLabelTarget(
                    database.getStimulusLabel(id));

            const bool success = monitorOut.checkLearningResponse(
                outputTarget, fcCell->getBestResponseId());
            std::cout << "   #" << id << " / success rate: "
                      << (monitorOut.getSuccessRate(avgWindow) * 100) << "%";

            if (!success)
                std::cout << "   [FAIL]";

            std::cout << std::endl;

            if (logStep) {
                deepNet->exportNetworkFreeParameters("weights_stdp");
                deepNet->logFreeParameters("kernels");

                monitorOut.logSuccessRate(
                    "learning_success_spike.dat", avgWindow, true);
                deepNet->logSpikeStats("stats_learning_spike", i + 1);
            }

            net.reset((i + 1) * TimeUs);
        }

        deepNet->exportNetworkFreeParameters("./");

        net.reset();

        deepNet->clearAll();
        deepNet->setCellsParameter("EnableStdp", false);
    }

    Utils::createDirectories("stimuli");

    for (unsigned int i
         = 0,
         nbTest = ((testIdx >= 0) ? 1 : database.getNbStimuli(Database::Test));
         i < nbTest;
         ++i) {
        const unsigned int idx = (testIdx >= 0) ? testIdx : i;
        const bool logStep = (i + 1) % log == 0 || i == nbTest - 1;

        // Run
        std::cout << "Test from " << i* TimeUs / (double)TimeS << "s to "
                  << (i + 1) * TimeUs / (double)TimeS << "s..." << std::endl;

        const Database::StimulusID id = env->readStimulus(Database::Test, idx);
        env->propagate(i * TimeUs, (i + 1) * TimeUs);
        net.run((i + 1) * TimeUs);

        // Check activity
        std::vector<std::pair<std::string, unsigned int> > activity
            = deepNet->update(logStep, i * TimeUs, (i + 1) * TimeUs, logStep);

        monitorEnv.update(i < 10);

        unsigned int sumActivity = 0;

        for (std::vector<std::pair<std::string, unsigned int> >::const_iterator
                 it = activity.begin(),
                 itEnd = activity.end();
             it != itEnd;
             ++it) {
            std::cout << "   " << (*it).first << ": " << (*it).second;
            sumActivity += (*it).second;
        }

        std::cout << "   (total: " << sumActivity << ")" << std::endl;

        // Check success
        std::shared_ptr<FcCell_Spike> fcCell = deepNet->getTargetCell
                                               <FcCell_Spike>();

        if (!fcCell)
            throw std::domain_error(
                "Only FcCell_Spike is currently supported for output cell.");

        const unsigned int outputTarget = deepNet->getTarget()->getLabelTarget(
            database.getStimulusLabel(id));
        const unsigned int targetId = fcCell->getOutput(outputTarget)->getId();

        if (testIdx >= 0) {
            const bool success = monitorOut.checkLearningResponse(
                database.getStimulusLabel(id),
                targetId,
                fcCell->getBestResponseId(true));

            std::cout << "   #" << idx << " / " << targetId
                      << " / success rate: "
                      << (monitorOut.getSuccessRate() * 100) << "%";

            if (!success)
                std::cout << "   [FAIL]";

            std::cout << std::endl;

            deepNet->logSpikeStats("stats-test-idx", 1);
            deepNet->spikeCodingCompare("compare-test-idx", idx);
        } else {
            // Work also for unsupervised learning
            const bool success = (learn > 0 || test)
                                     ? monitorOut.checkLearningResponse(
                                           database.getStimulusLabel(id),
                                           targetId,
                                           fcCell->getBestResponseId())
                                     : monitorOut.checkLearningResponse(
                                           database.getStimulusLabel(id),
                                           fcCell->getBestResponseId());

            std::cout << "   #" << idx << " / success rate: "
                      << (monitorOut.getSuccessRate() * 100) << "%";

            if (!success)
                std::cout << "   [FAIL]";

            std::cout << std::endl;

            if (logStep) {
                monitorOut.logSuccessRate("test_success_spike.dat", 0, true);
                deepNet->logSpikeStats("stats_spike", i + 1);
            }

            // Log env activity
            if (i < 10) {
                std::ostringstream baseName;
                baseName << "stimuli/stimuli_" << idx;
                /*
                                cv::Mat img;
                                cv::resize(env->reconstructFrame(idx,
                   Environment::Test), img, cv::Size(512, 512));
                                cv::imwrite(baseName.str() + ".jpg", img);
                */
                monitorEnv.logActivity(baseName.str() + "_activity.dat", true);
                monitorEnv.clearAll();
            }
        }

        net.reset((i + 1) * TimeUs);
    }

    if (testIdx < 0) {
        std::cout << "Final spiking recognition rate: "
                  << (100.0 * monitorOut.getSuccessRate())
                  << "%"
                     "    (error rate: "
                  << 100.0 * (1.0 - monitorOut.getSuccessRate()) << "%)"
                  << std::endl;
    }

    return 0;
}
catch (const std::exception& e)
{
    std::cout << "Error: " << e.what() << std::endl;
    return 0;
}
