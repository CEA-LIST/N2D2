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
#include "CEnvironment.hpp"
#include "Environment.hpp"
#include "NodeEnv.hpp"
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
#include "Transformation/RangeAffineTransformation.hpp"
#include "utils/ProgramOptions.hpp"

#ifdef CUDA
#include <cudnn.h>

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

//#define GPROF_INTERRUPT

#if defined(__GNUC__) && !defined(NDEBUG) && defined(GPROF_INTERRUPT)
#include <dlfcn.h>

void sigUsr1Handler(int /*sig*/)
{
    std::cerr << "Exiting on SIGUSR1" << std::endl;

    void (*_mcleanup)(void);

    _mcleanup = (void (*)(void))dlsym(RTLD_DEFAULT, "_mcleanup");

    if (_mcleanup == NULL)
        std::cerr << "Unable to find gprof exit hook" << std::endl;
    else
        _mcleanup();

    _exit(0);
}
#endif


void printVersionInformation() {
    // N2D2 version
    std::cout << "N2D2 (" __DATE__ " " __TIME__ ")\n"
        "(C) Copyright 2010-2019 CEA LIST. All Rights Reserved.\n\n";

    // Compiler version
#if defined(__clang__)
/* Clang/LLVM. ---------------------------------------------- */
    std::cout << "Clang/LLVM compiler version: " << __clang_major__ << "."
        << __clang_minor__ << "." << __clang_patchlevel__ << "\n\n";
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC. ------------------------------------------ */
    std::cout << "Intel ICC/ICPC compiler version: "
        << __INTEL_COMPILER << "\n\n";
#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++. --------------------------------------------- */
    std::cout << "GNU GCC/G++ compiler version: " << __GNUC__ << "."
        << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n\n";
#elif defined(__IBMC__) || defined(__IBMCPP__)
/* IBM XL C/C++. -------------------------------------------- */
    std::cout << "IBM XL C/C++ compiler version: "
        << std::hex <<  __IBMCPP__ << std::dec << "\n\n";
#elif defined(_MSC_VER)
/* Microsoft Visual Studio. --------------------------------- */
    std::cout << "Microsoft Visual Studio compiler version: "
        << _MSC_VER << "\n\n";
#else
    std::cout << "Unknown compiler\n\n";
#endif

    // OpenCV version
    std::cout << cv::getBuildInformation() << "\n\n";

#ifdef CUDA
    // CUDA version
    int deviceCount = 0;
    CHECK_CUDA_STATUS(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA"
            << std::endl;
    }
    else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)"
            << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice #" << dev << ": \""
            << deviceProp.name << "\"" << std::endl;

        int driverVersion = 0;
        int runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        std::cout << "  CUDA Driver Version / Runtime Version:         "
            << (driverVersion / 1000) << "." << ((driverVersion % 100) / 10)
            << " / " << (runtimeVersion / 1000) << "."
            << ((runtimeVersion % 100) / 10) << std::endl;
        std::cout << "  CUDA Capability Major/Minor version number:    "
            << deviceProp.major << "." << deviceProp.minor << std::endl;
    }

    std::cout << "\n";

    std::cout << "CuDNN version: " << CUDNN_MAJOR << "."
        << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << "\n\n";
#endif
}

class Options {
public:
    Options(int argc, char* argv[]) {
        ProgramOptions opts(argc, argv);

        seed =        opts.parse("-seed", 0U, "N2D2 random seed (0 = time based)");
        log =         opts.parse("-log", 1000U, "number of steps between logs");
        report =      opts.parse("-report", 100U, "number of steps between reportings");
        learn =       opts.parse("-learn", 0U, "number of backprop learning steps");
        preSamples =  opts.parse("-pre-samples", -1, "if >= 0, log pre-processing samples "
                                                     "of the corresponding stimulus ID");
        findLr =      opts.parse("-find-lr", 0U, "find an appropriate learning rate over a"
                                                 " number of iterations");
        validMetric = opts.parse("-valid-metric", ConfusionTableMetric::Sensitivity,
                                                  "validation metric to use (default is "
                                                  "Sensitivity)");
        objectDetector = opts.parse("-object-detector-metric", false,"use object detector metrics");
        stopValid =   opts.parse("-stop-valid", 0U, "max. number of successive lower score "
                                                   "validation");
        test =        opts.parse("-test", "perform testing");
        fuse =        opts.parse("-fuse", "fuse BatchNorm with Conv for test and export");
        bench =       opts.parse("-bench", "learning speed benchmarking");
        learnStdp =   opts.parse("-learn-stdp", 0U, "number of STDP learning steps");
        avgWindow =   opts.parse("-ws", 10000U, "average window to compute success rate "
                                                "during learning");
        testIndex =   opts.parse("-test-index", -1, "test a single specific stimulus index"
                                                    " in the Test set");
        testId =      opts.parse("-test-id", -1, "test a single specific stimulus ID (takes"
                                                 " precedence over -test-index)");
        check =       opts.parse("-check", "enable gradient computation checking");
        logOutputs =  opts.parse("-log-outputs", 0U, "log layers outputs for the n-th "
                                                     "stimulus (0 = no log)");
        logDbStats =  opts.parse("-log-db-stats", "log database stimuli and ROIs stats");
        genConfig =   opts.parse("-cfg", "save base configuration and exit");
        genExport =   opts.parse("-export", std::string(), "generate an export and exit");
        nbBits =      opts.parse("-nbbits", 8, "number of bits per weight for exports");
        calibration = opts.parse("-calib", 0, "number of stimuli used for the calibration "
                                              "(0 = no calibration, -1 = use the full "
                                              "test dataset)");
        nbCalibrationPasses = opts.parse("-calib-passes", 2, "number of KL passes for determining "
                                                             "the layer output values distribution "
                                                             "truncation threshold (0 = use the "
                                                             "max. value, no truncation)");
        timeStep =    opts.parse("-ts", 0.1, "timestep for clock-based simulations (ns)");
        saveTestSet = opts.parse("-save-test-set", std::string(), "save the test dataset to a "
                                                                  "specified location");
        load =        opts.parse("-l", std::string(), "start with a previously saved state from a "
                                                      "specified location");
        weights =     opts.parse("-w", std::string(), "start with weights imported from a specified "
                                                      "location (even when loading a previously "
                                                      "saved state)");
        exportNoUnsigned =   opts.parse("-no-unsigned", "disable the use of unsigned data type in "
                                                        "integer exports");
        exportNbStimuliMax = opts.parse("-db-export", -1, "max. number of stimuli to export "
                                                          "(0 = no dataset export, -1 = unlimited)");
        
        N2D2::DeepNetExport::setExportParameters(opts.parse("-export-parameters", std::string(), 
                                                                        "parameters for export"));

    #ifdef CUDA
        cudaDevice =  opts.parse("-dev", 0, "CUDA device ID");
    #endif

        version =     opts.parse("-v", "display version information");
        if (version) {
            printVersionInformation();
            std::exit(0);
        }

        iniConfig =   opts.grab<std::string>("<net>", "network config file (INI)");
        opts.done();  


        // Ensures that the seed is the same for the test than for the learning (to
        // avoid including learned stimuli in the test set)
        if (seed == 0 && learn == 0) {
            seed = Network::readSeed("seed.dat");
        }
    }

    unsigned int seed;
    unsigned int log;
    unsigned int report;
    unsigned int learn;
    int preSamples;
    unsigned int findLr;
    ConfusionTableMetric validMetric;
    bool objectDetector;
    unsigned int stopValid;
    bool test;
    bool fuse;
    bool bench;
    unsigned int learnStdp;
    unsigned int avgWindow;
    int testIndex;
    int testId;
    bool check;
    unsigned int logOutputs;
    bool logDbStats;
    bool genConfig;
    std::string genExport;
    int nbBits;
    int calibration;
    unsigned int nbCalibrationPasses;
    bool exportNoUnsigned;
    double timeStep;
    std::string saveTestSet;
    std::string load;
    std::string weights;
    int exportNbStimuliMax;
    bool version;
    std::string iniConfig;
};

void test(const Options& opt, std::shared_ptr<DeepNet>& deepNet, bool afterCalibration) {
    const std::string testName = (afterCalibration) ? "export" : "test";

    std::shared_ptr<Database> database = deepNet->getDatabase();
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();

    std::map<std::string, RangeStats> outputsRange;
    std::map<std::string, Histogram> outputsHistogram;
    
    std::vector<std::pair<std::string, double> > timings, cumTimings;

    // Static testing
    unsigned int nextLog = opt.log;
    unsigned int nextReport = opt.report;
    std::chrono::high_resolution_clock::time_point startTimeSp,
                                                    endTimeSp;

    const unsigned int nbTest = (opt.testIndex >= 0 || opt.testId >= 0)
        ? 1 : database->getNbStimuli(Database::Test);
    const unsigned int batchSize = sp->getBatchSize();
    const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

    for (unsigned int b = 0; b < nbBatch; ++b) {
        const unsigned int i = b * batchSize;
        const unsigned int idx = (opt.testIndex >= 0) ? opt.testIndex : i;

        startTimeSp = std::chrono::high_resolution_clock::now();
        if (opt.testId >= 0)
            sp->readStimulusBatch(Database::Test, opt.testId);
        else
            sp->readBatch(Database::Test, idx);
        endTimeSp = std::chrono::high_resolution_clock::now();

        deepNet->test(Database::Test, &timings);

        deepNet->reportOutputsRange(outputsRange);
        deepNet->reportOutputsHistogram(outputsHistogram);
        deepNet->logEstimatedLabels(testName);

        if (opt.logOutputs > 0 && b == (opt.logOutputs - 1) / batchSize) {
            const unsigned int batchPos = (opt.logOutputs - 1) % batchSize;

            std::cout << "Outputs log for stimulus #" << opt.logOutputs
                << " (" << (batchPos + 1) << "/" << batchSize
                << " in batch #" << b << "):" << std::endl;
            std::cout << "  Stimulus ID: " << sp->getBatch()[batchPos]
                        << std::endl;
            std::cout << "  Stimulus label: "
                        << sp->getLabelsData()[batchPos](0) << std::endl;

            std::stringstream numStr;
            numStr << opt.logOutputs;

            deepNet->logOutputs("outputs_" + testName + "_" + numStr.str(),
                                batchPos);
        }

        timings.push_back(std::make_pair(
            "sp", std::chrono::duration_cast
            <std::chrono::duration<double> >(endTimeSp - startTimeSp)
                                            .count()));

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
            nextReport += opt.report;
            std::cout << "Testing #" << idx << "   ";

            for (std::vector<std::shared_ptr<Target> >::const_iterator
                        itTargets = deepNet->getTargets().begin(),
                        itTargetsEnd = deepNet->getTargets().end();
                    itTargets != itTargetsEnd;
                    ++itTargets) {
                if(!opt.objectDetector)
                {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast
                        <TargetScore>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                    Database::Test)) << "% ";
                }
                else {
                    std::shared_ptr<TargetBBox> target
                        = std::dynamic_pointer_cast
                        <TargetBBox>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                    Database::Test)) << "% ";
                }
            }
            std::cout << std::endl;
        }

        if (i >= nextLog || b == nbBatch - 1) {
            nextLog += opt.report;

            for (std::vector<std::shared_ptr<Target> >::const_iterator
                        itTargets = deepNet->getTargets().begin(),
                        itTargetsEnd = deepNet->getTargets().end();
                    itTargets != itTargetsEnd;
                    ++itTargets) {
                if(!opt.objectDetector)
                {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast
                        <TargetScore>(*itTargets);

                    if (target) {
                        target->logSuccess(testName, Database::Test);
                        target->logTopNSuccess(testName, Database::Test);
                    }
                }
                else {

                    std::shared_ptr<TargetBBox> target
                        = std::dynamic_pointer_cast
                        <TargetBBox>(*itTargets);

                    if (target) 
                        target->logSuccess(testName, Database::Test);
                    
                }
            }
        }
    }

    if (nbTest > 0) {
        deepNet->log(testName, Database::Test);
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
            if(!opt.objectDetector)
            {

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

                    std::cout << "    Sensitivity: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Sensitivity))
                                << "% / Specificity: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Specificity))
                                << "% / Precision: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Precision))
                                << "%\n"
                                "    Accuracy: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Accuracy))
                                << "% / F1-score: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::F1Score))
                                << "% / Informedness: " << (100.0
                        * target->getAverageScore(Database::Test,
                                    ConfusionTableMetric::Informedness))
                                << "%\n" << std::endl;
                }
            }
            else {
                std::shared_ptr<TargetBBox> target
                    = std::dynamic_pointer_cast<TargetBBox>(*itTargets);

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
    }

    Histogram::logOutputsHistogram(testName + "_outputs_histogram",
                                    outputsHistogram);
    RangeStats::logOutputsRange(testName + "_outputs_range.dat",
                                outputsRange);

    if (!afterCalibration) {
        deepNet->normalizeFreeParameters();
        deepNet->exportNetworkFreeParameters("weights_normalized");
    }    
    
}

bool generateExport(const Options& opt, std::shared_ptr<DeepNet>& deepNet) {
    std::shared_ptr<Database> database = deepNet->getDatabase();
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();


    bool afterCalibration = false;
    try {
        if (!opt.weights.empty())
            deepNet->importNetworkFreeParameters(opt.weights);
        else if (opt.load.empty()) {
            if (database->getNbStimuli(Database::Validation) > 0)
                deepNet->importNetworkFreeParameters("weights_validation");
            else
                deepNet->importNetworkFreeParameters("weights");
        }
    }
    catch (const std::exception& e) {
        std::cout << Utils::cwarning << e.what()
            << Utils::cdef << std::endl;
    }

    deepNet->removeDropout();

    if (opt.fuse)
        deepNet->fuseBatchNormWithConv();

    std::stringstream exportDir;
    exportDir << "export_" << opt.genExport << "_"
                << ((opt.nbBits > 0) ? "int" : "float") << std::abs(opt.nbBits);

    DeepNetExport::mUnsignedData = (!opt.exportNoUnsigned);
    CellExport::mPrecision = static_cast<CellExport::Precision>(opt.nbBits);

    if (opt.calibration != 0 && opt.nbBits > 0) {
        if(opt.weights.empty() && opt.load.empty()) {
            std::cout << "No weights or saved state passed in parameters. "
                      << "The default weights are not normalized. "
                      << "Using the test database to normalize them before the calibration pass..." 
                      << std::endl;

            test(opt, deepNet, false);
        }

        const double stimuliRange = StimuliProviderExport::getStimuliRange(
                                            *sp,
                                            exportDir.str()
                                            + "/stimuli_before_calibration",
                                            Database::Test);

        if (stimuliRange == 0.0) {
            throw std::runtime_error("Stimuli range is 0."
                " Is there any stimulus? (required for calibration!)");
        }

        if (stimuliRange != 1.0) {
            // For calibration, normalizes the input in the full range of
            // export data precision
            // => stimuli range must therefore be in [-1,1] before export

            // If test is following (after calibration):
            // This step is necessary for mSignalsDiscretization parameter
            // to work as expected, which is set in normalizeOutputsRange()
            std::cout << "Stimuli range before transformation is "
                << stimuliRange << std::endl;

            sp->addOnTheFlyTransformation(RangeAffineTransformation(
                RangeAffineTransformation::Divides, stimuliRange),
                Database::TestOnly);
        }

        // Globally disable logistic activation, in order to evaluate the
        // correct range and shifting required for layers with logistic
        LogisticActivationDisabled = true;

        Utils::createDirectories(exportDir.str() + "/calibration");

        const std::string outputsRangeFile
            = exportDir.str() + "/calibration/outputs_range.bin";
        const std::string outputsHistogramFile
            = exportDir.str() + "/calibration/outputs_histogram.bin";

        std::map<std::string, RangeStats> outputsRange;
        std::map<std::string, Histogram> outputsHistogram;

        bool loadPrevState = false;

        if (std::ifstream(outputsRangeFile.c_str()).good()
            && std::ifstream(outputsHistogramFile.c_str()).good())
        {
            std::cout << "Load previously saved RangeStats and Histogram"
                " for the calibration? (y/n) ";

            std::string cmd;

            do {
                std::cin >> cmd;
            }
            while (cmd != "y" && cmd != "n");

            loadPrevState = (cmd == "y");
        }

        if (loadPrevState) {
            RangeStats::loadOutputsRange(outputsRangeFile, outputsRange);
            Histogram::loadOutputsHistogram(outputsHistogramFile,
                                            outputsHistogram);
        }
        else {
            unsigned int nextLog = opt.log;
            unsigned int nextReport = opt.report;

            const unsigned int nbTest = (opt.calibration > 0)
                ? std::min((unsigned int)opt.calibration,
                            database->getNbStimuli(Database::Test))
                : database->getNbStimuli(Database::Test);
            const unsigned int batchSize = sp->getBatchSize();
            const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

            for (unsigned int b = 0; b < nbBatch; ++b) {
                const unsigned int i = b * batchSize;

                sp->readBatch(Database::Test, i);
                deepNet->test(Database::Test);
                deepNet->reportOutputsRange(outputsRange);
                deepNet->reportOutputsHistogram(outputsHistogram);

                if (i >= nextReport || b == nbBatch - 1) {
                    nextReport += opt.report;
                    std::cout << "Calibration data #" << i << std::endl;
                }

                if (i >= nextLog || b == nbBatch - 1)
                    nextLog += opt.report;
            }

            RangeStats::saveOutputsRange(outputsRangeFile, outputsRange);
            Histogram::saveOutputsHistogram(outputsHistogramFile,
                                            outputsHistogram);
        }

        RangeStats::logOutputsRange(exportDir.str() + "/calibration"
                                    "/outputs_range.dat", outputsRange);
        Histogram::logOutputsHistogram(exportDir.str() + "/calibration"
                                    "/outputs_histogram", outputsHistogram);

        std::cout << "Calibration (" << opt.nbBits << " bits):" << std::endl;

        const unsigned int nbLevels = std::pow(2, opt.nbBits - 1);

        deepNet->normalizeOutputsRange(outputsHistogram, outputsRange,
                                        nbLevels, opt.nbCalibrationPasses);

        // For following test simulation
        deepNet->setParameter("SignalsDiscretization", nbLevels);
        deepNet->setParameter("FreeParametersDiscretization", nbLevels);

        // Clear the targets for the test that will occur afterward...
        deepNet->clear(Database::Test);

        LogisticActivationDisabled = false;
        afterCalibration = true;
    }

    StimuliProviderExport::generate(*sp,
                                    exportDir.str() + "/stimuli",
                                    opt.genExport,
                                    Database::Test,
                                    opt.exportNbStimuliMax,
                                    false,
                                    deepNet.get());

    // Must come after because StimuliProviderExport::generate() sets
    // mEnvDataUnsigned
    DeepNetExport::generate(*deepNet, exportDir.str(), opt.genExport);

    return afterCalibration;
}

void findLearningRate(const Options& opt, std::shared_ptr<DeepNet>& deepNet) {
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();

    std::cout << "Learning rate exploration over " << opt.findLr
        << " iterations..." << std::endl;

    const double startLr = 1.0e-6;
    const double endLr = 10.0;
    const double grow = (1.0 / opt.findLr) * std::log(endLr / startLr);

    const unsigned int batchSize = sp->getBatchSize();
    const unsigned int nbBatch = std::ceil(opt.findLr / (double)batchSize);
    std::vector<std::pair<std::string, double> >* timings = NULL;

    sp->readRandomBatch(Database::Learn);

    std::vector<double> learningRate;

    for (unsigned int b = 0; b < nbBatch; ++b) {
        const unsigned int i = b * batchSize;

        const double lr = startLr * std::exp(grow * i);
        Solver::mGlobalLearningRate = lr;
        learningRate.push_back(lr);

        sp->synchronize();
        std::thread learnThread(learnThreadWrapper, deepNet, timings);

        sp->future();
        sp->readRandomBatch(Database::Learn);

        std::ios::fmtflags f(std::cout.flags());

        std::cout << "\rRunning #" << std::setw(8) << std::left << i
                    << std::flush;

        std::cout.flags(f);

        learnThread.join();
    }

    Solver::mGlobalLearningRate = 0.0;

    std::string fileName;

    for (std::vector<std::shared_ptr<Target> >::const_iterator
                itTargets = deepNet->getTargets().begin(),
                itTargetsEnd = deepNet->getTargets().end();
            itTargets != itTargetsEnd;
            ++itTargets)
    {
        std::shared_ptr<TargetScore> target
            = std::dynamic_pointer_cast<TargetScore>(*itTargets);

        if (target) {
            fileName = "find_lr_" + target->getName() + ".dat";
            const std::vector<Float_T>& loss = target->getLoss();

            std::ofstream lrLoss(fileName.c_str());

            for (unsigned int i = 0; i < learningRate.size(); ++i) {
                lrLoss << learningRate[i] << " " << loss[i] << "\n";
            }

            lrLoss.close();

            Gnuplot gnuplot(fileName + ".gnu");
            gnuplot.set("grid").set("key off");
            gnuplot.setTitle("Find where the loss is still decreasing "
                                "but has not plateaued");
            gnuplot.setXlabel("Learning rate (log scale)");
            gnuplot.setYlabel("Loss");
            gnuplot.set("logscale x");
            gnuplot.set("samples 100");
            gnuplot.set("table \"" + fileName + ".smooth\"");
            gnuplot.saveToFile(fileName);
            gnuplot.plot(fileName, "using 1:2 smooth bezier with lines "
                            "lt 2 lw 2");
            gnuplot.unset("table");
            gnuplot.unset("logscale x");
            gnuplot.set("y2tics");

            // Get initial loss value
            gnuplot << "col=2";
            gnuplot << "row=0";
            gnuplot << "stats \"" + fileName + ".smooth\" every ::row::row "
                        "using col nooutput";
            gnuplot << "loss_init=STATS_min";

            // Find good y range
            gnuplot << "stats \"" + fileName + ".smooth\" using 1:2 "
                        "nooutput";
            gnuplot << "thres=STATS_max_y";
            gnuplot << "y_min=STATS_min_y-(loss_init-STATS_min_y)/10";
            gnuplot << "y_max=loss_init+(loss_init-STATS_min_y)/10";

            // Find max. learning rate
            gnuplot << "d(y) = ($0 == 0) ? (y1 = y, 1/0)"
                        " : (y2 = y1, y1 = y, y1-y2)";
            gnuplot << "dx=0.0";
            gnuplot << "valid=1";
            gnuplot << "loss_min=loss_init";
            gnuplot << "x_limit=0";
            gnuplot << "stats \"" + fileName + ".smooth\""
                " using ($1-dx):(valid = (valid && $2 <= thres), "
                "x_limit = (valid && $2 <= loss_min) ? $1 : x_limit, "
                "loss_min = (valid && $2 <= loss_min) ? $2 : loss_min) "
                "nooutput";

            // Find good learning rate
            gnuplot << "chunck0=0";
            gnuplot << "chunck1=0";
            gnuplot << "x_min=0";
            gnuplot << "x_max=0";
            gnuplot << "stats \"" + fileName + ".smooth\""
                " using ($1-dx):(v = d($2), "
                "chunck0 = ($2 < loss_init && v < 0 && $1 <= x_limit) ? 1 : 0, "
                "x_min = (chunck0 && !chunck1) ? $1 : x_min, "
                "x_max = (chunck1 && !chunck0) ? $1 : x_max, "
                "chunck1 = chunck0, "
                "(($2 < loss_init && v < 0 && $1 <= x_limit) "
                    "? $2 : loss_init)) nooutput";
            gnuplot << "lr=(x_max+x_min)/2";

            gnuplot.set("logscale x");
            gnuplot.set("yrange [y_min:y_max]");

            // Display result (LR bar + value)
            gnuplot.set("arrow from lr,graph 0 to lr,graph 1"
                        " nohead lc rgb \"red\"");
            gnuplot.set("label 1 sprintf(\"%3.4f\",lr) at lr,graph 1"
                        " offset 0.5,-1 tc rgb \"red\"");

            // Plot smooth curve
            gnuplot << "replot \"" + fileName + "\" using 1:2"
                " with lines lt 3"
                "#, \"" + fileName + ".smooth\" using "
                "($1-dx):(v = d($2), "
                "(($2 < loss_init && v < 0 && $1 <= x_limit) "
                    "? $2 : loss_init)) "
                "with lines lc 4";
        }
    }

    if (!fileName.empty()) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");

        std::stringstream xLabelStr;
        xLabelStr << "Iteration # (batch size: " << batchSize << ")";

        gnuplot.setXlabel(xLabelStr.str());
        gnuplot.setYlabel("Learning rate (log scale)");
        gnuplot.set("logscale y");
        gnuplot.saveToFile("find_lr-range.dat");
        gnuplot.plot(fileName, "using 0:1 with lines");
    }

    // We are still in future batch, need to synchronize for the following
    sp->synchronize();

    std::cout << "Done!" << std::endl;
}

void learn(const Options& opt, std::shared_ptr<DeepNet>& deepNet) {
    std::shared_ptr<Database> database = deepNet->getDatabase();
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();

    deepNet->exportNetworkFreeParameters("weights_init");

    std::chrono::high_resolution_clock::time_point startTime
        = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point startTimeSp, endTimeSp;
    double minTimeElapsed = 0.0;
    unsigned int nextLog = opt.log;
    unsigned int nextReport = opt.report;
    unsigned int nbNoValid = 0;

    const unsigned int batchSize = sp->getBatchSize();
    const unsigned int nbBatch = std::ceil(opt.learn / (double)batchSize);
    const unsigned int avgBatchWindow = opt.avgWindow / (double)batchSize;

    sp->readRandomBatch(Database::Learn);

    std::vector<std::pair<std::string, double> > timings, cumTimings;

    for (unsigned int b = 0; b < nbBatch; ++b) {
        const unsigned int i = b * batchSize;

        sp->synchronize();
        std::thread learnThread(learnThreadWrapper,
                                deepNet,
                                (opt.bench) ? &timings : NULL);

        sp->future();
        startTimeSp = std::chrono::high_resolution_clock::now();
        sp->readRandomBatch(Database::Learn);
        endTimeSp = std::chrono::high_resolution_clock::now();

        learnThread.join();

        if (opt.logOutputs > 0 && b == (opt.logOutputs - 1) / batchSize) {
            const unsigned int batchPos = (opt.logOutputs - 1) % batchSize;

            std::cout << "Outputs log for stimulus #" << opt.logOutputs
                << " (" << (batchPos + 1) << "/" << batchSize
                << " in batch #" << b << "):" << std::endl;
            std::cout << "  Stimulus ID: " << sp->getBatch()[batchPos]
                        << std::endl;
            std::cout << "  Stimulus label: "
                        << sp->getLabelsData()[batchPos](0) << std::endl;

            std::stringstream numStr;
            numStr << opt.logOutputs;

            deepNet->logOutputs("outputs_" + numStr.str(), batchPos);
            deepNet->logDiffInputs("diffinputs_" + numStr.str(), batchPos);
        }

        if (opt.bench) {
            timings.push_back(std::make_pair(
                "sp", std::chrono::duration_cast
                <std::chrono::duration<double> >(endTimeSp - startTimeSp)
                                                .count()));

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
            nextReport += opt.report;

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
                if(!opt.objectDetector)
                {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                    Database::Learn,
                                                    avgBatchWindow)) << "% ";
                }
                else {
                    std::shared_ptr<TargetBBox> target
                        = std::dynamic_pointer_cast<TargetBBox>(*itTargets);

                    if (target)
                        std::cout << (100.0 * target->getAverageSuccess(
                                                    Database::Learn)) << "% ";
                }
            }

            const double timeElapsed = std::chrono::duration_cast
                                        <std::chrono::duration<double> >(
                                            curTime - startTime).count();

            if (minTimeElapsed == 0.0 || minTimeElapsed < timeElapsed)
                minTimeElapsed = timeElapsed;

            std::cout << "at " << std::setw(7) << std::fixed
                        << std::setprecision(2) << (opt.report / timeElapsed)
                        << " p./s"
                            " (" << std::setw(7) << std::fixed
                        << std::setprecision(0)
                        << 60.0 * (opt.report / timeElapsed)
                        << " p./min)         " << std::setprecision(4)
                        << std::flush;

            std::cout.flags(f);

            startTime = std::chrono::high_resolution_clock::now();
        }

        if (i >= nextLog || b == nbBatch - 1) {
            nextLog += opt.log;

            std::cout << std::endl;

            for (std::vector<std::shared_ptr<Target> >::const_iterator
                        itTargets = deepNet->getTargets().begin(),
                        itTargetsEnd = deepNet->getTargets().end();
                    itTargets != itTargetsEnd;
                    ++itTargets) {
                if(!opt.objectDetector)
                {
                    std::shared_ptr<TargetScore> target
                        = std::dynamic_pointer_cast<TargetScore>(*itTargets);

                    if (target) {
                        target->logSuccess(
                            "learning", Database::Learn, avgBatchWindow);
                        // target->logTopNSuccess("learning", Database::Learn,
                        // avgBatchWindow);
                    }
                }
                else {
                    std::shared_ptr<TargetBBox> target
                        = std::dynamic_pointer_cast<TargetBBox>(*itTargets);

                    if (target) {
                        target->logSuccess(
                            "learning", Database::Learn, avgBatchWindow);
                        // target->logTopNSuccess("learning", Database::Learn,
                        // avgBatchWindow);
                    }
                }
            }

            if (opt.bench) {
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

            if (database->getNbStimuli(Database::Validation) > 0) {
                const unsigned int nbValid
                    = database->getNbStimuli(Database::Validation);
                const unsigned int nbBatchValid
                    = std::ceil(nbValid / (double)batchSize);

                std::cout << "Validation" << std::flush;
                unsigned int progress = 0, progressPrev = 0;

                sp->readBatch(Database::Validation, 0);

                for (unsigned int bv = 1; bv <= nbBatchValid; ++bv) {
                    const unsigned int k = bv * batchSize;

                    sp->synchronize();
                    std::thread validationThread(validationThreadWrapper,
                                                    deepNet);

                    sp->future();

                    if (bv < nbBatchValid)
                        sp->readBatch(Database::Validation, k);

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

                sp->readRandomBatch(Database::Learn);

                for (std::vector<std::shared_ptr<Target> >::const_iterator
                            itTargets = deepNet->getTargets().begin(),
                            itTargetsEnd = deepNet->getTargets().end();
                        itTargets != itTargetsEnd;
                        ++itTargets) {
                    if(!opt.objectDetector)
                    {
                        std::shared_ptr<TargetScore> target
                            = std::dynamic_pointer_cast
                            <TargetScore>(*itTargets);

                        if (!target)
                            continue;

                        const bool bestValidation = target->newValidationScore(
                                target->getAverageScore(Database::Validation,
                                                        opt.validMetric));

                        if (bestValidation) {
                            std::cout << "\n+++ BEST validation score: "
                                        << (100.0
                                            * target->getMaxValidationScore())
                                        << "% [" << opt.validMetric << "]\n";

                            deepNet->log("validation", Database::Validation);
                            deepNet->exportNetworkFreeParameters(
                                "weights_validation");
                            deepNet->save("net_state_validation");
                        }
                        else {
                            std::cout << "\n--- LOWER validation score: "
                                        << (100.0
                                            * target->getLastValidationScore())
                                        << "% [" << opt.validMetric << "] (best was "
                                        << (100.0
                                            * target->getMaxValidationScore())
                                        << "%)\n" << std::endl;

                        }

                        std::cout << "    Sensitivity: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::Sensitivity))
                                    << "% / Specificity: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::Specificity))
                                    << "% / Precision: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::Precision))
                                    << "%\n"
                                    "    Accuracy: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::Accuracy))
                                    << "% / F1-score: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::F1Score))
                                    << "% / Informedness: " << (100.0
                            * target->getAverageScore(Database::Validation,
                                        ConfusionTableMetric::Informedness))
                                    << "%\n" << std::endl;

                        if (!bestValidation) {
                            ++nbNoValid;

                            if (opt.stopValid > 0 && nbNoValid >= opt.stopValid) {
                                std::cout
                                    << "\n--- Validation did not improve after "
                                    << opt.stopValid << " steps\n" << std::endl;
                                std::cout << "\n--- STOPPING THE LEARNING\n"
                                            << std::endl;
                                break;
                            }
                        }
                        else
                            nbNoValid = 0;

                        target->newValidationTopNScore(
                            target->getAverageTopNScore(
                                Database::Validation, opt.validMetric));
                        target->logSuccess(
                            "validation", Database::Validation, avgBatchWindow);
                        target->logTopNSuccess(
                            "validation",
                            Database::Validation,
                            avgBatchWindow); // Top-N accuracy
                        target->clearSuccess(Database::Validation);
                    }
                    else {
                        std::shared_ptr<TargetBBox> target
                            = std::dynamic_pointer_cast
                            <TargetBBox>(*itTargets);

                        if (!target)
                            continue;

                        const bool bestValidation = target->newValidationScore(
                                target->getAverageSuccess(Database::Validation));

                        if (bestValidation) {
                            std::cout << "\n+++ BEST validation score: "
                                        << (100.0
                                            * target->getMaxValidationScore())
                                        << "% [" << opt.validMetric << "]\n";

                            deepNet->log("validation", Database::Validation);
                            deepNet->exportNetworkFreeParameters(
                                "weights_validation");
                            deepNet->save("net_state_validation");
                        }
                        else {
                            std::cout << "\n--- LOWER validation score: "
                                        << (100.0
                                            * target->getLastValidationScore())
                                        << "% [" << opt.validMetric << "] (best was "
                                        << (100.0
                                            * target->getMaxValidationScore())
                                        << "%)\n" << std::endl;

                        }

                        if (!bestValidation) {
                            ++nbNoValid;

                            if (opt.stopValid > 0 && nbNoValid >= opt.stopValid) {
                                std::cout
                                    << "\n--- Validation did not improve after "
                                    << opt.stopValid << " steps\n" << std::endl;
                                std::cout << "\n--- STOPPING THE LEARNING\n"
                                            << std::endl;
                                break;
                            }
                        }
                        else
                            nbNoValid = 0;
                        target->logSuccess("validation", Database::Validation, avgBatchWindow);
                        target->clearSuccess(Database::Validation);

                    }
                }

                deepNet->clear(Database::Validation);
            }
            else {
                deepNet->exportNetworkFreeParameters("weights");
                deepNet->save("net_state");
            }
        }
    }

    deepNet->logFreeParameters("kernels");

    // We are still in future batch, need to synchronize for the following
    sp->synchronize();
}

void learnStdp(const Options& opt, std::shared_ptr<DeepNet>& deepNet, 
               std::shared_ptr<Environment>& env, Network& net, 
               Monitor& monitorEnv, Monitor& monitorOut) 
{
    std::shared_ptr<Database> database = deepNet->getDatabase();
    
    Utils::createDirectories("weights_stdp");

    for (unsigned int i = 0; i < opt.learnStdp; ++i) {
        const bool logStep = (i + 1) % opt.log == 0 || i == opt.learnStdp - 1;

        // Run
        std::cout << "Learn from " << i* TimeUs / (double)TimeS << "s to "
                    << (i + 1) * TimeUs / (double)TimeS << "s..."
                    << std::endl;

        const Database::StimulusID id
            = env->readRandomStimulus(Database::Learn);
        env->propagate(i * TimeUs, (i + 1) * TimeUs);
        net.run((i + 1) * TimeUs);

        // Check activity
        std::vector<std::pair<std::string, long long int> > activity
            = deepNet->update(
                logStep, i * TimeUs, (i + 1) * TimeUs, logStep);

        monitorEnv.update();

        long long int sumActivity = 0;

        for (std::vector
                <std::pair<std::string, long long int> >::const_iterator it
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
                database->getStimulusLabel(id));

        const bool success = monitorOut.checkLearningResponse(
            outputTarget, fcCell->getBestResponseId());
        std::cout << "   #" << id << " / success rate: "
                    << (monitorOut.getSuccessRate(opt.avgWindow) * 100) << "%";

        if (!success)
            std::cout << "   [FAIL]";

        std::cout << std::endl;

        if (logStep) {
            deepNet->exportNetworkFreeParameters("weights_stdp");
            deepNet->logFreeParameters("kernels");

            monitorOut.logSuccessRate(
                "learning_success_spike.dat", opt.avgWindow, true);
            deepNet->logSpikeStats("stats_learning_spike", i + 1);
        }

        net.reset((i + 1) * TimeUs);
    }

    deepNet->exportNetworkFreeParameters("./");

    net.reset();

    deepNet->clearAll();
    deepNet->setCellsParameter("EnableStdp", false);    
}

void testStdp(const Options& opt, std::shared_ptr<DeepNet>& deepNet,
              std::shared_ptr<Environment>& env, Network& net, 
              Monitor& monitorEnv, Monitor& monitorOut) 
{
    std::shared_ptr<Database> database = deepNet->getDatabase();
    
    Utils::createDirectories("stimuli");

    for (unsigned int i = 0, nbTest = ((opt.testIndex >= 0 || opt.testId >= 0)
        ? 1 : database->getNbStimuli(Database::Test));
         i < nbTest;
         ++i) {
        const unsigned int idx = (opt.testIndex >= 0) ? opt.testIndex : i;
        const bool logStep = (i + 1) % opt.log == 0 || i == nbTest - 1;

        // Run
        std::cout << "Test from " << i* TimeUs / (double)TimeS << "s to "
                  << (i + 1) * TimeUs / (double)TimeS << "s..." << std::endl;

        Database::StimulusID id;

        if (opt.testId >= 0) {
            id = opt.testId;
            env->readStimulus(id, Database::Test);
        }
        else
            id = env->readStimulus(Database::Test, idx);

        env->propagate(i * TimeUs, (i + 1) * TimeUs);
        net.run((i + 1) * TimeUs);

        // Check activity
        std::vector<std::pair<std::string, long long int> > activity
            = deepNet->update(logStep, i * TimeUs, (i + 1) * TimeUs, logStep);

        monitorEnv.update(i < 10);

        long long int sumActivity = 0;

        for (std::vector<std::pair<std::string, long long int> >::const_iterator
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
            database->getStimulusLabel(id));
        const unsigned int targetId = fcCell->getOutput(outputTarget)->getId();

        if (opt.testIndex >= 0 || opt.testId >= 0) {
            const bool success = monitorOut.checkLearningResponse(
                database->getStimulusLabel(id),
                targetId,
                fcCell->getBestResponseId(true));

            std::cout << "   #" << idx << " / " << targetId
                      << " / success rate: "
                      << (monitorOut.getSuccessRate() * 100) << "%";

            if (!success)
                std::cout << "   [FAIL]";

            std::cout << std::endl;

            deepNet->logSpikeStats("stats-test-idx", 1);

            if (opt.testId < 0)
                deepNet->spikeCodingCompare("compare-test-idx", idx);
        } else {
            // Work also for unsupervised learning
            const bool success = (opt.learn > 0 || opt.test)
                                     ? monitorOut.checkLearningResponse(
                                           database->getStimulusLabel(id),
                                           targetId,
                                           fcCell->getBestResponseId())
                                     : monitorOut.checkLearningResponse(
                                           database->getStimulusLabel(id),
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

    if (opt.testIndex < 0 && opt.testId < 0) {
        std::cout << "Final spiking recognition rate: "
                  << (100.0 * monitorOut.getSuccessRate())
                  << "%"
                     "    (error rate: "
                  << 100.0 * (1.0 - monitorOut.getSuccessRate()) << "%)"
                  << std::endl;
    }
}

void testCStdp(const Options& opt, std::shared_ptr<DeepNet>& deepNet) {
    std::shared_ptr<Database> database = deepNet->getDatabase();
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();


    unsigned int nextLog = opt.log;
    unsigned int nextReport = opt.report;

    const unsigned int nbTest = (opt.testIndex >= 0 || opt.testId >= 0)
        ? 1 : database->getNbStimuli(Database::Test);
    const unsigned int batchSize = sp->getBatchSize();
    const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

    for (unsigned int b = 0; b < nbBatch; ++b) {
        const unsigned int i = b * batchSize;
        const unsigned int idx = (opt.testIndex >= 0) ? opt.testIndex : i;

        if (opt.testId >= 0)
            sp->readStimulusBatch(Database::Test, opt.testId);
        else
            sp->readBatch(Database::Test, idx);

        deepNet->cTicks(0, 1 * TimeUs, (Time_T)(opt.timeStep * TimeNs));
        deepNet->cTargetsProcess(Database::Test);

        if (i >= nextReport || b == nbBatch - 1) {
            nextReport += opt.report;
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
            nextLog += opt.report;

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

void logStats(const Options& opt, std::shared_ptr<DeepNet>& deepNet) {
    std::shared_ptr<Database> database = deepNet->getDatabase();
    std::shared_ptr<StimuliProvider> sp = deepNet->getStimuliProvider();

    sp->logTransformations("transformations.dot");

    DrawNet::drawGraph(*deepNet, Utils::baseName(opt.iniConfig));
    DrawNet::draw(*deepNet, Utils::baseName(opt.iniConfig) + ".svg");
    deepNet->logStats("stats");
    deepNet->logReceptiveFields("receptive_fields.log");
    deepNet->logLabelsMapping("labels_mapping.log");
    deepNet->logLabelsLegend("labels_legend.png");

    if (!opt.weights.empty())
        deepNet->importNetworkFreeParameters(opt.weights, true);

    if (opt.check) {
        std::cout << "Checking gradient computation..." << std::endl;
        deepNet->checkGradient(1.0e-3, 1.0e-3);
    }

    // Reconstruct some frames to see the pre-processing
    Utils::createDirectories("frames");

    for (unsigned int i = 0, size = std::min(10U,
                                    database->getNbStimuli(Database::Learn));
            i < size;
            ++i) {
        std::ostringstream fileName;
        fileName << "frames/frame_" << i << ".dat";

        sp->readStimulus(Database::Learn, i);
        StimuliProvider::logData(fileName.str(), sp->getData()[0]);

        const Tensor<int> labelsData = sp->getLabelsData()[0];

        if (labelsData.dimX() > 1 || labelsData.dimY() > 1) {
            fileName.str(std::string());
            fileName << "frames/frame_" << i << "_label.dat";

            Tensor<Float_T> displayLabelsData(labelsData.dims());

            for (unsigned int index = 0; index < labelsData.size();
                ++index)
            {
                displayLabelsData(index) = labelsData(index);
            }

            StimuliProvider::logData(fileName.str(), displayLabelsData);
        }
    }

    for (unsigned int i = 0, size = std::min(10U,
                                    database->getNbStimuli(Database::Test));
            i < size;
            ++i) {
        std::ostringstream fileName;
        fileName << "frames/test_frame_" << i << ".dat";

        sp->readStimulus(Database::Test, i);
        StimuliProvider::logData(fileName.str(), sp->getData()[0]);
    }

    if (opt.preSamples >= 0) {
        if (opt.preSamples >= (int)database->getNbStimuli(Database::Learn)) {
            throw std::runtime_error("Pre-sample stimulus ID is higher "
                        "than the number of stimuli in the Learn dataset");
        }

        std::ostringstream dirName;
        dirName << "pre_samples_" << opt.preSamples;

        // Larger sample of frames to see the pre-processing
        Utils::createDirectories(dirName.str());

        sp->readStimulus(Database::Learn, opt.preSamples);

        // Estimate if input of network is signed or unsigned
        const Tensor<Float_T> spData = sp->getData()[0];
        const std::pair<std::vector<Float_T>::const_iterator,
                        std::vector<Float_T>::const_iterator> minMaxIt
                = std::minmax_element(spData.begin(), spData.end());
        const bool isSigned = (*minMaxIt.first) < 0.0;

        for (unsigned int i = 0; i < 100; ++i) {
            std::ostringstream fileName;
            fileName << dirName.str() << "/sample_" << i << ".jpg";

            sp->readStimulus(Database::Learn, opt.preSamples);

            const cv::Mat mat = cv::Mat(sp->getData()[0]);
            cv::Mat frame;

            if (isSigned)
                mat.convertTo(frame, CV_8U, 127.0, 127.0);
            else
                mat.convertTo(frame, CV_8U, 255.0, 0.0);

            if (!cv::imwrite(fileName.str(), frame)) {
                throw std::runtime_error("Unable to write image: "
                                            + fileName.str());
            }
        }
    }
}

int main(int argc, char* argv[]) try
{
#if defined(__GNUC__) && !defined(NDEBUG) && defined(GPROF_INTERRUPT)
    signal(SIGINT, sigUsr1Handler);
#endif

    const Options opt(argc, argv);

#ifdef CUDA
    CudaContext::setDevice(cudaDevice);
#endif

    // Network topology construction
    SGDSolver::mMaxSteps = opt.learn;
    SGDSolver::mLogSteps = opt.log;

    Network net(opt.seed);
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, opt.iniConfig);
    deepNet->initialize();

    if (opt.genConfig) {
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
    }

    if (!opt.saveTestSet.empty()) {
        CompositeTransformation trans;
/*
        StimuliProvider& sp = *deepNet->getStimuliProvider();
        trans.push_back(SliceExtractionTransformation(sp.getSizeX(),
                                                      sp.getSizeY()));
        trans[0]->setParameter<bool>("AllowPadding", true);
*/
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

    if (!afterCalibration) {
        logStats(opt, deepNet);
    }

    if (opt.findLr > 0) {
        findLearningRate(opt, deepNet);
        std::exit(0);
    }

    if (opt.learn > 0) {
        learn(opt, deepNet);
    }

    if (!afterCalibration) {
        if (opt.learn > 0) {
            // Reload best state after learning
            if (database.getNbStimuli(Database::Validation) > 0)
                deepNet->load("net_state_validation");
            else
                deepNet->load("net_state");
        }
        else if (opt.load.empty() && opt.weights.empty()) {
            if (database.getNbStimuli(Database::Validation) > 0)
                deepNet->importNetworkFreeParameters("weights_validation");
            else
                deepNet->importNetworkFreeParameters("weights");
        }
 
        if (opt.fuse)
            deepNet->fuseBatchNormWithConv();
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
        if (cellFrame && (opt.learn > 0 || opt.test)) {
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

    testStdp(opt, deepNet, env, net, monitorEnv, monitorOut);

    return 0;
}
catch (const std::exception& e)
{
    std::cout << "Error: " << e.what() << std::endl;
    return 0;
}
