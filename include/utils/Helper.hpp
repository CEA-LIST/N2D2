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
#include "Pruning.hpp"
#ifdef CUDA
#include <cudnn.h>

#include "CudaContext.hpp"
#endif

using namespace N2D2;

namespace N2D2_HELPER{
    class Options {
    public:
        Options();

        Options(int argc, char* argv[]);
        unsigned int seed = 0U;
        unsigned int log = 1000U;
        unsigned int logEpoch = 1U;
        unsigned int report = 100U;
        unsigned int learn = 0U;
        unsigned int learnEpoch = 0U;
        int preSamples = -1;
        unsigned int findLr = 0U;
        ConfusionTableMetric validMetric = ConfusionTableMetric::Sensitivity;
        unsigned int stopValid = 0U;
        bool test = false;
        bool testQAT = false;
        bool fuse = false;
        bool bench = false;
        unsigned int learnStdp = 0U;
        double presentTime = 1.0;
        unsigned int avgWindow = 10000U;
        int testIndex = -1;
        int testId = -1;
        std::string testAdv = std::string();
        std::string pruningMethod = std::string();
        unsigned int fineTune = 0U;
        bool check = false;
        unsigned int logOutputs = 0U;
        bool logJSON = false;
        bool logDbStats = false;
        bool logKernels = false;
        bool genConfig = false;
        std::string genExport = std::string();
        int nbBits = 8;
        int calibration = 0;
        bool calibrationReload = false;
        bool calibOnly = false;
        // TODO : these attributes are not used as default on parser (see Options ctor)
        WeightsApprox cRoundMode = weightsScalingMode("NONE");
        WeightsApprox bRoundMode = weightsScalingMode("NONE");
        WeightsApprox wtRoundMode = weightsScalingMode("NONE");
        ClippingMode wtClippingMode = parseClippingMode("None");
        ClippingMode actClippingMode = parseClippingMode("MSE");
        ScalingMode actScalingMode = parseScalingMode("Floating-point");
        // end TODO 
        bool actRescalePerOutput = false;
        double actQuantileValue = 0.9999;
        bool exportNoUnsigned = false;
        bool exportNoCrossLayerEqualization = false;
        double timeStep = 0.1;
        std::string saveTestSet = std::string();
        std::string load = std::string();
        std::string weights = std::string();
        bool ignoreNoExist = false;
        bool banMultiDevice = false;
        int exportNbStimuliMax = -1;
        bool qatSAT = false;
        bool version = false;
        std::string iniConfig;

    };
    extern unsigned int verbosity;
    void setVerboseLevel(unsigned int value);
    #ifdef CUDA
    extern unsigned int cudaDevice;
    void setCudaDeviceOption(unsigned int value);
    std::vector<unsigned int> setMultiDevices(std::string cudaDev);
    #endif

    void learnThreadWrapper(const std::shared_ptr<DeepNet>& deepNet,
                            std::vector<std::pair<std::string, double> >* timings=NULL);

    void inferThreadWrapper(const std::shared_ptr<DeepNet>& deepNet,
                            Database::StimuliSet set,
                            std::vector<std::pair<std::string, double> >* timings=NULL);
    //#define GPROF_INTERRUPT

    #if defined(__GNUC__) && !defined(NDEBUG) && defined(GPROF_INTERRUPT)
    #include <dlfcn.h>
    void sigUsr1Handler(int /*sig*/);
    #endif
    void printVersionInformation();
    void test(const Options&, std::shared_ptr<DeepNet>&, bool);
    void importFreeParameters(const Options& opt, DeepNet& deepNet);
    bool generateExport(const Options&, std::shared_ptr<DeepNet>&);
    bool calibNetwork(const Options&, std::shared_ptr<DeepNet>&);
    void generateExportFromCalibration(const Options&, std::shared_ptr<DeepNet>&, std::string="");
    void findLearningRate(const Options&, std::shared_ptr<DeepNet>&);
    void learn_epoch(const Options&, std::shared_ptr<DeepNet>&);
    void learn(const Options&, std::shared_ptr<DeepNet>&);
    void learnStdp(const Options& opt, std::shared_ptr<DeepNet>& deepNet, 
                std::shared_ptr<Environment>& env, Network& net, 
                Monitor& monitorEnv, Monitor& monitorOut);
    void testStdp(const Options&, std::shared_ptr<DeepNet>&, std::shared_ptr<Environment>&, Network&, Monitor&, Monitor&);
    void testCStdp(const Options&, std::shared_ptr<DeepNet>&);
    void logStats(const Options&, std::shared_ptr<DeepNet>&);    
}