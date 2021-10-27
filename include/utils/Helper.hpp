#include <future>

#include "N2D2.hpp"
#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#ifdef N2D2_IP
#include "Quantizer/DeepNetQAT.hpp"
#endif
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
#ifdef CUDA
#include <cudnn.h>

#include "CudaContext.hpp"
#endif

using namespace N2D2;

namespace N2D2_HELPER{
    
    #ifdef CUDA
    extern unsigned int cudaDevice;
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

    class Options {
    public:
        Options();

        Options(int argc, char* argv[]);
        unsigned int seed;
        unsigned int log;
        unsigned int logEpoch;
        unsigned int report;
        unsigned int learn;
        unsigned int learnEpoch;
        int preSamples;
        unsigned int findLr;
        ConfusionTableMetric validMetric;
        unsigned int stopValid;
        bool test;
        bool testQAT;
        bool fuse;
        bool bench;
        unsigned int learnStdp;
        double presentTime;
        unsigned int avgWindow;
        int testIndex;
        int testId;
        std::string testAdv;
        bool check;
        unsigned int logOutputs;
        bool logJSON;
        bool logDbStats;
        bool logKernels;
        bool genConfig;
        std::string genExport;
        int nbBits;
        int calibration;
        bool calibrationReload;
        WeightsApprox cRoundMode;
        WeightsApprox bRoundMode;
        WeightsApprox wtRoundMode;
        ClippingMode wtClippingMode;
        ClippingMode actClippingMode;
        ScalingMode actScalingMode;
        bool actRescalePerOutput;
        double actQuantileValue;
        bool exportNoUnsigned;
        bool exportNoCrossLayerEqualization;
        double timeStep;
        std::string saveTestSet;
        std::string load;
        std::string weights;
        bool ignoreNoExist;
        bool banMultiDevice;
        int exportNbStimuliMax;
        bool qatSAT;
        bool version;
        std::string iniConfig;

    };

    void test(const Options&, std::shared_ptr<DeepNet>&, bool);
    void importFreeParameters(const Options& opt, DeepNet& deepNet);
    bool generateExport(const Options&, std::shared_ptr<DeepNet>&);
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