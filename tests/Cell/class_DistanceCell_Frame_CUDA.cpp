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

#ifdef CUDA

#include "N2D2.hpp"

#include "Cell/DistanceCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class DistanceCell_Frame_Test_CUDA : public DistanceCell_Frame_CUDA<Float_T> {
public:
    DistanceCell_Frame_Test_CUDA(const DeepNet& deepNet, 
                            const std::string& name,
                            unsigned int nbOutputs,
                            double margin,
                            double centercoef)
        : Cell(deepNet, name, nbOutputs),
          DistanceCell(deepNet, name, nbOutputs, margin, centercoef),
          DistanceCell_Frame_CUDA(deepNet, name, nbOutputs, margin, centercoef)
    {
    }
};

TEST(DistanceCell_Frame_CUDA,
     forward_margin_pytorch)
{
    // TEST COMPARE TO PYTORCH
    Network net;
    DeepNet dn(net);
    Random::mtSeed(0);

    //    ------------------------------- HYPERPARAMS --------------------------------------
    const unsigned int num_class = 7;
    const unsigned int feat_dim = 5;
    const unsigned int batch = 4;
    double margin = 0.3;
    double centercoef = 0.0;
    const unsigned int nbOutputs = num_class;

    //  ------------------------------- CLASS INSTANCE -------------------------------------
    DistanceCell_Frame_Test_CUDA Distance(dn, "Distance", nbOutputs, margin, centercoef);
    ASSERT_EQUALS(Distance.getName(), "Distance");
    ASSERT_EQUALS(Distance.getNbOutputs(), nbOutputs);

    // --------------------------------  AFFECT INPUT & DIFFOUTPUTS VALUES  ----------------
    std::vector<Float_T> inp = {0.09702432, 0.03260779, 0.44835210, 0.64129883, 0.56843597,
                                0.98161691, 0.24686563, 0.81066394, 0.06782889, 0.26821178,
                                0.64824677, 0.63800055, 0.22813201, 0.78627253, 0.43193740,
                                0.88817346, 0.13361007, 0.09437686, 0.64183086, 0.83134472};

    //Tensor<Float_T> input({1, 1, feat_dim, batch}, inp.begin(), inp.end());
    //CudaTensor<Float_T> inputs = cuda_tensor_cast<Float_T>(input);
    CudaTensor<Float_T> inputs ({1, 1, feat_dim, batch});
    for (unsigned int i = 0; i< inputs.size(); ++i){
        inputs(i) = inp[i];
    }
    
    CudaTensor<Float_T> diffOutputs({1, 1, feat_dim, batch});

    inputs.synchronizeHToD();
    Distance.addInput(inputs, diffOutputs);

    // -----------------------  AFFECT LABELS AND GRADIENTS FROM CEL ----------------------
    std::vector<Float_T> label =   {1.,-1.,-1.,-1.,-1.,-1.,-1.,
                                    -1.,-1.,-1.,-1.,-1., 1.,-1.,
                                    -1.,-1., 1.,-1.,-1.,-1.,-1.,
                                    -1.,-1.,-1.,-1., 1.,-1.,-1.};

    Tensor<Float_T> labels ({1, 1, nbOutputs, batch}, label.begin(), label.end());
    Distance.setDiffInputs(labels);
    Distance.getDiffInputs().synchronizeHToD();

    //  -------------------------------  INITIALIZE  ---------------------------------------
    Distance.initialize();

    //  -------------------------------  AFFECT MEANS VALUES  ------------------------------
    std::vector<Float_T> me =   {-0.23718369, -0.79255629, -0.42033827,  0.84314203,  0.07762730,
                                0.57348979,  0.71326137,  0.92338371,  0.24391103, -0.76223803,
                                -0.30844128,  0.06511474,  0.30207968, -0.77078927, -0.36950350,
                                0.51861346, -0.71375632,  0.91915584,  0.16888607,  0.73049676,
                                -0.68967485,  0.84039223, -0.06835222,  0.68810618, -0.13491511,
                                -0.59900522,  0.96154857, -0.27964711,  0.26722026, -0.34018731,
                                0.97479761,  0.93931770, -0.99238980,  0.48790610, -0.31445503};

    Tensor<Float_T> means({1, 1, feat_dim, num_class}, me.begin(), me.end());

    Distance.setMeans(means);
    Distance.getWeights()->synchronizeHToD();

    const Tensor<Float_T>& mMean = tensor_cast<Float_T>((*Distance.getWeights()));

    for (unsigned int o = 0; o < means.size(); ++o) {
        ASSERT_EQUALS(means(o), mMean(o));
    }

    // ------------------------------------ PROPAGATE  -------------------------------------
    Distance.propagate(false);

    // --------------------------- RETRIEVE AND TEST DISTANCES -----------------------------
    Distance.getDist().synchronizeDToH();
    const Tensor<Float_T>& dist = tensor_cast<Float_T>(Distance.getDist());

    std::vector<Float_T> dist_py = {1.82884765, 2.84457397, 3.06057787, 1.20589066, 2.03528833, 2.84290218, 4.47137165,
                                    4.71867228, 1.48963022, 3.06590271, 1.37285733, 4.46541452, 4.60782290, 4.24650478,
                                    3.37976336, 2.21483898, 4.31566715, 2.79186916, 2.24985790, 2.78375244, 2.33322525,
                                    2.99776077, 3.82012010, 4.91725111, 1.76871204, 3.95142651, 4.54989719, 3.17428017};

    Tensor<Float_T> dists_py({1, 1, num_class, batch}, dist_py.begin(), dist_py.end());

    for (unsigned int o = 0; o < dists_py.size(); ++o) {
        ASSERT_EQUALS_DELTA(dists_py(o), dist(o), 1e-5);
    }

    // ---------------------------- RETRIEVE AND TEST OUTPUTS ------------------------------

    Distance.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(Distance.getOutputs());

    std::vector<Float_T> out_py =   {-1.18875098, -1.42228699, -1.53028893, -0.60294533, -1.01764417, -1.42145109, -2.23568583,
                                    -2.35933614, -0.74481511, -1.53295135, -0.68642867, -2.23270726, -2.99508476, -2.12325239,
                                    -1.68988168, -1.10741949, -2.80518365, -1.39593458, -1.12492895, -1.39187622, -1.16661263,
                                    -1.49888039, -1.91006005, -2.45862556, -0.88435602, -2.56842709, -2.27494860, -1.58714008};

    Tensor<Float_T> outs_py({1, 1, num_class, batch}, out_py.begin(), out_py.end());

    for (unsigned int o = 0; o < outs_py.size(); ++o) {
        ASSERT_EQUALS_DELTA(outs_py(o), outputs(o), 1e-5);
    }

    //  --------------------------  AFFECT GRAD IN DIFFINPUTS  -------------------------------

    std::vector<Float_T> grad = {-0.21228614,  0.02985915,  0.02680235,  0.06775059,  0.04475192, 0.02988412,  0.01323801,
                                     0.01508844,  0.07582666,  0.03447773,  0.08038571,  0.01712532, -0.24201009,  0.01910619,
                                     0.02701523,  0.04836918, -0.24114397,  0.03624668,  0.04752964, 0.03639408,  0.04558915,
                                     0.04453522,  0.02952097,  0.01705657,  0.08233570, -0.23471712, 0.02049564,  0.04077303};

    CudaTensor<Float_T> grads({1, 1, num_class, batch});
    for (unsigned int i = 0; i< grads.size(); ++i){
        grads(i) = grad[i];
    }
    Distance.setDiffInputs(grads);
    Distance.getDiffInputs().synchronizeHToD();
    Distance.setDiffInputsValid();

    // ---------------------------- BACKPROPAGATE ----------------------------------------------
    Distance.backPropagate();

    // ---------------------------- RETRIEVE AND TEST GRAD OF MEANS ------------------------------
    Distance.getmDiffMean().synchronizeDToH();
    const Tensor<Float_T>& diffMean = tensor_cast<Float_T>(Distance.getmDiffMean());

    std::vector<Float_T> grad_mean_py = {1.95890665e-04, -1.32145077e-01, -1.80718780e-01,  3.35030258e-02, -8.94350559e-02,
                                            2.96257585e-02, -7.64411315e-02, -8.08330476e-02,  3.64945009e-02,  2.22673759e-01,
                                            -2.24153548e-01, -1.73028946e-01,  4.10942212e-02, -3.97263467e-01, -1.83633119e-01,
                                            4.37826961e-02,  2.46552110e-01, -1.33574516e-01,  8.52011368e-02, -5.06592244e-02,
                                            -3.54033768e-01,  1.59728184e-01,  2.61483807e-03,  6.06873631e-03, -2.29514748e-01,
                                            -4.00610924e-01,  1.68343663e-01, -2.95124501e-01,  1.00478455e-01, -1.12144843e-01,
                                            -2.99087912e-02, -7.18210936e-02,  1.53475374e-01,  1.38827898e-02,  1.03565387e-01};
    
    Tensor<Float_T> grads_mean_py({1, 1, feat_dim, num_class}, grad_mean_py.begin(), grad_mean_py.end());

    for (unsigned int o = 0; o < grads_mean_py.size(); ++o) {
        ASSERT_EQUALS_DELTA(grads_mean_py(o), diffMean(o), 1e-5);
    }

    // ---------------------------- RETRIEVE AND TEST GRAD OF INPUTS ------------------------------
    Distance.getDiffOutputs(0).synchronizeDToH();
    const Tensor<Float_T>& diffOutputs_cuda = tensor_cast<Float_T>(Distance.getDiffOutputs(0));

    std::vector<Float_T> grad_input_py = { 0.07976784,  0.27426407,  0.21794336, -0.14853716,  0.01123993,
                                            0.33749887, -0.26525882,  0.27468285, -0.03982272,  0.10753705,
                                            0.17357823,  0.13072076, -0.07023863,  0.40388489,  0.10565968,
                                            0.34425774, -0.26091370,  0.07067885, -0.09389018,  0.11471121};

    Tensor<Float_T> grads_input_py({1, 1, feat_dim, batch}, grad_input_py.begin(), grad_input_py.end());

    for (unsigned int o = 0; o < grads_input_py.size(); ++o) {
        ASSERT_EQUALS_DELTA(grads_input_py(o), diffOutputs_cuda(o), 1e-5);
    }
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
