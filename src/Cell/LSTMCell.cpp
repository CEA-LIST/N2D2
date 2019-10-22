/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Thibault ALLENET (thibault.allenet@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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

#include "Cell/LSTMCell.hpp"
#include "DeepNet.hpp"
#include "Solver/Solver.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Utils.hpp"

const char* N2D2::LSTMCell::Type = "LSTM";

N2D2::LSTMCell::LSTMCell(const DeepNet& deepNet, const std::string& name,
                unsigned int seqLength,
                unsigned int batchSize,
                unsigned int inputDim,
                unsigned int numberLayers,
                unsigned int hiddenSize,
                unsigned int algo,
                unsigned int nbOutputs,
                unsigned int bidirectional,
                unsigned int inputMode,
                float dropout,
                bool singleBackpropFeeding)
    : Cell(deepNet, name, nbOutputs),
        mSeqLength(seqLength),
        mBatchSize(batchSize),
        mInputDim(inputDim),
        mNumberLayers(numberLayers),
        mHiddenSize(hiddenSize),
        mAlgo(algo),
        mBidirectional(bidirectional),
        mInputMode(inputMode),
        mDropout(dropout),
        mSingleBackpropFeeding(singleBackpropFeeding)

{
    // ctor
}


void N2D2::LSTMCell::exportFreeParameters(const std::string
                                               & fileName) const
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    std::string weightsFile;
    std::string biasFile;

    weightsFile = fileBase + "_WeightsPreviousLayer_Layer_0" + fileExt;

    std::ofstream weights(weightsFile.c_str());

    if (!weights.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + weightsFile);
    unsigned int bds;
    bds = (mBidirectional? 2 : 1);

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int inputidx = 0; inputidx < mInputDim; ++inputidx){
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLIG1;
                getWeightPLIG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLIG1);
                weights << WPLIG1(0) << " ";
            }
            weights << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLFG1;
	            getWeightPLFG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLFG1);
                weights << WPLFG1(0) << " ";
            }
            weights << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLCG1;
	            getWeightPLCG_1stLayer(  inputidx,  hiddenidx,  bidir,WPLCG1);
                weights << WPLCG1(0) << " ";
            }
            weights << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLIG1;
	            getWeightPLOG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLIG1);
                weights << WPLIG1(0) << " ";
            }
            weights << std::endl;
            weights << std::endl;
        }
        weights << std::endl;
    }

    weightsFile = fileBase + "_WeightsRecurrent_Layer_0" + fileExt;

    std::ofstream weightsrec(weightsFile.c_str());

    if (!weights.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + weightsFile);

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRIG;
                getWeightRIG(  channelhiddenidx,  outputhiddenidx,  bidir, WRIG);
                weightsrec << WRIG(0) << " ";
            }
            weightsrec << std::endl;
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRFG;
	            getWeightRFG(  channelhiddenidx,  outputhiddenidx,  bidir, WRFG);
                weightsrec << WRFG(0) << " ";
            }
            weightsrec << std::endl;
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRCG;
	            getWeightRCG(  channelhiddenidx,  outputhiddenidx,  bidir, WRCG);
                weightsrec << WRCG(0) << " ";
            }
            weightsrec << std::endl;
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WROG;
	            getWeightROG(  channelhiddenidx,  outputhiddenidx,  bidir, WROG) ;
                weightsrec << WROG(0) << " ";
            }
            weightsrec << std::endl;
            weightsrec << std::endl;
        }
        weightsrec << std::endl;
    }

    biasFile = fileBase + "_Bias_Layer_0" + fileExt;

    std::ofstream bias(biasFile.c_str());

    if (!bias.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + biasFile);

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLIG;
            getBiasPLIG(  hiddenidx,   bidir, BPLIG);
            bias << BPLIG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLFG;
            getBiasPLFG(  hiddenidx,   bidir, BPLFG);
            bias << BPLFG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLCG;
            getBiasPLCG(  hiddenidx,   bidir, BPLCG);
            bias << BPLCG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLOG;
            getBiasPLOG(  hiddenidx,   bidir, BPLOG);
            bias << BPLOG(0) << " ";
        }
        bias << std::endl;
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRIG;
            getBiasRIG(  hiddenidx,   bidir, BRIG);
            bias << BRIG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRFG;
            getBiasRFG(  hiddenidx,   bidir, BRFG);
            bias << BRFG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRCG;
            getBiasRCG(  hiddenidx,   bidir, BRCG);
            bias << BRCG(0) << " ";
        }
        bias << std::endl;
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BROG;
            getBiasROG(  hiddenidx,   bidir, BROG);
            bias << BROG(0) << " ";
        }
        bias << std::endl;
        bias << std::endl;
        bias << std::endl;
    }



    for (unsigned int layer = 1; layer < mNumberLayers; ++layer){
        weightsFile = fileBase + "_WeightsPreviousLayer_Layer_" + std::to_string(layer) + fileExt;

        std::ofstream weightsl(weightsFile.c_str());

        if (!weights.good())
            throw std::runtime_error("Could not create parameter file: "
                                    + weightsFile);

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize*bds; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLIG;
                    getWeightPLIG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLIG);
                    weightsl << WPLIG(0)  << " ";
                }
                weightsl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLFG;
                    getWeightPLFG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLFG);
                    weightsl <<  WPLFG(0) << " ";
                }
                weightsl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLCG;
                    getWeightPLCG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLCG);
                    weightsl <<  WPLCG(0) << " ";
                }
                weightsl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLOG;
                    getWeightPLOG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLOG);
                    weightsl <<  WPLOG(0) << " ";
                }
                weightsl << std::endl;
                weightsl << std::endl;
            }
            weightsl << std::endl;
        }

        weightsFile = fileBase + "_WeightsRecurrent_Layer_" + std::to_string(layer) + fileExt;

        std::ofstream weightsrecl(weightsFile.c_str());

        if (!weights.good())
            throw std::runtime_error("Could not create parameter file: "
                                    + weightsFile);

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRIG;
                    getWeightRIG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRIG);
                    weightsrecl << WRIG(0) << " ";
                }
                weightsrecl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRFG;
                    getWeightRFG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRFG);
                    weightsrecl << WRFG(0) << " ";
                }
                weightsrecl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRCG;
                    getWeightRCG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRCG);
                    weightsrecl << WRCG(0) << " ";
                }
                weightsrecl << std::endl;
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WROG;
                    getWeightROG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WROG);
                    weightsrecl <<  WROG(0) << " ";
                }
                weightsrecl << std::endl;
                weightsrecl << std::endl;
            }
            weightsrecl << std::endl;
        }

        biasFile = fileBase + "_Bias_Layer_" + std::to_string(layer) + fileExt;

        std::ofstream biasl(biasFile.c_str());

        if (!bias.good())
            throw std::runtime_error("Could not create parameter file: "
                                    + biasFile);

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLIG;
                getBiasPLIG(  hiddenidx,   (bds*layer)+bidir, BPLIG);
                biasl <<  BPLIG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLFG;
                getBiasPLFG(  hiddenidx,   (bds*layer)+bidir, BPLFG);
                biasl <<  BPLFG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLCG;
                getBiasPLCG(  hiddenidx,   (bds*layer)+bidir,BPLCG);
                biasl <<  BPLCG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLOG;
                getBiasPLOG(  hiddenidx,   (bds*layer)+bidir, BPLOG);
                biasl <<  BPLOG(0) << " ";
            }
            biasl << std::endl;
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRIG;
                getBiasRIG(  hiddenidx,   (bds*layer)+bidir, BRIG);
                biasl <<  BRIG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRFG;
                getBiasRFG(  hiddenidx,   (bds*layer)+bidir, BRFG);
                biasl <<  BRFG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRCG;
                getBiasRCG(  hiddenidx,   (bds*layer)+bidir, BRCG );
                biasl <<  BRCG(0) << " ";
            }
            biasl << std::endl;
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BROG;
                getBiasROG(  hiddenidx,   (bds*layer)+bidir, BROG);
                biasl <<  BROG(0) << " ";
            }
            biasl << std::endl;
            biasl << std::endl;
            biasl << std::endl;
        }
    }
}



void N2D2::LSTMCell::importFreeParameters(const std::string& fileName,
                                               bool ignoreNotExists)
{

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    std::string weightsFile;
    std::string biasFile;

    weightsFile = fileBase + "_WeightsPreviousLayer_Layer_0" + fileExt;

    std::ifstream weights(weightsFile.c_str());

    if (!weights.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << weightsFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + weightsFile);
    }

    unsigned int bds;
    bds = (mBidirectional? 2 : 1);

    std::vector<size_t> dim = {(size_t) 1};

    double w;

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int inputidx = 0; inputidx < mInputDim; ++inputidx){
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLIG1(dim, 0.0);
                if (!(weights >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WPLIG1(0)= w;
                    setWeightPLIG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLIG1);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLFG1(dim, 0.0);
                if (!(weights >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WPLFG1(0)= w;
	                setWeightPLFG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLFG1);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLCG1(dim, 0.0);
                if (!(weights >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WPLCG1(0) = w;
	                setWeightPLCG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLCG1);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLOG1(dim, 0.0);
                if (!(weights >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WPLOG1(0) = w;
	                setWeightPLOG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLOG1);
                }
            }
        }
    }

    weightsFile = fileBase + "_WeightsRecurrent_Layer_0" + fileExt;

    std::ifstream weightsrec(weightsFile.c_str());

    if (!weightsrec.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << weightsFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + weightsFile);
    }

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRIG(dim, 0.0);
                if (!(weightsrec >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WRIG(0) = w;
                    setWeightRIG(  channelhiddenidx,  outputhiddenidx,  bidir, WRIG);
                }
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRFG(dim, 0.0);
                if (!(weightsrec >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WRFG(0) = w;
	                setWeightRFG(  channelhiddenidx,  outputhiddenidx,  bidir, WRFG);
                }
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRCG(dim, 0.0);
                if (!(weightsrec >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WRCG(0) = w;
	                setWeightRCG(  channelhiddenidx,  outputhiddenidx,  bidir, WRCG);
                }
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WROG(dim, 0.0);
                if (!(weightsrec >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + weightsFile);
                } else {
                    WROG(0) = w;
	                setWeightROG(  channelhiddenidx,  outputhiddenidx,  bidir, WROG);
                }
            }
        }
    }

    biasFile = fileBase + "_Bias_Layer_0" + fileExt;

    std::ifstream bias(biasFile.c_str());

    if (!bias.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << biasFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + biasFile);
    }

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLIG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BPLIG(0) = w;
                setBiasPLIG(  hiddenidx,   bidir, BPLIG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLFG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BPLFG(0) = w;
                setBiasPLFG(  hiddenidx,   bidir, BPLFG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLCG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BPLCG(0) = w;
                setBiasPLCG(  hiddenidx,   bidir, BPLCG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLOG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BPLOG(0) = w;
                setBiasPLOG(  hiddenidx,   bidir, BPLOG);
            }
        }

        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRIG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BRIG(0) = w;
                setBiasRIG(  hiddenidx,   bidir, BRIG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRFG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BRFG(0) = w;
                setBiasRFG(  hiddenidx,   bidir, BRFG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRCG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BRCG(0) = w;
                setBiasRCG(  hiddenidx,   bidir, BRCG);
            }
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BROG(dim, 0.0);
            if (!(bias >> w))
            {
                if(!ignoreNotExists)
                throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + biasFile);
            } else {
                BROG(0) = w;
                setBiasROG(  hiddenidx,   bidir, BROG);
            }
        }
    }



    for (unsigned int layer = 1; layer < mNumberLayers; ++layer){
        weightsFile = fileBase + "_WeightsPreviousLayer_Layer_" + std::to_string(layer) + fileExt;

        std::ifstream weightsl(weightsFile.c_str());

        if (!weightsl.good()) {
            if (ignoreNotExists) {
                std::cout << Utils::cnotice
                        << "Notice: Could not open synaptic file: " << weightsFile
                        << Utils::cdef << std::endl;
                return;
            } else
                throw std::runtime_error("Could not open synaptic file: "
                                        + weightsFile);
        }

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize*bds; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLIG(dim, 0.0);
                    if (!(weightsl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WPLIG(0) = w;
                        setWeightPLIG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLIG);
                    }
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLFG(dim, 0.0);
                    if (!(weightsl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WPLFG(0) = w;
                        setWeightPLFG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLFG);
                    }
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLCG(dim, 0.0);
                    if (!(weightsl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WPLCG(0) = w;
                        setWeightPLCG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLCG);
                    }
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLOG(dim, 0.0);
                    if (!(weightsl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WPLOG(0) = w;
                        setWeightPLOG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLOG);
                    }
                }
            }
        }

        weightsFile = fileBase + "_WeightsRecurrent_Layer_" + std::to_string(layer) + fileExt;

        std::ifstream weightsrecl(weightsFile.c_str());

        if (!weightsrecl.good()) {
            if (ignoreNotExists) {
                std::cout << Utils::cnotice
                        << "Notice: Could not open synaptic file: " << weightsFile
                        << Utils::cdef << std::endl;
                return;
            } else
                throw std::runtime_error("Could not open synaptic file: "
                                        + weightsFile);
        }

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRIG(dim, 0.0);
                    if (!(weightsrecl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WRIG(0) = w;
                        setWeightRIG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRIG);
                    }
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRFG(dim, 0.0);
                    if (!(weightsrecl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WRFG(0) = w;
                        setWeightRFG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRFG);
                    }
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRCG(dim, 0.0);
                    if (!(weightsrecl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WRCG(0) = w;
                        setWeightRCG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRCG);
                    }
                }Tensor<Float_T> BRIG(dim, 0.0);
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WROG(dim, 0.0);
                    if (!(weightsrecl >> w))
                    {
                        if(!ignoreNotExists)
                        throw std::runtime_error(
                                "Error while reading scale in parameter file: "
                                + weightsFile);
                    } else {
                        WROG(0) = w;
                        setWeightROG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WROG);
                    }
                }
            }
        }

        biasFile = fileBase + "_Bias_Layer_" + std::to_string(layer) + fileExt;

        std::ifstream biasl(biasFile.c_str());

        if (!biasl.good()) {
            if (ignoreNotExists) {
                std::cout << Utils::cnotice
                        << "Notice: Could not open synaptic file: " << biasFile
                        << Utils::cdef << std::endl;
                return;
            } else
                throw std::runtime_error("Could not open synaptic file: "
                                        + biasFile);
        }

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLIG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BPLIG(0) = w;
                    setBiasPLIG(  hiddenidx,   (bds*layer)+bidir, BPLIG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLFG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BPLFG(0) = w;
                    setBiasPLFG(  hiddenidx,   (bds*layer)+bidir, BPLFG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLCG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BPLCG(0) = w;
                    setBiasPLCG(  hiddenidx,   (bds*layer)+bidir, BPLCG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLOG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BPLOG(0) = w;
                    setBiasPLOG(  hiddenidx,   (bds*layer)+bidir, BPLOG);
                }
            }

            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRIG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BRIG(0) = w;
                    setBiasRIG(  hiddenidx,  (bds*layer)+bidir, BRIG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRFG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BRFG(0) = w;
                    setBiasRFG(  hiddenidx,  (bds*layer)+bidir, BRFG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRCG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BRCG(0) = w;
                    setBiasRCG(  hiddenidx,   (bds*layer)+bidir, BRCG);
                }
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BROG(dim, 0.0);
                if (!(biasl >> w))
                {
                    if(!ignoreNotExists)
                    throw std::runtime_error(
                            "Error while reading scale in parameter file: "
                            + biasFile);
                } else {
                    BROG(0) = w;
                    setBiasROG(  hiddenidx,   (bds*layer)+bidir, BROG);
                }
            }
        }
    }
}


void N2D2::LSTMCell::logFreeParametersDistrib(const std::string
                                                         & fileName) const
{
    unsigned int Dirscale;
    Dirscale = (mBidirectional? 2:1);
    // Append all weights
    std::vector<double> weights[5];

    unsigned int layer0Size,layerxSize;

	layer0Size= 4*mInputDim*mHiddenSize+4*mHiddenSize*mHiddenSize+8*mHiddenSize;
    layerxSize= 4*mHiddenSize*Dirscale*mHiddenSize + 4*mHiddenSize*mHiddenSize + 8*mHiddenSize;


    weights[0].reserve(layer0Size*Dirscale);

    for (unsigned int layer = 1; layer < mNumberLayers; ++layer){
        weights[layer].reserve(layerxSize*Dirscale);
    }


    unsigned int bds;
    bds = (mBidirectional? 2 : 1);


    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int inputidx = 0; inputidx < mInputDim; ++inputidx){
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLIG1;
                getWeightPLIG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLIG1);
                weights[0].push_back(WPLIG1(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLFG1;
	            getWeightPLFG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLFG1);
                weights[0].push_back(WPLFG1(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLCG1;
	            getWeightPLCG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLCG1);
                weights[0].push_back(WPLCG1(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> WPLOG1;
	            getWeightPLOG_1stLayer(  inputidx,  hiddenidx,  bidir, WPLOG1);
                weights[0].push_back(WPLOG1(0));
            }
        }
    }


    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRIG;
                getWeightRIG(  channelhiddenidx,  outputhiddenidx,  bidir, WRIG);
                weights[0].push_back(WRIG(0));
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRFG;
	            getWeightRFG(  channelhiddenidx,  outputhiddenidx,  bidir, WRFG);
                weights[0].push_back(WRFG(0));
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WRCG;
	            getWeightRCG(  channelhiddenidx,  outputhiddenidx,  bidir, WRCG);
                weights[0].push_back(WRCG(0));
            }
            for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                Tensor<Float_T> WROG;
	            getWeightROG(  channelhiddenidx,  outputhiddenidx,  bidir, WROG);
                weights[0].push_back(WROG(0));
            }
        }
    }

    for (unsigned int bidir = 0; bidir < bds; ++bidir) {
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLIG;
            getBiasPLIG(  hiddenidx,   bidir, BPLIG);
            weights[0].push_back(BPLIG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLFG;
            getBiasPLFG(  hiddenidx,   bidir, BPLFG);
            weights[0].push_back(BPLFG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLCG;
            getBiasPLCG(  hiddenidx,   bidir, BPLCG);
            weights[0].push_back(BPLCG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BPLOG;
            getBiasPLOG(  hiddenidx,   bidir, BPLOG);
            weights[0].push_back(BPLOG(0));
        }

        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRIG;
            getBiasRIG(  hiddenidx,   bidir, BRIG);
            weights[0].push_back(BRIG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRFG;
            getBiasRFG(  hiddenidx,   bidir, BRFG);
            weights[0].push_back(BRFG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BRCG;
            getBiasRCG(  hiddenidx,   bidir, BRCG);
            weights[0].push_back(BRCG(0));
        }
        for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
            Tensor<Float_T> BROG;
            getBiasROG(  hiddenidx,   bidir, BROG);
            weights[0].push_back(BROG(0));
        }
    }



    for (unsigned int layer = 1; layer < mNumberLayers; ++layer){

        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize*bds; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLIG;
                    getWeightPLIG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLIG);
                    weights[layer].push_back(WPLIG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLFG;
                    getWeightPLFG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLFG);
                    weights[layer].push_back(WPLFG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLCG;
                    getWeightPLCG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLCG);
                    weights[layer].push_back(WPLCG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WPLOG;
                    getWeightPLOG(  channelhiddenidx,  outputhiddenidx,  (bds*(layer-1))+bidir, WPLOG);
                    weights[layer].push_back(WPLOG(0));
                }
            }
        }


        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int channelhiddenidx = 0; channelhiddenidx < mHiddenSize; ++channelhiddenidx){
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRIG;
                    getWeightRIG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRIG);
                    weights[layer].push_back(WRIG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRFG;
                    getWeightRFG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRFG);
                    weights[layer].push_back(WRFG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WRCG;
                    getWeightRCG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WRCG);
                    weights[layer].push_back(WRCG(0));
                }
                for (unsigned int outputhiddenidx = 0; outputhiddenidx < mHiddenSize; ++outputhiddenidx){
                    Tensor<Float_T> WROG;
                    getWeightROG(  channelhiddenidx,  outputhiddenidx,  (bds*layer)+bidir, WROG);
                    weights[layer].push_back(WROG(0));
                }
            }
        }



        for (unsigned int bidir = 0; bidir < bds; ++bidir) {
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLIG;
                getBiasPLIG(  hiddenidx,   (bds*layer)+bidir, BPLIG);
                weights[layer].push_back(BPLIG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLFG;
                getBiasPLFG(  hiddenidx,   (bds*layer)+bidir,BPLFG);
                weights[layer].push_back(BPLFG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLCG;
                getBiasPLCG(  hiddenidx,   (bds*layer)+bidir, BPLCG);
                weights[layer].push_back(BPLCG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BPLOG;
                getBiasPLOG(  hiddenidx,   (bds*layer)+bidir, BPLOG );
                weights[layer].push_back(BPLOG(0));
            }

            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRIG;
                getBiasRIG(  hiddenidx,   (bds*layer)+bidir, BRIG);
                weights[layer].push_back(BRIG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRFG;
                getBiasRFG(  hiddenidx,   (bds*layer)+bidir, BRFG);
                weights[layer].push_back(BRFG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BRCG;
                getBiasRCG(  hiddenidx,   (bds*layer)+bidir, BRCG);
                weights[layer].push_back(BRCG(0));
            }
            for (unsigned int hiddenidx = 0; hiddenidx < mHiddenSize; ++hiddenidx){
                Tensor<Float_T> BROG;
                getBiasROG(  hiddenidx,   (bds*layer)+bidir, BROG);
                weights[layer].push_back(BROG(0));
            }
        }
    }

    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    for (unsigned int layer=0; layer<mNumberLayers; ++layer){
        std::sort(weights[layer].begin(), weights[layer].end());

        // Write data file
        const std::string fileBase = Utils::fileBaseName(fileName);
        std::string fileExt = Utils::fileExtension(fileName);

        if (!fileExt.empty())
            fileExt = "." + fileExt;

        std::string distribFile = fileBase + "_Layer_"+ std::to_string(layer) + fileExt;

        std::ofstream distrib(distribFile.c_str());

        if (!distrib.good())
            throw std::runtime_error("Could not create parameter file: "
                                    + distribFile);


        std::copy(weights[layer].begin(),
                weights[layer].end(),
                std::ostream_iterator<double>(distrib, "\n"));
        distrib.close();

        const std::pair<double, double> meanStdDev = Utils::meanStdDev(weights[layer]);

        std::ostringstream label;
        label << "\"Average: " << meanStdDev.first << "\\n";
        label << "Std. dev.: " << meanStdDev.second << "\"";
        label << " at graph 0.7, graph 0.8 front";

        // Plot results
        Gnuplot gnuplot;
        gnuplot.set("grid front").set("key off");
        gnuplot << "binwidth=0.0078";   // < 1/128
        gnuplot << "bin(x,width)=width*floor(x/width+0.5)";
        gnuplot.set("boxwidth", "binwidth");
        gnuplot.set("style data boxes").set("style fill solid noborder");
        gnuplot.set("xtics", "0.2");
        gnuplot.set("mxtics", "2");
        gnuplot.set("grid", "mxtics");
        gnuplot.set("label", label.str());
        gnuplot.set("yrange", "[0:]");

        gnuplot.set("style rect fc lt -1 fs solid 0.15 noborder behind");
        gnuplot.set("obj rect from graph 0, graph 0 to -1, graph 1");
        gnuplot.set("obj rect from 1, graph 0 to graph 1, graph 1");

        const double minVal = (weights[layer].front() < -1.0) ? weights[layer].front() : -1.0;
        const double maxVal = (weights[layer].back() > 1.0) ? weights[layer].back() : 1.0;
        gnuplot.setXrange(minVal - 0.05, maxVal + 0.05);

        gnuplot.saveToFile(distribFile);
        gnuplot.plot(distribFile,
                    "using (bin($1,binwidth)):(1.0) smooth freq with boxes");
    }
}

//TODO
void N2D2::LSTMCell::getStats(Stats& stats) const
{
    stats.nbNeurons = mHiddenSize;
    stats.nbNodes = 0;
}


void N2D2::LSTMCell::setOutputsDims()
{
    mOutputsDims[0] =(unsigned int)std::ceil(1);
    mOutputsDims[1] =(unsigned int)std::ceil(mHiddenSize*(mBidirectional? 2: 1));
}


