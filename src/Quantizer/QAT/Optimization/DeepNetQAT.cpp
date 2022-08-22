/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "RangeStats.hpp"
#include "ScalingMode.hpp"
#include "StimuliProvider.hpp"
#include "ScalingMode.hpp"
#include "Activation/LinearActivation.hpp"
#include "Activation/LinearActivation_Frame.hpp"
#ifdef CUDA
#include "Activation/LinearActivation_Frame_CUDA.hpp"
#endif
#include "Activation/LogisticActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/RectifierActivation_Frame.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/ResizeCell.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "Export/DeepNetExport.hpp"
#include "Transformation/RangeAffineTransformation.hpp"
#include "StimuliData.hpp"
#include "Quantizer/QAT/Activation/QuantizerActivation.hpp"
#include "Quantizer/QAT/Cell/QuantizerCell.hpp"
#include "Quantizer/QAT/Optimization/DeepNetQAT.hpp"

#ifdef CUDA
#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation_Frame_CUDA.hpp"
#endif
#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation_Frame.hpp"

#ifdef CUDA
#include "Quantizer/QAT/Cell/SAT/SATQuantizerCell_Frame_CUDA.hpp"
#endif
#include "Quantizer/QAT/Cell/SAT/SATQuantizerCell_Frame.hpp"


N2D2::DeepNetQAT::DeepNetQAT(DeepNet& deepNet)
    : DeepNetQuantization(deepNet)
{
}

void N2D2::DeepNetQAT::fuseQATGraph(StimuliProvider& sp,
                                    ScalingMode actScalingMode,
                                    WeightsApprox wMode,
                                    WeightsApprox bMode,
                                    WeightsApprox cMode) {

    std::cout << "[DeepNetQAT] Fuse the QAT Graph for hardware compatibility..." << std::endl;
    
    std::cout << "[DeepNetQAT] ==> StimuliData analysis to pass in full-range " << std::endl;
    const Database::StimuliSetMask applyTo = Database::TestOnly;
    StimuliData stimuliData("stimuli_stats", sp);
    stimuliData.generate(applyTo, true);
    stimuliData.logValueRange();
    StimuliData::Value globalValue = stimuliData.getGlobalValue();
    const double databaseFirstAlpha = 1.0;
    const double databaseMaxRange = 255.0;
 
    const double databaseMaxVal = std::max(   std::abs(globalValue.minVal), 
                                        std::abs(globalValue.maxVal));
    std::cout << "[DeepNetQAT] ==> Database Max Value = " << databaseMaxVal << std::endl;
    if(databaseMaxVal > 1.0) {
        std::cout << Utils::cwarning << "[DeepNetQAT] ==> No support max alpha value superior to 1.0 ! "
        << "First Max Alpha is set to 1.0, a potential loss can appears..." 
        << Utils::cdef << std::endl;
    }

    std::cout << "[DeepNetQAT] ==> Rescale stimuli"
            << " in unsigned range [0.0 to " << databaseMaxRange << "] " << std::endl;
    sp.addTopTransformation(
        RangeAffineTransformation(RangeAffineTransformation::Multiplies, databaseMaxRange),
        Database::All
    );
    std::cout << "[DeepNetQAT] ==> Rescale first Alpha to [" 
            << databaseFirstAlpha << "] and first Range to [" <<  databaseMaxRange << "]\n"
            << std::endl;

    std::pair <size_t, size_t> ranges(databaseMaxRange, 255);
    std::pair <double, double> alphas(databaseFirstAlpha, 1.0);

    //to keep track of ranges and alpha for mobilenet-v2 type graph
    double alpha_ElWiseParent0 = 1.0;
    size_t range_ElWiseParent0 = 255;
    double alphaCurrent = 1.0;
    size_t rangeCurrent = 255;
    std::string elemWiseParent0Name = "";
    bool isParent0=false;

    int nLayer = 0;
    int nCell = 0;

    int lastCell = 10000;

    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();

    for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
        nLayer++;
        for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
            nCell++;
            //std::cout << "[DeepNetQAT] ==> nLayer = " << nLayer << " , nCell = " << nCell << std::endl;

            if(nLayer > lastCell) continue;
            if(nLayer == lastCell-1){
                std::cout << "[DeepNetQAT] ==> VAR = 0 in BatchNorm layers : " << std::endl;
                for(std::size_t it = 0; it < mVarNulName.size(); ++it){
                    std::cout << mVarNulName[it] << " : " << mVarNul[it] << std::endl;
                }
                std::cout << "[DeepNetQAT] ==> counter only : " << std::endl;
                for(std::size_t it = 0; it < mVarNulName.size(); ++it){
                    std::cout << mVarNul[it] << std::endl;
                }
            }

            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            std::cout << "[DeepNetQAT] ==> " << cell->getName() << " is type " << cell->getType() << std::endl;
            //Not sure that it will be used as it for QAT fusion...
            //++it; // increase it before being potentially invalided by removeCell()
            if (cell->getType() != BatchNormCell::Type && cell->getType() != FcCell::Type && cell->getType() != ElemWiseCell::Type) {
                //Work on case when there is pre-activation
                /*
                const std::vector<std::shared_ptr<Cell> > parents 
                        = mDeepNet.getParentCells(cell->getName());

                if(parents[0]) {
                    std::shared_ptr<Cell_Frame_Top> parentCellTop =
                        std::dynamic_pointer_cast<Cell_Frame_Top>(parents[0]);

                    if(parentCellTop->getActivation()) {
                        if(parentCellTop->getActivation()->getQuantizer()) {
                            if(std::string(parentCellTop->getActivation()->getQuantizer()->getType()) != "SAT") {
                                std::cout << Utils::cnotice <<  "  batchnorm fusion only support SAT QAT method"
                                << Utils::cdef << std::endl;
                            }
                            else {
                               ranges.first 
                                    = parentCellTop->getActivation()->getQuantizer()->getRange();
                                alphas.first = alphas.second;
                            }
                        }
                    }
                }
                */
                continue;
            }
            // check layers type
            const std::vector<std::shared_ptr<Cell> > parents  
                    = mDeepNet.getParentCells(cell->getName());

            const std::vector<std::shared_ptr<Cell> > grandParents  
                    = mDeepNet.getParentCells(parents[0]->getName());

            if(parents.size() > 1 && cell->getType() != ElemWiseCell::Type) {
                std::cout << Utils::cnotice << "[DeepNetQAT] ==> cannot fuse QAT cell \""
                    << cell->getName() << "\" because it has multiple "
                    "parents (not supported)" << Utils::cdef << std::endl;
                continue;
            }

            if(grandParents.empty() || (!grandParents[0])) {
                ranges.first = 255.0;
                //alphas.first = 255.0;
                std::cout << Utils::cnotice << "[DeepNetQAT] ==> Fusion of the QAT Graph on cell \""
                    << cell->getName() << "\" will use range 255 (8-bits) for the input stimuli (default range) " 
                    << Utils::cdef << std::endl;
            }

            if(cell->getType() == BatchNormCell::Type) {
                if(!parents[0] || parents[0]->getType() != ConvCell::Type) {
                    std::cout << Utils::cnotice << "[DeepNetQAT] ==> cannot fuse BatchNorm \""
                        << cell->getName() << "\" because parent cell (\""
                        << ((parents[0]) ? parents[0]->getName() : "env")
                        << "\") is not a Conv" << Utils::cdef << std::endl;

                    continue;
                }

                // only a single Conv is preceding
                // check if BatchNorm is the only child
                const std::vector<std::shared_ptr<Cell>> convChilds
                    = mDeepNet.getChildCells(parents[0]->getName());

                if (convChilds.size() != 1) {
                    std::cout << Utils::cnotice << "[DeepNetQAT] ==> cannot fuse BatchNorm \""
                        << cell->getName() << "\" because parent Conv "
                        "(\"" << parents[0]->getName() << "\") has multiple "
                        "childs" << Utils::cdef << std::endl;
                    
                    continue;
                }

                assert(convChilds[0] == cell);

                //if it's the last cell, insert rescaling cell after it
                if(nLayer == lastCell){
                    std::cout << "[DeepNetQAT] ==> Last layer to be fused..." << std::endl;
                    std::shared_ptr<BatchNormCell> bnCell0 =
                    std::dynamic_pointer_cast<BatchNormCell>(cell);

                    std::shared_ptr<Cell_Frame_Top> bnCellTop =
                        std::dynamic_pointer_cast<Cell_Frame_Top>(bnCell0);

                    if(bnCellTop->getActivation()) {
                            if(bnCellTop->getActivation()->getQuantizer()) {
                                if(std::string(bnCellTop->getActivation()->getQuantizer()->getType()) == "SAT") {
                                    const std::shared_ptr<QuantizerActivation>& quantizerActivation
                                        = bnCellTop->getActivation()->getQuantizer();
                                    if(bnCellTop->getActivation()->getQuantizer()->isCuda()) {
                                        #ifdef CUDA
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alpha.synchronizeHToD();
                                        alphaCurrent = alpha(0);
                                        #endif
                                    }
                                    else{
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alphaCurrent = alpha(0);
                                    }
                                    rangeCurrent = bnCellTop->getActivation()->getQuantizer()->getRange();
                                }
                            }
                    }
                    std::vector<Float_T> scalingPerOutput
                    = std::vector<Float_T>(bnCell0->getNbOutputs(), 0.0f);
                    for (std::size_t output = 0; output < bnCell0->getNbOutputs(); ++output) {
                        //scalingPerOutput[output] = alphaCurrent/rangeCurrent;
                        scalingPerOutput[output] = alphaCurrent/(double)rangeCurrent;
                        std::cout << "[DeepNetQAT] ==> rescaling factor for Cell :: "
                        << "\"  scalingPerOutput[output] = " << scalingPerOutput[output] << "\""
                        << std::endl;
                    }

                    std::cout << "[DeepNetQAT] ==> rescaling factor for Cell :: "
                        << "\"  alphaCurrent = " << alphaCurrent << "\""
                        << "\"  rangeCurrent = " << rangeCurrent << "\""
                        << std::endl;

                    bool isClipped = false;
                    auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*bnCell0))
                                (mDeepNet,
                                    mDeepNet.generateNewCellName(bnCell0->getName() + "_rescale_act"),
                                    bnCell0->getNbOutputs(),
                                    Scaling::floatingPointScaling(
                                        scalingPerOutput,
                                        isClipped,
                                        std::vector<Float_T>(0.0f)) );

                        //add scaling cell after current BN cell
                        mDeepNet.addCellAfter(scalingCell, bnCell0);
                        ++it;
                }

                //check if BN has more than one child and ElementWise is one of them
                const std::vector<std::shared_ptr<Cell>> bnChilds
                    = mDeepNet.getChildCells(cell->getName());

                for(unsigned int iChild = 0; iChild < bnChilds.size(); iChild++){
                    std::cout << "[DeepNetQAT] ==> BatchNorm child[" << iChild
                              << "]->" << bnChilds[iChild]->getName()
                        << std::endl;

                    //parent 0 (before the split into 2 branches)
                    if(bnChilds[iChild]->getType() == ElemWiseCell::Type && bnChilds.size() > 1){

                        std::cout << Utils::cnotice << "[DeepNetQAT] ==> Will save \""
                            << cell->getName() << "\" activation alpha and range for ElementWise quantization !"
                            << Utils::cdef << std::endl;
                        elemWiseParent0Name = cell->getName();

                        std::shared_ptr<Cell_Frame_Top> bnCellTop =
                            std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

                        if(bnCellTop->getActivation()) {
                            if(bnCellTop->getActivation()->getQuantizer()) {
                                if(std::string(bnCellTop->getActivation()->getQuantizer()->getType()) == "SAT") {
                                    const std::shared_ptr<QuantizerActivation>& quantizerActivation
                                        = bnCellTop->getActivation()->getQuantizer();
                                    if(bnCellTop->getActivation()->getQuantizer()->isCuda()) {
                                        #ifdef CUDA
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alpha.synchronizeHToD();
                                        alpha_ElWiseParent0 = alpha(0);
                                        #endif
                                    }
                                    else{
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alpha_ElWiseParent0 = alpha(0);
                                    }
                                    range_ElWiseParent0 = bnCellTop->getActivation()->getQuantizer()->getRange();
                                }
                            }
                        }
                    }

                    //parent 1
                    if(bnChilds[iChild]->getType() == ElemWiseCell::Type && bnChilds.size() == 1){
                        //have to rescale the output of this fused bn as
                        //(range_ElWiseParent0/alpha_ElWiseParent0 * alphaCurrent/rangeCurrent)
                        //before passing it to the elementWise for Sum
                        //insert scaling cell after BN to rescale the outputs
                        std::cout << "[DeepNetQAT] ==> Adding rescaling cell after \"" << cell->getName()
                        << "\" with rescaling factor levels from \"" << cell->getName() << "\""
                        << "\" and \"" << elemWiseParent0Name << "\""
                        << std::endl;

                        std::shared_ptr<BatchNormCell> currentBnCell =
                        std::dynamic_pointer_cast<BatchNormCell>(cell);
                        bool isClipped = false;

                        //get its own alpha and range for the rescaling cell
                        std::shared_ptr<Cell_Frame_Top> bnCellTop =
                            std::dynamic_pointer_cast<Cell_Frame_Top>(currentBnCell);

                        if(bnCellTop->getActivation()) {
                            if(bnCellTop->getActivation()->getQuantizer()) {
                                if(std::string(bnCellTop->getActivation()->getQuantizer()->getType()) == "SAT") {
                                    const std::shared_ptr<QuantizerActivation>& quantizerActivation
                                        = bnCellTop->getActivation()->getQuantizer();
                                    if(bnCellTop->getActivation()->getQuantizer()->isCuda()) {
                                        #ifdef CUDA
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alpha.synchronizeHToD();
                                        alphaCurrent = alpha(0);
                                        #endif
                                    }
                                    else{
                                        const auto quantizerSAT
                                            = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                                        const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                                        alphaCurrent = alpha(0);
                                    }
                                    rangeCurrent = bnCellTop->getActivation()->getQuantizer()->getRange();
                                }
                            }
                        }

                        std::vector<Float_T> scalingPerOutput
                        = std::vector<Float_T>(currentBnCell->getNbOutputs(), 0.0f);

                        std::cout << "[DeepNetQAT] ==> Scaling factor is computed using :: "
                        << "\"  range_ElWiseParent0 = " << range_ElWiseParent0 << "\""
                        << "\"  alpha_ElWiseParent0 = " << alpha_ElWiseParent0 << "\""
                        << "\"  alphaCurrent = " << alphaCurrent << "\""
                        << "\"  rangeCurrent = " << rangeCurrent << "\""
                        << std::endl;

                        for (std::size_t output = 0; output < currentBnCell->getNbOutputs(); ++output) {
                            scalingPerOutput[output] = ((float)range_ElWiseParent0/alpha_ElWiseParent0)*(alphaCurrent/(float)rangeCurrent);
                            std::cout << "[DeepNetQAT] ==> scalingPerOutput: " << scalingPerOutput[output] << std::endl;
                        }
                        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*currentBnCell))
                                (mDeepNet,
                                    mDeepNet.generateNewCellName(currentBnCell->getName() + "_rescale_act"),
                                    currentBnCell->getNbOutputs(),
                                    Scaling::floatingPointScaling(
                                        scalingPerOutput,
                                        isClipped,
                                        std::vector<Float_T>(0.0f)) );

                        //add scaling cell after the current BN cell
                        mDeepNet.addCellAfter(scalingCell, cell);
                        ++it;
                    }
                }

                // OK, Conv's only child is BatchNorm, fuse them...
                std::shared_ptr<ConvCell> convCell =
                    std::dynamic_pointer_cast<ConvCell>(parents[0]);
                std::shared_ptr<BatchNormCell> bnCell =
                    std::dynamic_pointer_cast<BatchNormCell>(cell);

                // Replace BatchNorm by Conv for BatchNorm childs
                // and BatchNorm cell removal from DeepNet
                if(QuantizeAndfuseBatchNormWithConv( ranges, alphas, convCell, bnCell, actScalingMode, wMode, bMode, cMode)) {
                    mDeepNet.removeCell(cell, true);
                    --it;
                    std::cout << "[DeepNetQAT] ==> " << convCell->getName() 
                        << " have been properly quantized and fused with " 
                        << bnCell->getName() << "\n"
                        << std::endl;
                    break;
                }
            }
            else if(cell->getType() == FcCell::Type) {
                if(!parents[0]) {
                    std::cout << Utils::cnotice << "[DeepNetQAT] ==> cannot optimize Fc \""
                        << cell->getName() << "\" because parent cell (\""
                        << ((parents[0]) ? parents[0]->getName() : "env")
                        << "\") is not a layer" << Utils::cdef << std::endl;

                    continue;
                }
                std::cout << "[DeepNetQAT] ==> Quantized fully-connected \"" << cell->getName()
                    << "\" with activation levels from \"" << parents[0]->getName() << "\""
                    << std::endl;
                std::shared_ptr<FcCell> fcCell =
                    std::dynamic_pointer_cast<FcCell>(cell);

                if(QuantizeFC( ranges, alphas, fcCell, wMode, bMode, cMode)) {
                    std::cout <<"[DeepNetQAT] ==> " <<  cell->getName() << " have been properly quantized\n" << std::endl;
                } 
            }
            else if(cell->getType() == ElemWiseCell::Type) {
                isParent0 = false;
                if(!parents[0]) {
                    std::cout << Utils::cnotice << "[DeepNetQAT] ==> cannot optimize ElemWiseCell \""
                        << cell->getName() << "\" because parent cell (\""
                        << ((parents[0]) ? parents[0]->getName() : "env")
                        << "\") is not a layer" << Utils::cdef << std::endl;

                    continue;
                }
                std::cout << "[DeepNetQAT] ==> Quantizing element-wise \"" << cell->getName()
                    << "\" with activation levels from \"" << elemWiseParent0Name << "\""
                    << std::endl;

                std::shared_ptr<ElemWiseCell> elemWiseCell =
                    std::dynamic_pointer_cast<ElemWiseCell>(cell);

                //check if ElementWise has more than one child and ElementWise is one of them
                const std::vector<std::shared_ptr<Cell>> elWiseChilds
                    = mDeepNet.getChildCells(cell->getName());

                for(unsigned int iChild = 0; iChild < elWiseChilds.size(); iChild++){
                    std::cout << "[DeepNetQAT] ==> ElementWise child[" << iChild
                              << "]->" << elWiseChilds[iChild]->getName()
                        << std::endl;
                    std::cout  << std::endl;
                    //this is parent 0 (before the split into 2 branches)
                    if(elWiseChilds[iChild]->getType() == ElemWiseCell::Type && elWiseChilds.size() > 1){
                        std::cout << Utils::cnotice << "[DeepNetQAT] ==> Will save \""
                            << cell->getName() << "\" activation alpha and range for the next ElementWise quantization !"
                            << Utils::cdef << std::endl;
                        elemWiseParent0Name = cell->getName();
                        isParent0 = true;
                    }
                }

                std::cout << "[DeepNetQAT] ==> Parent0 parameters : "
                << "\"  range_ElWiseParent0 = " << range_ElWiseParent0 << "\""
                << "\"  alpha_ElWiseParent0 = " << alpha_ElWiseParent0 << "\""
                << std::endl;

                if(QuantizeElemWise( ranges, alphas, range_ElWiseParent0, alpha_ElWiseParent0, elemWiseCell)) {
                    std::cout << "[DeepNetQAT] ==> " << cell->getName() << " have been properly quantized " << std::endl;
                }

                //set new Parent0 parameters for the next elementWise cell
                if(isParent0){
                    range_ElWiseParent0 = ranges.first;
                    alpha_ElWiseParent0 = alphas.first;

                    std::cout << "[DeepNetQAT] ==> Parent0 parameters : "
                    << "\"  range_ElWiseParent0 = " << range_ElWiseParent0 << "\""
                    << "\"  alpha_ElWiseParent0 = " << alpha_ElWiseParent0 << "\""
                    << std::endl;
                }

            }
        }
    }
}


bool N2D2::DeepNetQAT::QuantizeElemWise( std::pair <size_t, size_t>& rangeElWise,
                                        std::pair <double, double>& alphaElWise,
                                        size_t rangeParent0,
                                        double alphaParent0,
                                        const std::shared_ptr<ElemWiseCell>& elemWiseCell)
{

    std::cout << "[DeepNetQAT] ==> Quantized Layer \"" << elemWiseCell->getName() << "\"" << std::endl;

    std::shared_ptr<Cell_Frame_Top> elemWiseCellTop =
        std::dynamic_pointer_cast<Cell_Frame_Top>(elemWiseCell);

    //for element wise layer, use the alpha and range from parent0
    //parent1 was already rescaled by scaling cell
    rangeElWise.first = rangeParent0;
    alphaElWise.first = alphaParent0;

    if(elemWiseCellTop->getActivation()) {
        if(elemWiseCellTop->getActivation()->getQuantizer()) {
            if(std::string(elemWiseCellTop->getActivation()->getQuantizer()->getType()) != "SAT") {
                std::cout << Utils::cnotice
                <<  "[DeepNetQAT] ==> quantization only support SAT QAT method for cell "
                << elemWiseCell->getName()
                << Utils::cdef << std::endl;
                return false;
            }

            const std::shared_ptr<QuantizerActivation>& quantizerActivation
                = elemWiseCellTop->getActivation()->getQuantizer();

            if(elemWiseCellTop->getActivation()->getQuantizer()->isCuda()) {
                #ifdef CUDA
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alpha.synchronizeDToH();
                alphaElWise.second = alpha(0);
                #endif
            }
            else {
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alphaElWise.second = alpha(0);
            }
            rangeElWise.second
                    = elemWiseCellTop->getActivation()->getQuantizer()->getRange();
            std::cout << Utils::cnotice
                << "[DeepNetQAT] ==> Dynamic range: [" << rangeElWise.first << "->"  << rangeElWise.second << "]\n"
                << "[DeepNetQAT] ==> Activation thresholds: [" << alphaElWise.first << "->"  << alphaElWise.second << "]"
                << Utils::cdef
                << std::endl;
        }
    }

    std::vector<Float_T> clipPerOutput
        = std::vector<Float_T>(elemWiseCell->getNbOutputs(), 0.0f);
    std::vector<Float_T> scalingPerOutput
        = std::vector<Float_T>(elemWiseCell->getNbOutputs(), 0.0f);

    double scaleFactorInput = (double)rangeElWise.first / alphaElWise.first;
    double scaleFactorOutput = (double)rangeElWise.second / alphaElWise.second;

    for (std::size_t output = 0; output < elemWiseCell->getNbOutputs(); ++output) {
        scalingPerOutput[output] = scaleFactorOutput / scaleFactorInput;
        clipPerOutput[output] = alphaElWise.second * scaleFactorInput;

        std::cout   << "[DeepNetQAT] ==> clipPerOutput: " << clipPerOutput[output] << "\n"
                    << "    scalingPerOutput: " << scalingPerOutput[output] << std::endl;
    }

    std::shared_ptr<Activation> elemWiseCellActivation = elemWiseCellTop->getActivation();
    bool isClipped = false;

    if(elemWiseCellActivation) {
        if(elemWiseCellActivation->getQuantizer()) {
            std::cout << Utils::cnotice
            <<  "[DeepNetQAT] ==> Quantization is also apply on outputs activations for cell "
            << elemWiseCell->getName()
            << Utils::cdef << std::endl;
            elemWiseCell->setQuantized(rintf(log2(rangeElWise.second)) );
            isClipped = true;
        }
        else {
            std::cout << Utils::cnotice <<  "[DeepNetQAT] ==> "
                << elemWiseCell->getName()
                <<  "   activation doesn't have quantizer "
                << elemWiseCell->getName()
            << Utils::cdef << std::endl;
        }
    }
    else {
        #ifdef CUDA
        const std::shared_ptr<Activation>& activation
                = std::make_shared<LinearActivation_Frame_CUDA<float> >();
        elemWiseCellTop->setActivation(activation);

        std::cout << Utils::cnotice << "[DeepNetQAT] ==> "
        << elemWiseCell->getName()
        <<  "   doesn't have output activation and quantizer "
        << elemWiseCell->getName()
        << Utils::cdef << std::endl;
        #endif
    }

    if(elemWiseCellActivation){
        #ifdef CUDA
        //reset the activation
        const std::shared_ptr<Activation>& activationRelu
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
        elemWiseCellTop->setActivation(activationRelu);
        std::shared_ptr<Activation> elemWiseCellActivationReLu = elemWiseCellTop->getActivation();
        elemWiseCellActivationReLu->setParameter<double>("Clipping", (double)rangeElWise.second);

        isClipped = true;
        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*elemWiseCell))
                                (mDeepNet,
                                    mDeepNet.generateNewCellName(elemWiseCell->getName() + "_clip_then_rescale_act"),
                                    elemWiseCell->getNbOutputs(),
                                    Scaling::floatingPointScaling(
                                    scalingPerOutput,
                                    isClipped,
                                    clipPerOutput) );

        elemWiseCellActivationReLu->setQuantizer(std::shared_ptr<QuantizerActivation>());
        elemWiseCellActivationReLu->setActivationScaling( std::move(scalingCell->getScaling()) );
        #endif
    }

    //Switch range and alpha
    rangeElWise.first = rangeElWise.second;
    alphaElWise.first = alphaElWise.second;
    return true;
}

bool N2D2::DeepNetQAT::QuantizeFC(  std::pair <size_t, size_t>& rangeOpFc,
                                    std::pair <double, double>& alphaOpFC,
                                    const std::shared_ptr<FcCell>& fcCell,
                                    WeightsApprox wMode,
                                    WeightsApprox bMode,
                                    WeightsApprox cMode)
{
    // OK, Conv's only child is BatchNorm, fuse them...
    std::cout << "[DeepNetQAT] ==> Quantized Layer \"" << fcCell->getName() << "\"" << std::endl;
    const bool noBias = fcCell->getParameter<bool>("NoBias");

    std::shared_ptr<Cell_Frame_Top> fcCellTop =
        std::dynamic_pointer_cast<Cell_Frame_Top>(fcCell);

    if(fcCellTop->getActivation()) {
        if(fcCellTop->getActivation()->getQuantizer()) {
            if(std::string(fcCellTop->getActivation()->getQuantizer()->getType()) != "SAT") {
                std::cout << Utils::cnotice 
                <<  "[DeepNetQAT] ==> quantization only support SAT QAT method for cell "
                << fcCell->getName()
                << Utils::cdef << std::endl;
                return false;
            }

            const std::shared_ptr<QuantizerActivation>& quantizerActivation
                = fcCellTop->getActivation()->getQuantizer();

            if(fcCellTop->getActivation()->getQuantizer()->isCuda()) {
                #ifdef CUDA
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alpha.synchronizeDToH();
                alphaOpFC.second = alpha(0);
                #endif
            } 
            else {
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alphaOpFC.second = alpha(0);
            }
            rangeOpFc.second 
                    = fcCellTop->getActivation()->getQuantizer()->getRange();
            std::cout << Utils::cnotice 
                << "[DeepNetQAT] ==> Dynamic range: [" << rangeOpFc.first << "->"  << rangeOpFc.second << "]\n" 
                << "[DeepNetQAT] ==> Activation thresholds: [" << alphaOpFC.first << "->"  << alphaOpFC.second << "]" 
                << Utils::cdef 
                << std::endl;
        }
    }
    /*
        else {
            return false;
        }
    }
    else {
        return false;
    }
    */
    long double weightsScaleFactor = 1.0; //std::floor((rangeOpFc.first) / 2.0);
    long double biasScaleFactor = weightsScaleFactor;
    // FC weights are rescaled with variance
    // take this into account when scale and round weights
    long double SAT_scaling = 1.0;

    if(fcCell->getQuantizer()) {
        const std::shared_ptr<QuantizerCell>& quantizerCell
                = fcCell->getQuantizer();
        weightsScaleFactor = 1.0;

        if(wMode == WeightsApprox::RINTF) {

            if((fcCell->getQuantizer()->getRange()) > 1){
                weightsScaleFactor = std::floor((quantizerCell->getRange()) / 2.0);
            }
            else{
                weightsScaleFactor = 1;
            }

            if(std::string(quantizerCell->getType()) == "SAT"){
                if(fcCell->getQuantizer()->isCuda()) {
                    #ifdef CUDA
                    const auto quantizerCellSAT
                            = std::dynamic_pointer_cast<SATQuantizerCell_Frame_CUDA<float>>(quantizerCell);
                    SAT_scaling = quantizerCellSAT->getSAT_scaling();
                    #endif
                }
                else{
                    const auto quantizerCellSAT
                            = std::dynamic_pointer_cast<SATQuantizerCell_Frame<float>>(quantizerCell);
                    SAT_scaling = quantizerCellSAT->getSAT_scaling();
                }
            }
        }
        biasScaleFactor = weightsScaleFactor*SAT_scaling;
    }

    //if (noBias) {
    //    fcCell->setParameter<bool>("NoBias", false);
    //}

    std::shared_ptr<Activation> fcCellActivation = fcCellTop->getActivation();
    fcCellTop->synchronizeToH(false);
    std::vector<Float_T> biasPerOutput 
        = std::vector<Float_T>(fcCell->getNbOutputs(), 0.0f);
    std::vector<Float_T> clipPerOutput 
        = std::vector<Float_T>(fcCell->getNbOutputs(), 0.0f);
    std::vector<Float_T> scalingPerOutput 
        = std::vector<Float_T>(fcCell->getNbOutputs(), 0.0f);

    double scaleFactorInput = rangeOpFc.first / alphaOpFC.first;
    double scaleFactorOutput = rangeOpFc.first;// / alphaOpFC.first;;
    if(fcCellTop->getActivation()->getQuantizer()) {
        scaleFactorOutput = rangeOpFc.second / alphaOpFC.second;
    }
    for (std::size_t output = 0; output < fcCell->getNbOutputs(); ++output) {
        // Biases adjustments
        Tensor<double> bias;

        if (!noBias)
            fcCell->getBias(output, bias);

        // Weights Quantization adjustments
        for (std::size_t channel = 0; channel < fcCell->getNbChannels(); ++channel) {
            Tensor<double> kernel;

            fcCell->getQuantWeight(output, channel, kernel);

            for (std::size_t index = 0; index < kernel.size(); ++index) {
                kernel(index) = kernel(index)*(weightsScaleFactor*SAT_scaling);
                if(wMode == WeightsApprox::RINTF){
                    kernel(index) = rintf(kernel(index));
                }
            }
            fcCell->setWeight(output, channel, kernel);
        }

        scalingPerOutput[output] = (scaleFactorOutput / scaleFactorInput) * (1.0 / (SAT_scaling*weightsScaleFactor));
        clipPerOutput[output] = alphaOpFC.second * scaleFactorInput * (weightsScaleFactor*SAT_scaling);
        if (!noBias)
            biasPerOutput[output]  = bias(0) * scaleFactorInput * (biasScaleFactor);
        if(clipPerOutput[output] < 0.0) {
            std::cout << Utils::cwarning << "[DeepNetQAT] ==> clip[" << output << "] = " << clipPerOutput[output]
            << "is negatif ! Is the network correctly trained? Clipping value will be set to 0"
                << Utils::cdef << std::endl;
            clipPerOutput[output] = 0.0;
        }

        if(cMode == WeightsApprox::RINTF)
            clipPerOutput[output] = rintf(clipPerOutput[output]);

        if(bMode == WeightsApprox::RINTF && !noBias)
            biasPerOutput[output] = rintf(biasPerOutput[output]);

        
        //std::cout << "    biasPerOutput: " << biasPerOutput[output] << "\n"
        //        << "    clipPerOutput: " << clipPerOutput[output] << "\n"
        //        << "    scalingPerOutput: " << scalingPerOutput[output] << std::endl;
        
        if (!noBias) {
            bias(0) = biasPerOutput[output];
            fcCell->setBias(output, bias);
        }
    }


    fcCellTop->synchronizeToD(true);

    std::size_t weightsRange = 1;
    if(fcCell->getQuantizer()) {
        weightsRange = fcCell->getQuantizer()->getRange();
        //if 1b weights, set range to 2
        //to get correct nbBits for setQuantized
        if(weightsRange == 1){
            weightsRange = 2;
        }
    }
    //Set weights range
    fcCell->setQuantized(rintf(log2(weightsRange)));

    bool isClipped = false;
    std::size_t activationsRange = std::pow(2, 32) - 1;
    if(fcCellActivation) {
        if(fcCellActivation->getQuantizer()) {
            std::cout << Utils::cnotice 
            <<  "[DeepNetQAT] ==> Quantization is also apply on outputs activations for cell "
            << fcCell->getName()
            << Utils::cdef << std::endl;
            activationsRange = rangeOpFc.second;
            fcCellActivation->setQuantized(rintf(log2(activationsRange)) );
            isClipped = true;
        }
        else {
            std::cout << Utils::cnotice  << "[DeepNetQAT] ==> "
                << fcCell->getName()
                <<  "   activation doesn't have quantizer "
                << fcCell->getName()
            << Utils::cdef << std::endl;
        }
    }
    else {
        if(fcCellTop->isCuda()){
            #ifdef CUDA
            const std::shared_ptr<Activation>& activation
                = std::make_shared<LinearActivation_Frame_CUDA<float> >();
            fcCellTop->setActivation(activation);
            #endif
        }
        else{
            const std::shared_ptr<Activation>& activation
                = std::make_shared<LinearActivation_Frame<float> >();
            fcCellTop->setActivation(activation);
        }
        std::cout << Utils::cnotice << "[DeepNetQAT] ==> "
        << fcCell->getName()
        <<  "   doesn't have output activation and quantizer "
        << fcCell->getName()
        << Utils::cdef << std::endl;
    }
    if(fcCellActivation) {
        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*fcCell))
                                (mDeepNet, 
                                    mDeepNet.generateNewCellName(fcCell->getName() + "_clip_then_rescale_act"), 
                                    fcCell->getNbOutputs(), 
                                    Scaling::floatingPointScaling(
                                    scalingPerOutput,
                                    isClipped,
                                    clipPerOutput) );
        //Delete QuantizerCell from fcCell
        fcCell->setQuantizer(std::shared_ptr<QuantizerCell>());
        fcCellActivation->setQuantizer(std::shared_ptr<QuantizerActivation>());
        fcCellActivation->setActivationScaling( std::move(scalingCell->getScaling()) );
        fcCellActivation->setQuantized(rintf(log2(activationsRange)) );
        std::cout << Utils::cnotice << "[DeepNetQAT] ==> "
            <<  fcCell->getName() << " activations are quantized on "
            << rintf(log2(activationsRange)) << " bits"
            << Utils::cdef << std::endl;
    }

    //Switch range and alpha
    rangeOpFc.first = rangeOpFc.second;
    alphaOpFC.first = alphaOpFC.second;
    return true;
    
}


bool N2D2::DeepNetQAT::QuantizeAndfuseBatchNormWithConv(std::pair <size_t, size_t>& rangeConvBN,
                                                        std::pair <double, double>& alphasConvBN,
                                                        const std::shared_ptr<ConvCell>& convCell, 
                                                        const std::shared_ptr<BatchNormCell>& bnCell,
                                                        ScalingMode actScalingMode,
                                                        WeightsApprox wMode,
                                                        WeightsApprox bMode,
                                                        WeightsApprox cMode) 
{
    // OK, Conv's only child is BatchNorm, fuse them...
    std::cout << "[DeepNetQAT] ==> Fuse BatchNorm \"" << bnCell->getName()
        << "\" with Conv \"" << convCell->getName() << "\""
        << std::endl;
    const bool noBias = convCell->getParameter<bool>("NoBias");

    const Tensor<double>& bnScales = tensor_cast<double>(*(bnCell->getScales()));
    const Tensor<double>& bnBiases = tensor_cast<double>(*(bnCell->getBiases()));
    const Tensor<double>& bnMeans = tensor_cast<double>(*(bnCell->getMeans()));
    const Tensor<double>& bnVariances = tensor_cast<double>(*(bnCell->getVariances()));
    const double eps = bnCell->getParameter<double>("Epsilon");

    assert(bnScales.size() == convCell->getNbOutputs());
    assert(bnBiases.size() == convCell->getNbOutputs());
    assert(bnMeans.size() == convCell->getNbOutputs());
    assert(bnVariances.size() == convCell->getNbOutputs());
    assert(eps > 0.0);

    int factor0Count = 0;

    std::shared_ptr<Cell_Frame_Top> convCellTop =
        std::dynamic_pointer_cast<Cell_Frame_Top>(convCell);
    std::shared_ptr<Cell_Frame_Top> bnCellTop =
        std::dynamic_pointer_cast<Cell_Frame_Top>(bnCell);

    // Fuse only if  the convolution has a linear activation
    if (convCellTop->getActivation()
        && std::string(convCellTop->getActivation()->getType()) != "Linear") {
        std::cout << Utils::cwarning << "[DeepNetQAT] ==> non-linear "
            "activation before BatchNorm prevents fuse!"
            << Utils::cdef << std::endl;
        return false;
    }

    if(bnCellTop->getActivation()) {
        if(bnCellTop->getActivation()->getQuantizer()) {
            if(std::string(bnCellTop->getActivation()->getQuantizer()->getType()) != "SAT") {
                std::cout << Utils::cnotice 
                <<  "[DeepNetQAT] ==> batchnorm fusion only support SAT QAT method for cell "
                << bnCell->getName()
                << Utils::cdef << std::endl;
                return false;
            }

            const std::shared_ptr<QuantizerActivation>& quantizerActivation
                = bnCellTop->getActivation()->getQuantizer();

            if(bnCellTop->getActivation()->getQuantizer()->isCuda()) {
                #ifdef CUDA
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame_CUDA<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alpha.synchronizeHToD();
                alphasConvBN.second = alpha(0);
                #endif
            } 
            else {
                const auto quantizerSAT
                    = std::dynamic_pointer_cast<SATQuantizerActivation_Frame<float>>(quantizerActivation);
                const Tensor<Float_T>& alpha = tensor_cast<Float_T>(quantizerSAT->getAlpha());
                alphasConvBN.second = alpha(0);
            }
            rangeConvBN.second 
                    = bnCellTop->getActivation()->getQuantizer()->getRange();

            std::cout << Utils::cnotice 
                << "[DeepNetQAT] ==> Dynamic range: [" << rangeConvBN.first << "->"  << rangeConvBN.second << "]\n" 
                << "[DeepNetQAT] ==> Activation thresholds: [" << alphasConvBN.first << "->"  << alphasConvBN.second << "]" 
                << Utils::cdef 
                << std::endl;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }

    long double weightsScaleFactor = 1.0;

    if(wMode == WeightsApprox::RINTF) {
        weightsScaleFactor = std::floor(rangeConvBN.first / 2.0);
    }

    long double biasScaleFactor = weightsScaleFactor;
    double corrFactor = 1.0;

    if(convCell->getQuantizer()) {

        if(wMode == WeightsApprox::RINTF) {
            if((convCell->getQuantizer()->getRange()) > 1){
                weightsScaleFactor = std::floor((convCell->getQuantizer()->getRange()) / 2.0);
            }
            else{
                weightsScaleFactor = 1;
            }
        }

        //if weights are quantized to 4b with SAT default method ->
        //rescale range to 5b, as it gives the best results
        //as the values w/o round are very close to real INT
        if(convCell->getQuantizer()->getQuantMode() == QuantizerCell::Default){
            long double bitRangeThreshold = 8.0;
            if(wMode == WeightsApprox::RINTF) {
                bitRangeThreshold = 7.0;
            }
            if(weightsScaleFactor == bitRangeThreshold){
                weightsScaleFactor = 15.0;
                corrFactor = 2.066666666666667;
                rangeConvBN.first = 31;
            }
        }

        biasScaleFactor = weightsScaleFactor;
    }

    if(convCellTop->isCuda()){
        #ifdef CUDA
        const std::shared_ptr<Activation>& activation
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
        convCellTop->setActivation(activation);
        #endif
    }
    else{
        const std::shared_ptr<Activation>& activation
                = std::make_shared<RectifierActivation_Frame<float> >();
        convCellTop->setActivation(activation);
    }

    std::shared_ptr<Activation> convCellActivation = convCellTop->getActivation();

    if (noBias)
        convCell->setParameter<bool>("NoBias", false);
    
    double meanVariance = 0.0;
    unsigned int count = 0;

    for (std::size_t output = 0; output < convCell->getNbOutputs(); ++output) {
        if (bnVariances(output) > 1.0e-12) {
            meanVariance += bnVariances(output);
            ++count;
        }
        else {
            factor0Count++;
        }
    }

    if (count > 0)
        meanVariance /= count;
    else {
        std::cout << Utils::cwarning << "[DeepNetQAT] ==> variance < 1e-12 for all"
            " outputs! Is the network correctly trained?"
            << Utils::cdef << std::endl;
    }

    convCellTop->synchronizeToH(false);

    std::vector<Float_T> biasPerOutput 
        = std::vector<Float_T>(convCell->getNbOutputs(), 0.0f);
    std::vector<Float_T> clipPerOutput 
        = std::vector<Float_T>(convCell->getNbOutputs(), 0.0f);
    std::vector<Float_T> scalingPerOutput 
        = std::vector<Float_T>(convCell->getNbOutputs(), 0.0f);

    const size_t inputX = convCell->getChannelsWidth();
    const size_t inputY = convCell->getChannelsHeight();
    const size_t strideX = convCell->getStrideX();
    const size_t strideY = convCell->getStrideY();
    // double nbMultPerChannel = 0;
    std::pair<double, double> mScalingMinMax = std::make_pair(100.0, 0.0);
    std::pair<double, double> mBiasMinMax = std::make_pair(10000.0, -10000.0);
    std::pair<double, double> mClipMinMax = std::make_pair(100.0, 0.0);

    for (std::size_t output = 0; output < convCell->getNbOutputs(); ++output) {

        const double factor = bnVariances(output) >  1.0e-12 ?
            bnScales(output) / std::sqrt(eps +   bnVariances(output) ) : 
            //bnScales(output) / std::sqrt(eps);
            //when export, put 0 instead of bnScales(output) / std::sqrt(eps)
            0.0;

        double factorSign = factor < 0.0 ? -1.0 : 1.0;
        Tensor<double> bias;

        if (noBias)
            bias.resize({1}, 0.0);
        else
            convCell->getBias(output, bias);

        //Quantized Weights adjustmens
        for (std::size_t channel = 0; channel < convCell->getNbChannels(); ++channel) {
            Tensor<double> kernel;
            convCell->getQuantWeight(output, channel, kernel);
            // if(kernel.size() > 0) {
            //     nbMultPerChannel 
            //             = ((inputX / strideX) * kernel.dimX()) 
            //                 +  ((inputY / strideY) * kernel.dimY());
            // }
            for (std::size_t index = 0; index < kernel.size(); ++index) {
                kernel(index) = kernel(index)*(weightsScaleFactor);
                if(wMode == WeightsApprox::RINTF){
                    kernel(index) = rintf(kernel(index));
                }
                //Inverted kernel sign when scale factor is negatif
                kernel(index) *= factorSign;
            }
            convCell->setWeight(output, channel, kernel);
        }

        if(factor < 0.0){
            std::cout << Utils::cwarning << "[DeepNetQAT] ==> factor[" << output << "] = " << factor
            << " is zero ! Is the network correctly trained? Clipping, bias and scaling will be set to 0!"
                << Utils::cdef << std::endl;
        }
        if(factor == 0.0){
            std::cout << Utils::cwarning << "[DeepNetQAT] ==> factor[" << output << "] = " << factor
            << " is zero ! Is the network correctly trained? Clipping, bias and scaling will be set to 0!"
                << Utils::cdef << std::endl;
            scalingPerOutput[output] = 0.0;
            biasPerOutput[output] = 0.0;
            clipPerOutput[output] = 0.0;
        }
        else{
            scalingPerOutput[output] = (alphasConvBN.first / (float) rangeConvBN.first)
                                    * ((float) rangeConvBN.second / alphasConvBN.second) * factor
                                    * (1 / weightsScaleFactor) * corrFactor;

            biasPerOutput[output] = ((((bnBiases(output) + (bias(0) - bnMeans(output)) * factor) / factor)
                                    * rangeConvBN.first / alphasConvBN.first) * biasScaleFactor)* (1.0/corrFactor);

            clipPerOutput[output] = ((alphasConvBN.second * rangeConvBN.first)
                                    / (factor * alphasConvBN.first)) *weightsScaleFactor * (1.0/corrFactor);

            scalingPerOutput[output] *= factorSign;              
            biasPerOutput[output] *= factorSign;       
            clipPerOutput[output] *= factorSign;
        }

        if(cMode == WeightsApprox::RINTF)
            clipPerOutput[output] = rintf(clipPerOutput[output]);
        if(bMode == WeightsApprox::RINTF)
            biasPerOutput[output] = rintf(biasPerOutput[output]);
        if(scalingPerOutput[output] < mScalingMinMax.first) {
            mScalingMinMax.first = scalingPerOutput[output];
        }
        if(scalingPerOutput[output] > mScalingMinMax.second) {
            mScalingMinMax.second = scalingPerOutput[output];
        }

        if(biasPerOutput[output] < mBiasMinMax.first) {
            mBiasMinMax.first = biasPerOutput[output];
        }
        if(biasPerOutput[output] > mBiasMinMax.second) {
            mBiasMinMax.second = biasPerOutput[output];
        }
        if(clipPerOutput[output] < mClipMinMax.first) {
            mClipMinMax.first = clipPerOutput[output];
        }
        if(clipPerOutput[output] > mClipMinMax.second) {
            mClipMinMax.second = clipPerOutput[output];
        }


        /*
        std::cout << std::setprecision(7)  << "    bias[" << output << "]: " << biasPerOutput[output] 
            << "    clip[" << output << "]: " << clipPerOutput[output] << " with factor= " 
            << factor << " bnScales=" << bnScales(output) 
            << " bnVariances=" << bnVariances(output) 
            << " scalingPerOutput[" << output << "]= " << scalingPerOutput[output]
            << " meanVariance = " << meanVariance << std::endl;
        */
        

        bias(0) = biasPerOutput[output];
        convCell->setBias(output, bias);
    }
    std::cout << std::setprecision(7) << "[DeepNetQAT] ==> scaling min,max[" << mScalingMinMax.first << "," 
    << mScalingMinMax.second << "]" << std::endl; 
    std::cout << std::setprecision(7) << "[DeepNetQAT] ==> bias min,max[" << mBiasMinMax.first << "," 
    << mBiasMinMax.second << "]" << std::endl; 
    std::cout << std::setprecision(7) << "[DeepNetQAT] ==> clip min,max[" << mClipMinMax.first << "," 
    << mClipMinMax.second << "]" << std::endl; 

    mVarNulName.push_back(bnCell->getName());
    mVarNul.push_back(factor0Count);

    convCellTop->synchronizeToD(true);
    std::size_t weightsRange = 1;
    if(convCell->getQuantizer()) {
        weightsRange = convCell->getQuantizer()->getRange();
        //if 1b weights, set range to 2
        //to get correct nbBits for setQuantized
        if(weightsRange == 1){
            weightsRange = 2;
        }
    }
    //Delete QuantizerCell from convCell
    convCell->setQuantizer(std::shared_ptr<QuantizerCell>());

    //Set weights range
    convCell->setQuantized(rintf(log2(weightsRange)) );

    //Set activations range
    convCellActivation->setQuantized(rintf(log2(rangeConvBN.second)) );

    bool isClipped = true;
    if(actScalingMode == ScalingMode::FLOAT_MULT) {
        /*for(std::size_t out = 0; out < scalingPerOutput.size(); ++out) {
            std::cout << "    SCALING FACTOR["<< out << "]:"
                <<  scalingPerOutput[out] << std::endl;
        }*/
        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*bnCell))
                                (mDeepNet, 
                                    mDeepNet.generateNewCellName(bnCell->getName() + "_clip_then_rescale_act"), 
                                    bnCell->getNbOutputs(), 
                                    Scaling::floatingPointScaling(
                                        scalingPerOutput,
                                        isClipped,
                                        clipPerOutput));

        convCellActivation->setActivationScaling( std::move(scalingCell->getScaling()) );
    }
    else if(actScalingMode == ScalingMode::FIXED_MULT16 
            || actScalingMode == ScalingMode::FIXED_MULT32) {
        /**
         * Find the highest nbFractionalBits so that the scaling 
         * 'std::round(sc * (1ull << nbFractionalBits)' of each output
         * can be stored in an int32_t an thus in scalingFixedPoint.
         * 
         * TODO With unsigned activation like ReLU we could use the maximum
         * of an uint32_t to gain a bit more precision.
         */
        // const std::uint64_t limit = std::numeric_limits<std::int32_t>::max();
        // const std::size_t maxNbFractionalBits = 24;
        
        std::size_t nbFractionalBits = 16;
        std::vector<std::int32_t> scalingFixedPoint;

        for(std::size_t out = 0; out < scalingPerOutput.size(); ++out) {
            /*
            const double maxScaling = scalingPerOutput[out];
            assert(std::round(maxScaling * (1ull << nbFractionalBits)) < limit);
            while(std::round(maxScaling * (1ull << (nbFractionalBits + 1))) < limit && 
                nbFractionalBits + 1 <= maxNbFractionalBits) 
            {
                nbFractionalBits++;
            }
            */

            const Float_T sc = scalingPerOutput[out];
            scalingFixedPoint.push_back(std::round(sc * (1ull << nbFractionalBits)));
           // std::cout << "    SCALING FACTOR["<< out << "]:" <<  sc
            //<< " APPROX FIXED_MULT: " << scalingFixedPoint[out] << "2 ^ [- " << nbFractionalBits << "]" 
           // << std::endl;
        }
        
        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*bnCell))
                                (mDeepNet, 
                                    mDeepNet.generateNewCellName(bnCell->getName() + "_clip_then_rescale_act"), 
                                    bnCell->getNbOutputs(), 
                                    Scaling::fixedPointScaling(actScalingMode,
                                        nbFractionalBits,
                                        scalingFixedPoint,
                                        isClipped,
                                        clipPerOutput) );

        convCellActivation->setActivationScaling( std::move(scalingCell->getScaling()) );
        
    }
    else if(actScalingMode == ScalingMode::SINGLE_SHIFT) {
        std::vector<unsigned char> scalingSingleShiftPerOutput 
            = std::vector<unsigned char>(scalingPerOutput.size(), 0);

        for(std::size_t out = 0; out < scalingPerOutput.size(); ++out) {
            if(scalingPerOutput[out] > 0.0) {
                const auto singleDivApprox 
                        = approximateScalingWithPowerOf2Divs(scalingPerOutput[out], 1);
                /*std::cout << "    SCALING FACTOR["<< out << "]:" <<  scalingPerOutput[out] 
                << " APPROX SINGLE_SHIFT: 2 ^ [- " << (int)singleDivApprox.first[0] << "]" 
                << " PRECISION : " << singleDivApprox.second
                << std::endl;*/
                scalingSingleShiftPerOutput[out] = singleDivApprox.first[0];
            } else {
                std::cout << "[DeepNetQAT] ==> SCALING FACTOR["<< out << "]: 0.0"   
                << " NO APPROXIMATION POSSIBLE" << std::endl;
            }
        }

        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*bnCell))
                                (mDeepNet, 
                                    mDeepNet.generateNewCellName(bnCell->getName() + "_clip_then_rescale_act"), 
                                    bnCell->getNbOutputs(), 
                                    Scaling::singleShiftScaling(
                                        scalingSingleShiftPerOutput,
                                        isClipped,
                                        clipPerOutput) );

        convCellActivation->setActivationScaling( std::move(scalingCell->getScaling()) );
    }

    //std::shared_ptr<Activation> convCellActivation = convCellTop->getActivation();
    std::cout << Utils::cnotice 
            << "[DeepNetQAT] ==> " 
            << convCell->getName() << " is now quantized on " 
            << rintf(log2(rangeConvBN.second)) 
            << " bits"
            << Utils::cdef
            << std::endl;


    //Switch range and alpha
    rangeConvBN.first = rangeConvBN.second;
    alphasConvBN.first = alphasConvBN.second;
    return true;
}
//TODO: Improve output layers export for more compressed format and more genericity...
void N2D2::DeepNetQAT::exportOutputsLayers(StimuliProvider& sp,
                                            const std::string& dirName,
                                            Database::StimuliSet set,
                                            int nbStimuliMax) 
{
    std::vector<std::pair<std::string, double> >* timings = NULL;

    const std::size_t nbStimuli = nbStimuliMax >= 0?std::min(sp.getDatabase().getNbStimuli(set),
                                                            static_cast<unsigned int>(nbStimuliMax)):
                                                    sp.getDatabase().getNbStimuli(set);
    for(std::size_t iStimulus = 0; iStimulus < nbStimuli; iStimulus++) {
        const std::string stimuliDirName = dirName + "/" + "stimuli_" + std::to_string(iStimulus);
        Utils::createDirectories(stimuliDirName);
        std::cout << "[DeepNetQAT] ==> LOG Features maps of Stimuli " 
                    << iStimulus 
                    << std::endl; 
        sp.readStimulus(set, iStimulus);
        mDeepNet.test(set, timings);

        const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();

        for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
            for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
                const std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
                const std::shared_ptr<Cell_Frame_Top> cellFrame
                    = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
                const std::string cellMapDirName = stimuliDirName + "/" + cell->getName();
                Utils::createDirectories(cellMapDirName);

                cellFrame->getOutputs().synchronizeDToH();
                const Tensor<Float_T> outputs
                    = tensor_cast<Float_T>(cellFrame->getOutputs())[0];

                for(std::size_t map = 0; map < outputs.dimZ(); ++map) {
                    const std::string stimuliMapName = cellMapDirName 
                                                            + "/map_" +  std::to_string(map) 
                                                            + ".txt";
                    std::ofstream fileMap(stimuliMapName);

                    const Tensor<Float_T>& channel = outputs[map];

                    for(std::size_t y = 0; y < channel.dimY(); ++y) {
                        for(std::size_t x = 0; x < channel.dimX(); ++x) {
                            fileMap << static_cast<long long int>(std::round(channel(x, y))) 
                                    << ", ";
                        }
                        fileMap << "\n";
                    }
                    fileMap.close();
                    /*
                    const Tensor<unsigned int> channelU = tensor_cast<unsigned int>(channel);
                    Tensor<uint8_t> channelUint8 = tensor_cast<uint8_t>(channelU);
                    for(unsigned int i = 0; i < channelU.dimX() * channelU.dimY() ; ++i)
                        channelUint8(i) = uint8_t (channelU(i) * 17U);
                    //std::cout << channelU << std::endl;
                    cv::Mat mapAsCV = cv::Mat(channelUint8.dimY(), channelUint8.dimX(), CV_8U, channelUint8(0));
                    cv::imwrite(stimuliMapName, mapAsCV);
                    */
                    //const unsigned int 
                    //Utils::createDirectories(stimuliMapDirName);
                    //cv::Mat mapAsCV;

                }

            }
        }

    }

                            
}
