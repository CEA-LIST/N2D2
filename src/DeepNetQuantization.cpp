/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <vector>

#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "Histogram.hpp"
#include "RangeStats.hpp"
#include "StimuliProvider.hpp"
#include "Activation/ActivationScalingMode.hpp"
#include "Activation/LinearActivation.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/PoolCell.hpp"
#include "Export/DeepNetExport.hpp"


N2D2::DeepNetQuantization::DeepNetQuantization(DeepNet& deepNet): mDeepNet(deepNet) {
}

void N2D2::DeepNetQuantization::clipWeights(std::size_t nbBits, ClippingMode wtClippingMode) {
    if(wtClippingMode == ClippingMode::NONE) {
        return;
    }

    const std::size_t nbBins = getNbBinsForClippingMode(nbBits, wtClippingMode);

    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
        for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
            const auto& cell = mDeepNet.getCell(*itCell);

            const auto range = cell->getFreeParametersRange(false);
            const Float_T maxWeight = Utils::max_abs(range.first, range.second);

            if(maxWeight == 0) {
                continue;
            }

            Histogram hist(-maxWeight, maxWeight, nbBins);
            cell->processFreeParameters([&](double wt) { 
                hist(wt);
                return wt; 
            }, Cell::Multiplicative);

            double threshold;
            switch(wtClippingMode) {
                case ClippingMode::KL_DIVERGENCE:
                    threshold = hist.calibrateKLDivergence(nbBits);
                    break;
                case ClippingMode::MSE:
                    threshold = hist.calibrateMSE(nbBits);
                    break;
                default:
                    throw std::runtime_error("Unsupported clipping mode.");
            }

            cell->processFreeParameters([&](double wt) { 
                return Utils::clamp(wt, -threshold, threshold); 
            }, Cell::Multiplicative);
        }
    }
}

void N2D2::DeepNetQuantization::normalizeFreeParameters(double normFactor) {
    double bNorm = 1.0;

    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        if(itLayer->size() != 1) {
            throw std::runtime_error("Normalization of multi-branch networks are not supported yet.");
        }

        std::shared_ptr<Cell> cell = mDeepNet.getCell(itLayer->front());
        assert(cell != nullptr);

        const auto wMinMax = cell->getFreeParametersRange(false);
        const Float_T norm = Utils::max_abs(wMinMax.first, wMinMax.second)/normFactor;


        if (bNorm != 1.0) {
            assert(bNorm > 0.0);
            cell->processFreeParameters([&](double d) { return d/bNorm; }, Cell::Additive);
        }

        if(norm != 0.0) {
            cell->processFreeParameters([&](double d) { return d/norm; });
            bNorm *= norm;
        }
    }
}

void N2D2::DeepNetQuantization::normalizeFreeParametersPerOutputChannel(double normFactor) {
    double bNorm = 1.0;

    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        if(itLayer->size() != 1) {
            throw std::runtime_error("Normalization of multi-branch networks are not supported yet.");
        }

        std::shared_ptr<Cell> cell = mDeepNet.getCell(itLayer->front());
        assert(cell != nullptr);
        
        std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
        std::shared_ptr<Activation> activation = cellFrame->getActivation();

        if(!activation) {
            continue;
        }


        if (bNorm != 1.0) {
            assert(bNorm > 0.0);
            cell->processFreeParameters([&](double d) { return d/bNorm; }, Cell::Additive);
        }

        const auto wMinMax = cell->getFreeParametersRange(false);
        const Float_T maxNorm = Utils::max_abs(wMinMax.first, wMinMax.second);

        if(maxNorm == 0) {
            continue;
        }

        std::vector<double> actScalingPerOutput(cell->getNbOutputs());
        for(std::size_t output = 0; output < cell->getNbOutputs(); output++) {
            const auto woMinMax = cell->getFreeParametersRangePerOutput(output, false);
            const Float_T norm = std::max(std::min(maxNorm, 0.1f), 
                                          Utils::max_abs(woMinMax.first, woMinMax.second))/normFactor;

            cell->processFreeParametersPerOutput([&](double d) { return d/norm; }, output);
            actScalingPerOutput[output] = norm/maxNorm;
        }


        activation->setActivationScaling(ActivationScaling::floatingPointScaling(
                                            std::move(actScalingPerOutput
                                        )));
        bNorm *= maxNorm;
    }
}

void N2D2::DeepNetQuantization::rescaleAdditiveParameters(double rescaleFactor) {
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();

    for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
        for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            cell->processFreeParameters([&](double v) { return v/rescaleFactor; }, 
                                        Cell::Additive);
        }
    }
}

void N2D2::DeepNetQuantization::reportOutputsRange(std::unordered_map<std::string, RangeStats>& outputsRange) const {
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    std::map<std::string, std::shared_ptr<Cell>>& cells = mDeepNet.getCells();

    if (outputsRange.empty()) {
        // Populate outputsRange first to avoid thread issues
        for (auto itLayer = layers.begin(); itLayer != layers.end(); ++itLayer) {
            for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
                outputsRange.insert(std::make_pair(*itCell, RangeStats()));
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)layers.size(); ++i) {
        for(auto itCell = layers[i].begin(); itCell != layers[i].end(); ++itCell) {
            std::shared_ptr<Cell_Frame_Top> cellFrame;

            if (cells.find(*itCell) != cells.end()) {
                cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cells.at(*itCell));
                cellFrame->getOutputs().synchronizeDToH();
            }

            const Tensor<Float_T>& outputs = (cellFrame)
                ? tensor_cast<Float_T>(cellFrame->getOutputs())
                : mDeepNet.getStimuliProvider()->getData();

            RangeStats& rangeStats = outputsRange.at(*itCell);
            assert(outputs.size() == outputs.dimB()*outputs.dimZ()*outputs.dimY()*outputs.dimX());
            
            for(std::size_t batch = 0; batch < outputs.dimB(); batch++) {
                if(mDeepNet.getStimuliProvider()->getBatch().at(batch) == -1) {
                    continue;
                }
                
                for(Float_T val: outputs[batch]) {
                    rangeStats(val);
                }
            }
        }
    }
}

void N2D2::DeepNetQuantization::reportOutputsHistogram(
                        std::unordered_map<std::string, Histogram>& outputsHistogram,
                        const std::unordered_map<std::string, RangeStats>& outputsRange,
                        std::size_t nbBits, ClippingMode actClippingMode) const
{
    if(actClippingMode == ClippingMode::NONE) {
        return;
    }

    const std::size_t nbBins = getNbBinsForClippingMode(nbBits, actClippingMode);
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    std::map<std::string, std::shared_ptr<Cell>>& cells = mDeepNet.getCells();

    if (outputsHistogram.empty()) {
        // Populate outputsHistogram first to avoid thread issues
        for (auto itLayer = layers.begin(); itLayer != layers.end(); ++itLayer) {
            for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
                const auto range = outputsRange.at(*itCell);
                const bool isCellOutputUnsigned = (itLayer == layers.begin())?
                                            DeepNetExport::mEnvDataUnsigned:
                                            DeepNetExport::isCellOutputUnsigned(*cells.at(*itCell));

                const Float_T val = Utils::max_abs(range.minVal(), range.maxVal());
                const Float_T min = isCellOutputUnsigned?0:-val;
                const Float_T max = val;
                outputsHistogram.insert(std::make_pair(*itCell, Histogram(min, max, nbBins)));
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)layers.size(); ++i) {
        for(auto itCell = layers[i].begin(); itCell != layers[i].end(); ++itCell) {
            std::shared_ptr<Cell_Frame_Top> cellFrame;

            if (cells.find(*itCell) != cells.end()) {
                cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cells.at(*itCell));
                cellFrame->getOutputs().synchronizeDToH();
            }

            const Tensor<Float_T>& outputs = (cellFrame)
                ? tensor_cast<Float_T>(cellFrame->getOutputs())
                : mDeepNet.getStimuliProvider()->getData();


            Histogram& hist = outputsHistogram.at(*itCell);
            assert(outputs.size() == outputs.dimB()*outputs.dimZ()*outputs.dimY()*outputs.dimX());

            for(std::size_t batch = 0; batch < outputs.dimB(); batch++) {
                if(mDeepNet.getStimuliProvider()->getBatch().at(batch) == -1) {
                    continue;
                }

                for(Float_T val: outputs[batch]) {
                    hist(val);
                }
            }
        }
    }
}

void N2D2::DeepNetQuantization::normalizeOutputsRange(const std::unordered_map<std::string, Histogram>& outputsHistogram,
                                          const std::unordered_map<std::string, RangeStats>& outputsRange,
                                          std::size_t nbBits,
                                          ClippingMode actClippingMode)
{
    double prevScalingFactor = 1.0;
    bool nextIsMaxPool = false;

    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        if(itLayer->size() != 1) {
            throw std::runtime_error("Normalization of multi-branch networks are not supported yet.");
        }

        std::shared_ptr<Cell> cell = mDeepNet.getCell(itLayer->front());
        std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

        if(!cellFrame || (cell->getType() == PoolCell::Type)) {
            nextIsMaxPool = false;
            continue;
        }

        if (itLayer + 1 != layers.end() && (itLayer + 1)->size() == 1) {
            const auto& nextCell = mDeepNet.getCell((itLayer + 1)->front());
            nextIsMaxPool = nextCell->getType() == PoolCell::Type && 
                            dynamic_cast<const PoolCell&>(*nextCell).getPooling() == PoolCell::Max;
        }



        const std::shared_ptr<Activation>& activation = cellFrame->getActivation();
        if(!activation) {
            throw std::runtime_error("Missing activation function for cell " + cell->getName() + ".");
        }
        

        double scalingFactor;
        if(activation->getType() == RectifierActivation::Type || 
           (activation->getType() == LinearActivation::Type && cell->getNbOutputs() > 2) || 
           (activation->getType() == SaturationActivation::Type && cell->getNbOutputs() > 2))
        {
            const std::string cellStats = nextIsMaxPool?(itLayer + 1)->front():itLayer->front();
            scalingFactor = getCellThreshold(cellStats, 
                                             outputsHistogram, outputsRange, 
                                             nbBits, actClippingMode);
        }
        else {
            scalingFactor = getCellThreshold(itLayer->front(),
                                             outputsHistogram, outputsRange, 
                                             nbBits, ClippingMode::NONE);
        }



        
        rescaleActivationOutputs(*cell, *activation, 
                                 scalingFactor, prevScalingFactor);

        cell->processFreeParameters([&](double d) { return d/prevScalingFactor; },
                                    Cell::Additive);
        

        std::cout << std::setprecision(4) 
                  << cell->getName() << ": " << "scalingFactor = " << scalingFactor << "   " 
                  << std::endl;

        prevScalingFactor = scalingFactor;
    }
}

void N2D2::DeepNetQuantization::quantizeNormalizedNetwork(std::size_t nbBits, 
                                                          ActivationScalingMode actScalingMode) 
{
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        if(itLayer->size() != 1) {
            throw std::runtime_error("Normalization of multi-branch networks are not supported yet.");
        }

        std::shared_ptr<Cell> cell = mDeepNet.getCell(itLayer->front());
        std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

        if(!cellFrame) {
            continue;
        }

        const std::shared_ptr<Activation>& activation = cellFrame->getActivation();
        if(!activation) {
            continue;
        }

        quantizeActivationScaling(*cell, *activation, nbBits, actScalingMode);

        // Must come after quantizeActivationScaling as it may modify the weights
        // if the actScalingMode is SINGLE_SHIFT or DOUBLE_SHIFT
        quantizeFreeParemeters(*cell, nbBits);
    }
}

std::pair<std::vector<unsigned char>, double> N2D2::DeepNetQuantization::approximateRescalingWithPowerOf2Divs(
                                                        double scaling, std::size_t nbDivisions) 
{
    static const double ROUNDING_THRESHOLD = 0.98;

    assert(nbDivisions > 0);
    assert(scaling <= 1.0);


    double precision = 0.0;

    std::vector<unsigned char> powerOf2Divs(nbDivisions);
    for(std::size_t iDiv = 0; iDiv < nbDivisions; iDiv++) {
        if(precision == 1.0) {
            powerOf2Divs[iDiv-1]++;
            powerOf2Divs[iDiv] = powerOf2Divs[iDiv-1];
        }
        else {
            const std::size_t exponent = std::ceil(std::log2(1.0/(scaling*(1.0 - precision))));
            precision += 1.0/(scaling*std::pow(2, exponent));

            powerOf2Divs[iDiv] = static_cast<unsigned char>(exponent);
        }
    }

    assert(precision <= 1.0);

    if(precision >= ROUNDING_THRESHOLD) {
        precision = 1.0;
    }
    else if(precision < 1.0) {
        precision += 1.0/(scaling*std::pow(2, powerOf2Divs.back()));
        powerOf2Divs.back() = powerOf2Divs.back() - 1;
    }

    assert(precision >= 1.0);

    return std::make_pair(powerOf2Divs, precision);
}

std::vector<std::vector<unsigned char>> N2D2::DeepNetQuantization::approximateRescalingsWithPowerOf2Divs(Cell& cell, 
                                                        const std::vector<double>& scalingPerOutput, 
                                                        std::size_t nbDivisions)
{
    std::vector<std::vector<unsigned char>> exponentsPerOutput(cell.getNbOutputs());
    for(std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        double rescaleOutputsBy;
        if(nbDivisions == 1) {
            const auto singleDivApprox = approximateRescalingWithPowerOf2Divs(scalingPerOutput[output], 1);

            exponentsPerOutput[output] = std::move(singleDivApprox.first);
            rescaleOutputsBy = 1/singleDivApprox.second;
        }
        else if(nbDivisions == 2) {
            const auto doubleDivApprox = approximateRescalingWithPowerOf2Divs(scalingPerOutput[output], 2);

            exponentsPerOutput[output] = std::move(doubleDivApprox.first);
            rescaleOutputsBy = 1/doubleDivApprox.second;
        }
        else {
            throw std::runtime_error("Currently only an approximation with 1 or 2 divisions is supported.");
        }

        // Rescale the weights and biasses of the cell to compensate the lost precision
        // of the approximation.
        cell.processFreeParametersPerOutput([&](double d){ 
                                                return rescaleOutputsBy*d; 
                                            }, output);
    }

    return exponentsPerOutput;
}

double N2D2::DeepNetQuantization::getCellThreshold(const std::string& cellName,
                                       const std::unordered_map<std::string, Histogram>& outputsHistogram,
                                       const std::unordered_map<std::string, RangeStats>& outputsRange,
                                       std::size_t nbBits, ClippingMode actClippingMode) 
{
    switch(actClippingMode) {
        case ClippingMode::KL_DIVERGENCE:
            return outputsHistogram.at(cellName).calibrateKLDivergence(nbBits);
        case ClippingMode::MSE:
            return outputsHistogram.at(cellName).calibrateMSE(nbBits);
        default: {
            const auto& range = outputsRange.at(cellName);
            return Utils::max_abs(range.minVal(), range.maxVal());
        }
    }
}

void N2D2::DeepNetQuantization::approximateRescalings(Cell& cell, Activation& activation,
                                          ActivationScalingMode actScalingMode) 
{
    assert(activation.getActivationScaling().getMode() == ActivationScalingMode::FLOAT_MULT);

    const std::vector<double>& scalingPerOutput = activation.getActivationScaling()
                                                            .getFloatingPointScaling()
                                                            .getScalingPerOutput();
    if(actScalingMode == ActivationScalingMode::FLOAT_MULT) {
        // Nothing to do.
    }
    else if(actScalingMode == ActivationScalingMode::FIXED_MULT) {
        const double maxScaling = *std::max_element(scalingPerOutput.begin(), scalingPerOutput.end());
        assert(maxScaling <= 1.0);
        
        /**
         * Find the highest nbFractionalBits so that the scaling 
         * 'std::round(sc * (1ull << nbFractionalBits)' of each output
         * can be stored in an int32_t an thus in scalingFixedPoint.
         * 
         * TODO With unsigned activation like ReLU we could use the maximum
         * of an uint32_t to gain a bit more precision.
         */
        const std::size_t limit = std::numeric_limits<std::int32_t>::max();
         // We need to keep enough bits for the non-fractional part (64-maxNbFractionalBits).
        const std::size_t maxNbFractionalBits = 36;

        std::size_t nbFractionalBits = 30;
        assert(std::round(maxScaling * (1ull << nbFractionalBits)) < limit);
        while(std::round(maxScaling * (1ull << (nbFractionalBits + 1))) < limit && 
              nbFractionalBits + 1 <= maxNbFractionalBits) 
        {
            nbFractionalBits++;
        }
        
        

        std::vector<std::int32_t> scalingFixedPoint;
        for(auto sc: scalingPerOutput) {
            scalingFixedPoint.push_back(std::round(sc * (1ull << nbFractionalBits)));
        }

        activation.setActivationScaling(ActivationScaling::fixedPointScaling(nbFractionalBits, 
                                                                             scalingFixedPoint));
    }
    else if(actScalingMode == ActivationScalingMode::SINGLE_SHIFT) {
        std::vector<unsigned char> shifts;
        for(const auto& powOf2Exponents: approximateRescalingsWithPowerOf2Divs(cell, scalingPerOutput, 1)) {
            assert(powOf2Exponents.size() == 1);
            shifts.push_back(powOf2Exponents[0]);
        }

        activation.setActivationScaling(ActivationScaling::singleShiftScaling(shifts));
    }
    else if(actScalingMode == ActivationScalingMode::DOUBLE_SHIFT) {
        std::vector<std::pair<unsigned char, unsigned char>> shifts;
        for(const auto& powOf2Exponents: approximateRescalingsWithPowerOf2Divs(cell, scalingPerOutput, 2)) {
            assert(powOf2Exponents.size() == 2);
            assert(powOf2Exponents[0] <= powOf2Exponents[1]);
            shifts.push_back({powOf2Exponents[1] - powOf2Exponents[0], powOf2Exponents[1]});
        }

        activation.setActivationScaling(ActivationScaling::doubleShiftScaling(shifts));
    }
    else {
        throw std::runtime_error("Unsupported scaling mode.");
    }
}

void N2D2::DeepNetQuantization::rescaleActivationOutputs(const Cell& cell, Activation& activation,
                                                         double scalingFactor, double prevScalingFactor)
{
    const ActivationScalingMode scalingMode = activation.getActivationScaling().getMode();
    
    std::vector<double> scalingPerOutput(cell.getNbOutputs());
    for(std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        if(scalingMode == ActivationScalingMode::NONE) {
            scalingPerOutput[output] = 1 / (scalingFactor / prevScalingFactor);
        }
        else if(scalingMode == ActivationScalingMode::FLOAT_MULT) {
            const double actScaling = activation.getActivationScaling()
                                                .getFloatingPointScaling()
                                                .getScalingPerOutput()[output];
            scalingPerOutput[output] = actScaling / (scalingFactor / prevScalingFactor);
        }
        else {
            throw std::runtime_error("Unsupported scaling mode.");
        }
    }

    activation.setActivationScaling(ActivationScaling::floatingPointScaling(std::move(scalingPerOutput)));
}

void  N2D2::DeepNetQuantization::quantizeActivationScaling(Cell& cell, Activation& activation, 
                                                           std::size_t nbBits, 
                                                           ActivationScalingMode actScalingMode) 
{
    const ActivationScalingMode scalingMode = activation.getActivationScaling().getMode();
    if(scalingMode != ActivationScalingMode::FLOAT_MULT) {
        assert(scalingMode == ActivationScalingMode::NONE);
        return;
    }

    const double unsignedMax = std::pow(2, nbBits) - 1;
    const double signedMax = std::pow(2, nbBits - 1) - 1;
    
    std::vector<double> scalingPerOutput = activation.getActivationScaling()
                                                     .getFloatingPointScaling()
                                                     .getScalingPerOutput();
    for(double& scaling: scalingPerOutput) {
        const std::string activationType = activation.getType();
        if(activationType == RectifierActivation::Type) {
            scaling /= DeepNetExport::isCellInputsUnsigned(cell)?
                            unsignedMax*signedMax/unsignedMax:
                            signedMax*signedMax/unsignedMax;
        }
        else if(activationType == LogisticActivation::Type || 
                activationType == LogisticActivation::TypeWithLoss) 
        {
            scaling /= 2*(DeepNetExport::isCellInputsUnsigned(cell)?
                            unsignedMax*signedMax/signedMax:
                            signedMax*signedMax/signedMax);
        }
        else {
            scaling /= DeepNetExport::isCellInputsUnsigned(cell)?
                            unsignedMax*signedMax/signedMax:
                            signedMax*signedMax/signedMax;
        }
    }

    // Ensure that every scalingPerOutput[o] <= 1.0 so that we only have to manage 
    // downscalling when approximating the rescaling. A scalingPerOutput[o] > 1.0 should
    // be really rare
    // TODO Find a network where it happens and test how well it works
    const double maxScaling = *std::max_element(scalingPerOutput.begin(), scalingPerOutput.end());
    if(actScalingMode != ActivationScalingMode::FLOAT_MULT && maxScaling > 1.0) {
        for(double& scaling: scalingPerOutput) {
            scaling /= maxScaling;
        }
    }

    activation.setActivationScaling(
        ActivationScaling::floatingPointScaling(std::move(scalingPerOutput))
    );

    approximateRescalings(cell, activation, actScalingMode);

}

void N2D2::DeepNetQuantization::quantizeFreeParemeters(Cell& cell, std::size_t nbBits) {
    cell.processFreeParameters([&](double wt) { 
        const double scaling = (double) std::pow(2, nbBits - 1) - 1;
        return std::round(wt*scaling);
    }, Cell::Multiplicative);

    cell.processFreeParameters([&](double bias) { 
        // For the bias we also need to scale it by the maximum value of the input type.
        // A bias is just like an extra connection where the input is equal to 1.0.
        
        double scaling = (double) std::pow(2, nbBits - 1) - 1;
        if(DeepNetExport::isCellInputsUnsigned(cell)) {
            scaling *= std::pow(2, nbBits) - 1;
        }
        else {
            scaling *= std::pow(2, nbBits - 1) - 1;
        }
        
        return std::round(bias*scaling);
    }, Cell::Additive);
}
