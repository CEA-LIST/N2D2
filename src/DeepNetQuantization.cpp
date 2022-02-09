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

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepNet.hpp"
#include "DeepNetQuantization.hpp"
#include "Histogram.hpp"
#include "RangeStats.hpp"
#include "ScalingMode.hpp"
#include "StimuliProvider.hpp"
#include "ScalingMode.hpp"
#include "Activation/LinearActivation.hpp"
#include "Activation/LinearActivation_Frame.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/RectifierActivation_Frame.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/ResizeCell.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/SoftmaxCell.hpp"
#include "Cell/TransposeCell.hpp"
#include "Export/DeepNetExport.hpp"
#include "Transformation/RangeAffineTransformation.hpp"

#define VERBOSE_QUANT

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

            const auto range
                = cell->getFreeParametersRange(Cell::Multiplicative);
            const Float_T maxWeight = Utils::max_abs(range.first, range.second);

            if(maxWeight == 0) {
                continue;
            }

            Histogram hist(-maxWeight, maxWeight, nbBins);
            cell->processFreeParameters([&](Float_T wt) { 
                hist(wt);
                return wt; 
            }, Cell::Multiplicative);

            Float_T threshold;
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

            cell->processFreeParameters([&](Float_T wt) { 
                return Utils::clamp(wt, -threshold, threshold); 
            }, Cell::Multiplicative);
        }
    }
}

long double N2D2::DeepNetQuantization::getMaxParentsScaling(const std::shared_ptr<Cell>& cell, 
                                const std::unordered_map<std::string, long double>& scalingForCells) const 
{
    long double maxParentsScaling = 0.0;
    for(const auto& parentCell: cell->getParentsCells()) {
        const long double parentScaling = parentCell?scalingForCells.at(parentCell->getName()):1.0;
        maxParentsScaling = std::max(maxParentsScaling, parentScaling);

        assert(parentScaling > 0.0);
    }

    return maxParentsScaling;
}

void N2D2::DeepNetQuantization::rescaleParentsToScaling(const std::shared_ptr<Cell>& cell, 
                                        const std::unordered_map<std::string, long double>& scalingForCells,
                                        long double scaling)
{
    // Get a copy, the loop modify the graph
    const std::vector<std::shared_ptr<Cell>> parentsCells = cell->getParentsCells();

    for(const std::shared_ptr<Cell>& parentCell: parentsCells) {
        const long double parentScaling = parentCell?scalingForCells.at(parentCell->getName()):1.0;
        if(parentScaling == scaling) {
            continue;
        }
        
        assert(parentScaling < scaling);
        auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*parentCell))
                                (mDeepNet, 
                                 mDeepNet.generateNewCellName(parentCell->getName() + "_rescale_branch"), 
                                 parentCell->getNbOutputs(), 
                                 Scaling::floatingPointScaling(
                                     std::vector<Float_T>(parentCell->getNbOutputs(), 
                                                          parentScaling/scaling), false, std::vector<Float_T>(0.0f))
                                );

        mDeepNet.addCellBetween(scalingCell, parentCell, cell);
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
#ifdef CUDA
        CudaContext::setDevice(mDeepNet.getMasterDevice());
#endif

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
                
                double val = Utils::max_abs(range.minVal(), range.maxVal());
                // Take 0.1 as minimum value as we don't want a range of [0;0]
                val = std::max(val, 0.1);

                const double min = isCellOutputUnsigned?0:-val;
                const double max = val;
                outputsHistogram.insert(std::make_pair(*itCell, Histogram(min, max, nbBins)));
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)layers.size(); ++i) {
#ifdef CUDA
        CudaContext::setDevice();
#endif

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

            const auto range = outputsRange.at(*itCell);
            const bool enlargeSymetric = hist.getMinVal() < 0.0;
            hist.enlarge(Utils::max_abs(range.minVal(), range.maxVal()), enlargeSymetric);

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

void N2D2::DeepNetQuantization::rescaleAdditiveParameters(double rescaleFactor) {
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();

    for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
        for (auto itCell = it->begin(); itCell != it->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            cell->processFreeParameters([&](Float_T v) { return v/rescaleFactor; },
                                        Cell::Additive);
        }
    }
}

void N2D2::DeepNetQuantization::quantizeNetwork(const std::unordered_map<std::string, Histogram>& outputsHistogram,
                                                const std::unordered_map<std::string, RangeStats>& outputsRange,
                                                std::size_t nbBits,
                                                ClippingMode actClippingMode,
                                                ScalingMode actScalingMode,
                                                bool rescalePerOutputChannel,
                                                double quantileValue)
{
    const std::vector<std::vector<std::string>>& layers = mDeepNet.getLayers();

    // Synchronize to H and no keep in sync
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
            if(!cell || !cellFrame) {
                throw std::runtime_error("Invalid cell.");
            }

            cellFrame->synchronizeToH(false);
        }
    }

    std::unordered_map<std::string, long double> biasScalings;
    if(rescalePerOutputChannel) {
        biasScalings = quantizeFreeParemetersPerOutputCh(nbBits);
    }
    else {
        biasScalings = quantizeFreeParemeters(nbBits);
    }

    quantizeActivations(outputsHistogram, outputsRange, biasScalings,
                        nbBits, actClippingMode, quantileValue);

#ifdef VERBOSE_QUANT
    std::cout << "  Scaling approximation [" << (int)actScalingMode << "]:"
        << std::endl;
#endif

    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
            if(!cell || !cellFrame) {
                throw std::runtime_error("Invalid cell.");
            }

            if(cell->getType() == ScalingCell::Type) {
                const ScalingMode scalingCellMode
                    = (actScalingMode == ScalingMode::SINGLE_SHIFT
                        || actScalingMode == ScalingMode::DOUBLE_SHIFT)
                            ? ScalingMode::FIXED_MULT16 : actScalingMode;
                approximateScalingCell(dynamic_cast<ScalingCell&>(*cell), scalingCellMode, nbBits);
            }

            const std::shared_ptr<Activation>& activation = cellFrame->getActivation();
            if(activation) {
                approximateActivationScaling(*cell, *activation, actScalingMode);
            }

            // Must come after approximateActivationScaling as the approximation may modify the weights
            // if the actScalingMode is SINGLE_SHIFT or DOUBLE_SHIFT
            cell->processFreeParameters([&](Float_T p) { return std::round(p); });

            cell->setQuantized(nbBits);
        }
    }

    // Quantize inputs
#ifdef VERBOSE_QUANT
    std::cout << "  Inputs quantization" << std::endl;
#endif

    assert(mDeepNet.getStimuliProvider() != nullptr);
    RangeAffineTransformation affineTrans(RangeAffineTransformation::Multiplies, 
                DeepNetExport::mEnvDataUnsigned?std::pow(2, nbBits) - 1:
                                                std::pow(2, nbBits - 1) - 1);
    // Ensure that the inputs are representable as integers.
    // This is the case for unsigned 8-bit images scaled in the [0,1] range.
    // But for signed inputs, it depends on the scaling method.
    affineTrans.setParameter<bool>("Truncate", true);

    mDeepNet.getStimuliProvider()->addTopTransformation(
        affineTrans,
        Database::All
    );

    // Synchronize to D and keep in sync
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
            if(!cell || !cellFrame) {
                throw std::runtime_error("Invalid cell.");
            }

            cellFrame->synchronizeToD(true);
        }
    }

#ifdef VERBOSE_QUANT
    std::cout << "  Done!" << std::endl;
#endif
}

// See https://arxiv.org/pdf/1906.04721.pdf
void N2D2::DeepNetQuantization::crossLayerEqualization(
    double maxQuantRangeDelta,
    bool removeReLUClipping)
{
#ifdef VERBOSE_QUANT
    std::cout << "  Cross-layer equalization:" << std::endl;
#endif

    const std::vector<std::vector<std::string>> layers = mDeepNet.getLayers();
    double maxRangeDelta;

    std::set<std::shared_ptr<Cell_Frame_Top> > keepInSync;

    do {
        maxRangeDelta = 0.0;

        for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
            for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
                std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
                if(!cell) {
                    throw std::runtime_error("Invalid cell.");
                }

                auto parentsCells = cell->getParentsCells();

                if ((cell->getType() == ConvCell::Type
                    || cell->getType() == FcCell::Type)
                    && parentsCells.size() == 1
                    && parentsCells[0]
                    && parentsCells[0]->getChildrenCells().size() == 1
                    && (parentsCells[0]->getType() == ConvCell::Type
                    || parentsCells[0]->getType() == FcCell::Type))
                {
                    std::shared_ptr<Cell_Frame_Top> cellFrame
                        = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
                    std::shared_ptr<Cell_Frame_Top> parentCellFrame
                        = std::dynamic_pointer_cast<Cell_Frame_Top>(parentsCells[0]);
                    const std::shared_ptr<Activation>& parentActivation
                        = parentCellFrame->getActivation();

                    if (parentActivation
                        && parentActivation->getType() != LinearActivation::Type
                        && parentActivation->getType() != RectifierActivation::Type)
                    {
                        // Activation function must be piece-wise linear.
                        // Otherwise, skip this pair of layer
                        continue;
                    }

                    if (parentActivation
                        && parentActivation->getType() == RectifierActivation::Type)
                    {
                        // Remove clipping
                        const double clipping
                            = parentActivation->getParameter<double>("Clipping");

                        if (clipping > 0.0) {
                            if (removeReLUClipping) {
                                parentActivation
                                    ->setParameter<double>("Clipping", 0.0);
                            }
                            else {
                                continue;
                            }
                        }
                    }

#ifdef VERBOSE_QUANT
                    std::cout << "    - eq. " << cell->getName()
                        << " and " << parentsCells[0]->getName() << std::endl;
#endif

                    bool newInsert;
                    std::tie(std::ignore, newInsert)
                        = keepInSync.insert(cellFrame);

                    if (newInsert)
                        cellFrame->synchronizeToH(false);

                    std::tie(std::ignore, newInsert)
                        = keepInSync.insert(parentCellFrame);

                    if (newInsert)
                        parentCellFrame->synchronizeToH(false);

                    for(std::size_t output = 0;
                        output < parentsCells[0]->getNbOutputs(); output++)
                    {
                        const auto bias1MinMax = parentsCells[0]
                            ->getFreeParametersRangePerOutput(output,
                                                        Cell::Additive);
                        const auto r1MinMax = parentsCells[0]
                            ->getFreeParametersRangePerOutput(output,
                                                        Cell::Multiplicative);
                        const auto r2MinMax = cell
                            ->getFreeParametersRangePerChannel(output);

                        const Float_T bias1 = Utils::max_abs(bias1MinMax.first,
                                                          bias1MinMax.second);
                        const Float_T r1 = Utils::max_abs(r1MinMax.first,
                                                          r1MinMax.second);
                        const Float_T r2 = Utils::max_abs(r2MinMax.first,
                                                          r2MinMax.second);

                        // If r1 is 0.0, meaning the parent cell's output 
                        // weights range is 0, the bias can still contribute
                        // to the cell channel, in which case no rescaling is
                        // done. If the bias is 0, then the cell's channel 
                        // weights can be set to 0.
                        if (r1 > 0.0 || bias1 == 0.0) {
                            const double scalingPerOutput1 = (r1 > 0.0)
                                ? (1.0 / r1) * std::sqrt(r1 * r2) : 0.0;
                            const double scalingPerOutput2 = (r2 > 0.0)
                                ? (1.0 / r2) * std::sqrt(r1 * r2) : 0.0;

                            parentsCells[0]->processFreeParametersPerOutput(
                                [&](Float_T w) { 
                                    return w*scalingPerOutput1; 
                                }, output, Cell::All);
                            cell->processFreeParametersPerChannel(
                                [&](Float_T w) { 
                                    return w*scalingPerOutput2; 
                                }, output);
                        }

                        const double rangeDelta = std::abs(r1 - r2);

                        if (rangeDelta > maxRangeDelta)
                            maxRangeDelta = rangeDelta;
                    }
                }
            }
        }

#ifdef VERBOSE_QUANT
        std::cout << "    quant. range delta = " << maxRangeDelta
            << std::endl;
#endif
    }
    while (maxRangeDelta > maxQuantRangeDelta);

    for (std::set<std::shared_ptr<Cell_Frame_Top> >::const_iterator it
        = keepInSync.begin(), itEnd = keepInSync.end(); it != itEnd; ++it)
    {
        (*it)->synchronizeToD(true);
    }
}

std::unordered_map<std::string, long double> N2D2::DeepNetQuantization::quantizeFreeParemeters(std::size_t nbBits) {
#ifdef VERBOSE_QUANT
    std::cout << "  Quantizing free parameters:" << std::endl;
#endif

    std::unordered_map<std::string, long double> biasScalings;

    std::vector<std::vector<std::string>> layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            long double biasScaling = getMaxParentsScaling(cell, biasScalings);
            rescaleParentsToScaling(cell, biasScalings, biasScaling);


            const long double wQuantScaling = std::pow(2, nbBits - 1) - 1;
            const long double bQuantScaling = DeepNetExport::isCellInputsUnsigned(*cell)?
                                                  wQuantScaling*(std::pow(2, nbBits) - 1):
                                                  wQuantScaling*(std::pow(2, nbBits - 1) - 1);


            const std::pair<Float_T, Float_T> wMinMax
                = cell->getFreeParametersRange(Cell::Multiplicative);
            const Float_T wScalingCell = Utils::max_abs(wMinMax.first, wMinMax.second);
            if(wScalingCell != 0.0) {
                cell->processFreeParameters([&](Float_T w) { return w*(wQuantScaling/wScalingCell); }, 
                                            Cell::Multiplicative);

                biasScaling *= wScalingCell;
            }

            cell->processFreeParameters([&](Float_T b) { return b*(bQuantScaling/biasScaling); }, 
                                        Cell::Additive);
            biasScalings[cell->getName()] = biasScaling;

#ifdef VERBOSE_QUANT
            std::cout << "  - " << cell->getName() << ": " << biasScaling
                << std::endl;
#endif
        }
    }

    fuseScalingCells();

    return biasScalings;
}

std::unordered_map<std::string, long double> N2D2::DeepNetQuantization::quantizeFreeParemetersPerOutputCh(
                                                                            std::size_t nbBits) 
{
#ifdef VERBOSE_QUANT
    std::cout << "  Quantizing free parameters [per output channel]:" << std::endl;
#endif

    std::unordered_map<std::string, long double> biasScalings;

    std::vector<std::vector<std::string>> layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            long double biasScaling = getMaxParentsScaling(cell, biasScalings);
            rescaleParentsToScaling(cell, biasScalings, biasScaling);


            const long double wQuantScaling = std::pow(2, nbBits - 1) - 1;
            const long double bQuantScaling = DeepNetExport::isCellInputsUnsigned(*cell)?
                                                  wQuantScaling*(std::pow(2, nbBits) - 1):
                                                  wQuantScaling*(std::pow(2, nbBits - 1) - 1);


            const std::pair<Float_T, Float_T> wMinMax
                = cell->getFreeParametersRange(Cell::Multiplicative);
            const Float_T wScalingCell = Utils::max_abs(wMinMax.first, wMinMax.second);
            if(wScalingCell != 0.0) {
                std::vector<Float_T> scalingPerOutput(cell->getNbOutputs());
                for(std::size_t output = 0; output < cell->getNbOutputs(); output++) {
                    const auto woMinMax
                        = cell->getFreeParametersRangePerOutput(output,
                                                          Cell::Multiplicative);
                    const Float_T wScalingCellOutput = std::max(
                                                         std::min(wScalingCell, 0.1f), 
                                                         Utils::max_abs(woMinMax.first, woMinMax.second)
                                                       );

                    cell->processFreeParametersPerOutput([&](Float_T w) { 
                                                            return w*(wQuantScaling/wScalingCellOutput); 
                                                         }, output, Cell::Multiplicative);
                    cell->processFreeParametersPerOutput([&](Float_T b) { 
                                                             return b*(bQuantScaling/
                                                                       (biasScaling*wScalingCellOutput)); 
                                                         }, output, Cell::Additive);

                    scalingPerOutput[output] = wScalingCellOutput/wScalingCell;
                }


                auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*cell))
                                        (mDeepNet, 
                                         mDeepNet.generateNewCellName(cell->getName() + "_rescale_params"), 
                                         cell->getNbOutputs(), 
                                         Scaling::floatingPointScaling(
                                             std::move(scalingPerOutput),
                                             false,
                                             std::vector<Float_T>(0.0f))
                                        );
                mDeepNet.addCellAfter(scalingCell, cell);


                biasScaling *= wScalingCell;
                biasScalings[scalingCell->getName()] = biasScaling;
            }


            biasScalings[cell->getName()] = biasScaling;

#ifdef VERBOSE_QUANT
            std::cout << "  - " << cell->getName() << ": " << biasScaling
                << std::endl;
#endif
        }
    }

    fuseScalingCells();

    return biasScalings;
}

void N2D2::DeepNetQuantization::quantizeActivations(
                const std::unordered_map<std::string, Histogram>& outputsHistogram,
                const std::unordered_map<std::string, RangeStats>& outputsRange,
                std::unordered_map<std::string, long double>& biasScalings,
                std::size_t nbBits, ClippingMode actClippingMode, double quantileValue)
{
#ifdef VERBOSE_QUANT
    std::cout << "  Quantizing activations:" << std::endl;
#endif

    std::unordered_map<std::string, long double> activationScalings;
    
    std::vector<std::vector<std::string>> layers = mDeepNet.getLayers();
    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            std::shared_ptr<Cell_Frame_Top> cellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);
            if(!cell || !cellFrame) {
                throw std::runtime_error("Invalid cell.");
            }

            const long double prevActivationScaling = getMaxParentsScaling(cell, activationScalings);
            rescaleParentsToScaling(cell, activationScalings, prevActivationScaling);


            long double activationScaling;

            const std::shared_ptr<Activation>& activation = cellFrame->getActivation();
            if(cell->getType() == ElemWiseCell::Type) {
                activationScaling = getCellThreshold(cell->getName(),
                                                    outputsHistogram, outputsRange, 
                                                    nbBits, ClippingMode::NONE);
            }
            else if(cell->getType() == PaddingCell::Type || 
                    cell->getType() == PoolCell::Type || 
                    cell->getType() == ResizeCell::Type || 
                    cell->getType() == ScalingCell::Type || 
                    cell->getType() == SoftmaxCell::Type || 
                    cell->getType() == TransposeCell::Type)
            {
                activationScalings[cell->getName()] = prevActivationScaling;
                continue;
            }
            else if(activation) {
                const bool clip =  cell->getNbOutputs() > 2 && 
                                   (activation->getType() == RectifierActivation::Type || 
                                    activation->getType() == LinearActivation::Type || 
                                    activation->getType() == SaturationActivation::Type);
                

                /**
                 * When clipping with MSE or KL-Divergence and the next cell is a max pooling cell,
                 * use the histogram of the max pooling to calculate the clipping threshold.
                 */
                auto childrenCells = cell->getChildrenCells();
                const bool isNextCellMaxPool = childrenCells.size() == 1 && 
                                               childrenCells[0]->getType() == PoolCell::Type && 
                                               dynamic_cast<const PoolCell&>(*childrenCells[0]).getPooling() == PoolCell::Max;


                const std::string cellStatsName = clip && isNextCellMaxPool?childrenCells[0]->getName():
                                                                            cell->getName();
                activationScaling = getCellThreshold(cellStatsName, 
                                                    outputsHistogram, outputsRange, 
                                                    nbBits, clip?actClippingMode:ClippingMode::NONE, quantileValue);
            }
            else {
                throw std::runtime_error("Quantization of cell '" + cell->getName() + "' of type '" + 
                                         cell->getType() + "' is not supported yet.");
            }

            const long double biasScaling = biasScalings.at(cell->getName());

#ifdef VERBOSE_QUANT
            std::cout << "  - " << cell->getName() << ": "
                << "prev=" << prevActivationScaling
                << ", act=" << activationScaling
                << ", bias=" << biasScaling << std::endl;
#endif

            activationScaling /= biasScaling;
            activationScaling = (activationScaling == 0.0)?1.0:activationScaling;

            activationScalings[cell->getName()] = activationScaling;

            cell->processFreeParameters([&](Float_T d) { return d/prevActivationScaling; },
                                        Cell::Additive);
                                        

            const long double actQuantScaling = getActivationQuantizationScaling(*cell, nbBits);
            auto scalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*cell))
                                    (mDeepNet, 
                                     mDeepNet.generateNewCellName(cell->getName() + "_rescale_act"), 
                                     cell->getNbOutputs(), 
                                     Scaling::floatingPointScaling(
                                         std::vector<Float_T>(cell->getNbOutputs(), 
                                            (prevActivationScaling/activationScaling)/actQuantScaling
                                         ),
                                         false,
                                         std::vector<Float_T>(0.0f)
                                     )
                                    );

            mDeepNet.addCellAfter(scalingCell, cell);

            activationScalings[scalingCell->getName()] = activationScalings[cell->getName()];
            biasScalings[scalingCell->getName()] = biasScaling;

#ifdef VERBOSE_QUANT
            std::cout << "      quant=" << actQuantScaling
                << ", global scaling=" << Utils::cnotice << activationScaling
                << Utils::cdef << " -> cell scaling=" << Utils::cwarning
                    << ((prevActivationScaling/activationScaling)
                                /actQuantScaling)
                << Utils::cdef << std::endl;
#endif
        }
    }

    fuseScalingCells();
}

double N2D2::DeepNetQuantization::getActivationQuantizationScaling(const Cell& cell, std::size_t nbBits) const {
    const double unsignedMax = std::pow(2, nbBits) - 1;
    const double signedMax = std::pow(2, nbBits - 1) - 1;

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const std::shared_ptr<Activation>& activation = cellFrame.getActivation();

    if(!activation || cell.getType() == ElemWiseCell::Type) {
        return 1.0;
    }

    const std::string activationType = activation->getType();
    
    if(activationType == LogisticActivation::Type || 
       activationType == LogisticActivation::TypeWithLoss) 
    {
        return 2*(DeepNetExport::isCellInputsUnsigned(cell)?
                     signedMax*unsignedMax/signedMax:
                     signedMax*signedMax/signedMax);
    }
    else if(DeepNetExport::isCellOutputUnsigned(cell)) {
        return DeepNetExport::isCellInputsUnsigned(cell)?
                   signedMax*unsignedMax/unsignedMax:
                   signedMax*signedMax/unsignedMax;
    }
    else {
        return DeepNetExport::isCellInputsUnsigned(cell)?
                   signedMax*unsignedMax/signedMax:
                   signedMax*signedMax/signedMax;
    }
}

void N2D2::DeepNetQuantization::fuseScalingCells() {
#ifdef VERBOSE_QUANT
    std::cout << "  Fuse scaling cells:" << std::endl;
#endif

    // Get a copy, the loop may modify the graph
    const std::vector<std::vector<std::string>> layers = mDeepNet.getLayers();

    for (auto itLayer = layers.begin() + 1; itLayer != layers.end(); ++itLayer) {
        for (auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = mDeepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            if(cell->getType() != ScalingCell::Type) {
                continue;
            }

            auto parentsCells = cell->getParentsCells();
            if(parentsCells.size() != 1 || !parentsCells.front() || 
               parentsCells.front()->getChildrenCells().size() != 1) 
            {
                continue;
            }


            auto scalingCell = std::dynamic_pointer_cast<ScalingCell>(cell);

            auto parentCell = parentsCells.front();
            auto parentCellFrame = std::dynamic_pointer_cast<Cell_Frame_Top>(parentCell);
            if(!parentCellFrame) {
                throw std::runtime_error("Invalid cell.");
            }

#ifdef VERBOSE_QUANT
            std::cout << "  - fuse: " << cell->getName() << std::endl;
#endif

            std::shared_ptr<Activation> parentCellActivation = parentCellFrame->getActivation();
            if(parentCellActivation)  {
                fuseScalingCellWithParentActivation(scalingCell, *parentCellActivation);
            }
            else if(parentCell->getType() == ScalingCell::Type) {
                auto parentScalingCell = std::dynamic_pointer_cast<ScalingCell>(parentCell);
                fuseScalingCellWithParentScalingCell(scalingCell, parentScalingCell);
            }
            else if(parentCell->getType() == ElemWiseCell::Type) {
                auto parentElemWiseCell = std::dynamic_pointer_cast<ElemWiseCell>(parentCell);
                const bool moved = moveScalingCellAboveParentElemWiseCell(scalingCell, parentElemWiseCell);

                if (moved) {
                    // The ScalingCell has been moved as parent of the ElemeWiseCell.
                    // Recurse to try to merge this ScalingCell with its new parents.
                    fuseScalingCells();
                    return;
                }
            }
        }
    }
}

void N2D2::DeepNetQuantization::fuseScalingCellWithParentActivation(
                                                    const std::shared_ptr<ScalingCell>& scalingCell, 
                                                    Activation& parentCellActivation)
{
    const std::vector<Float_T>& scalingPerOutput = scalingCell->getScaling()
                                                                .getFloatingPointScaling()
                                                                .getScalingPerOutput();
    const ScalingMode parentScalingMode = parentCellActivation.getActivationScaling().getMode();
    if(parentScalingMode == ScalingMode::NONE) {
        parentCellActivation.setActivationScaling(
            std::move(scalingCell->getScaling())
        );

//#ifdef VERBOSE_QUANT
//        std::cout << "      with parent act: NONE"
//            << " x " << scalingPerOutput[0] << " -> "
//            << Utils::cwarning << scalingPerOutput[0]
//            << Utils::cdef << std::endl;
//#endif

        mDeepNet.removeCell(scalingCell);
    }
    else if(parentScalingMode == ScalingMode::FLOAT_MULT) {
        std::vector<Float_T> parentScalingPerOutput = parentCellActivation.getActivationScaling()
                                                                          .getFloatingPointScaling()
                                                                          .getScalingPerOutput();

#ifdef VERBOSE_QUANT
        std::cout << "      with parent act: " << parentScalingPerOutput[0]
            << " x " << scalingPerOutput[0] << " -> "
            << Utils::cwarning
            << (scalingPerOutput[0] * parentScalingPerOutput[0])
            << Utils::cdef << std::endl;
#endif

        for(std::size_t o = 0; o < parentScalingPerOutput.size(); o++) {
            parentScalingPerOutput[o] *= scalingPerOutput[o];
        }

        parentCellActivation.setActivationScaling(
            Scaling::floatingPointScaling(std::move(parentScalingPerOutput), false, std::vector<Float_T>(0.0f))
        );

        mDeepNet.removeCell(scalingCell);
    }
}

void N2D2::DeepNetQuantization::fuseScalingCellWithParentScalingCell(
                                        const std::shared_ptr<ScalingCell>& scalingCell, 
                                        const std::shared_ptr<ScalingCell>& parentScalingCell)
{
    assert(scalingCell->getNbOutputs() == parentScalingCell->getNbOutputs());

    const std::vector<Float_T>& scalingPerOutput = scalingCell->getScaling()
                                                               .getFloatingPointScaling()
                                                               .getScalingPerOutput();
    std::vector<Float_T> parentScalingPerOutput = parentScalingCell->getScaling()
                                                                    .getFloatingPointScaling()
                                                                    .getScalingPerOutput();

#ifdef VERBOSE_QUANT
    std::cout << "      with parent scaling: " << parentScalingPerOutput[0]
        << " x " << scalingPerOutput[0] << " -> "
        << Utils::cwarning << (scalingPerOutput[0] * parentScalingPerOutput[0])
        << Utils::cdef << std::endl;
#endif

    for(std::size_t o = 0; o < parentScalingPerOutput.size(); o++) {
        parentScalingPerOutput[o] *= scalingPerOutput[o];
    }

    parentScalingCell->setScaling(Scaling::floatingPointScaling(std::move(parentScalingPerOutput), false, std::vector<Float_T>(0.0f)));

    mDeepNet.removeCell(scalingCell);
}

bool N2D2::DeepNetQuantization::moveScalingCellAboveParentElemWiseCell(
                                        const std::shared_ptr<ScalingCell>& scalingCell, 
                                        const std::shared_ptr<ElemWiseCell>& parentElemWiseCell)
{
    const auto& weights = parentElemWiseCell->getWeights();
    const auto& shifts = parentElemWiseCell->getShifts();

    if(parentElemWiseCell->getOperation() == ElemWiseCell::Sum && 
        std::all_of(weights.begin(), weights.end(), [](Float_T v) { return v == 1.0; }) && 
        std::all_of(shifts.begin(), shifts.end(), [](Float_T v) { return v == 0.0; })) 
    {
        const std::vector<Float_T>& scalingPerOutput = scalingCell->getScaling()
                                                                   .getFloatingPointScaling()
                                                                   .getScalingPerOutput();

#ifdef VERBOSE_QUANT
        std::cout << "      move above parent: "
            << scalingPerOutput[0] << std::endl;
#endif

        auto grandParentsCells = parentElemWiseCell->getParentsCells();
        for(auto grandParentCell: grandParentsCells) {
            auto grandParentScalingCell = Registrar<ScalingCell>::create<Float_T>(getCellModelType(*grandParentCell))
                                            (mDeepNet, 
                                             mDeepNet.generateNewCellName(grandParentCell->getName() + "_rescale_elemwise"), 
                                             grandParentCell->getNbOutputs(), 
                                             Scaling::floatingPointScaling(scalingPerOutput, false, std::vector<Float_T>(0.0f)));

            mDeepNet.addCellBetween(grandParentScalingCell, grandParentCell, parentElemWiseCell);
        }

        mDeepNet.removeCell(scalingCell);
        return true;
    }

    return false;
}

void N2D2::DeepNetQuantization::approximateScalingCell(ScalingCell& cell, ScalingMode scalingCellMode, 
                                                       std::size_t /*nbBits*/) 
{
    assert(cell.getScaling().getMode() == ScalingMode::FLOAT_MULT);
    if(scalingCellMode == ScalingMode::FLOAT_MULT) {
        return;
    }

    if(scalingCellMode != ScalingMode::FIXED_MULT16
        && scalingCellMode != ScalingMode::FIXED_MULT32)
    {
        throw std::runtime_error("The scaling cell can only be approximated by a fixed-point approximation.");
    }

    const std::vector<Float_T>& floatScalingPerOutput = cell.getScaling()
                                                            .getFloatingPointScaling()
                                                            .getScalingPerOutput();

#ifdef VERBOSE_QUANT
    std::cout << "  - " << cell.getName() << ": " << floatScalingPerOutput[0]
        << std::endl;
#endif

    auto scalingFixedPoint = approximateScalingWithFixedPoint(scalingCellMode,
                                                        floatScalingPerOutput);

    cell.setScaling(Scaling::fixedPointScaling(scalingCellMode,
                                               scalingFixedPoint.first,
                                               scalingFixedPoint.second, 
                                               false, std::vector<Float_T>(0)));

#ifdef VERBOSE_QUANT
    std::cout << "    FIXED_MULT"
        << ((scalingCellMode == ScalingMode::FIXED_MULT32) ? 32 : 16)
        << ": " << scalingFixedPoint.second[0]
        << " x 2 ^ [- " << scalingFixedPoint.first << "]" << std::endl;
#endif
}

std::pair<std::size_t, std::vector<std::int32_t>>
N2D2::DeepNetQuantization::approximateScalingWithFixedPoint(
    ScalingMode& mode,
    const std::vector<Float_T>& scalingPerOutput)
{
    /**
     * Find the highest nbFractionalBits so that the scaling 
     * 'std::round(sc * (1ull << nbFractionalBits)' of each output
     * can be stored in an int32_t or int16_t an thus in scalingFixedPoint.
     * 
     * TODO With unsigned activation like ReLU we could use the maximum
     * of an uint32_t to gain a bit more precision.
     */
    const std::uint64_t limit
        = (mode == ScalingMode::FIXED_MULT16)
            ? std::numeric_limits<std::int16_t>::max()
            : std::numeric_limits<std::int32_t>::max();
    const double maxScaling = *std::max_element(scalingPerOutput.begin(),
                                                scalingPerOutput.end());

    if (maxScaling >= limit) {
        if (mode == ScalingMode::FIXED_MULT16) {
            std::cout << Utils::cwarning << "Max scaling (" << maxScaling
                << ") doesn't fit in FIXED_MULT16. "
                "Falling back to FIXED_MULT32." << Utils::cdef << std::endl;

            mode = ScalingMode::FIXED_MULT32;
            return approximateScalingWithFixedPoint(mode, scalingPerOutput);
        }
        else {
            std::stringstream errorStr;
            errorStr << "Max scaling (" << maxScaling
                << ") doesn't fit in FIXED_MULT32." << std::endl;

            throw std::runtime_error(errorStr.str());
        }
    }

    const std::size_t maxNbFractionalBits = 50;
    const std::size_t nbFractionalBits
        = std::min((std::size_t)std::floor(std::log(limit / maxScaling)
                                            / std::log(2.0)),
                   maxNbFractionalBits);

    std::vector<std::int32_t> scalingFixedPoint;
    for(auto sc: scalingPerOutput) {
        scalingFixedPoint.push_back(std::round(sc * (1ull << nbFractionalBits)));
    }

    return std::make_pair(nbFractionalBits, scalingFixedPoint);
}

std::pair<std::vector<unsigned char>, double> N2D2::DeepNetQuantization::approximateScalingWithPowerOf2Divs(
                                                        Float_T scaling, std::size_t nbDivisions) 
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
            // std::cout << scaling*(1.0 - precision) << std::endl;
            const std::size_t exponent = std::ceil(std::log2(1.0/(scaling*(1.0 - precision))));
            // std::cout << scaling*std::pow(2, exponent) << std::endl;
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

bool N2D2::DeepNetQuantization::checkActivationScalingWithPowerOf2Divs(
    Cell& cell, 
    const std::vector<Float_T>& scalingPerOutput)
{
    for(std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        if (scalingPerOutput[output] > 1.0)
            return false;
    }

    return true;
}

std::vector<std::vector<unsigned char>> N2D2::DeepNetQuantization::approximateActivationScalingWithPowerOf2Divs(Cell& cell, 
                                                        const std::vector<Float_T>& scalingPerOutput, 
                                                        std::size_t nbDivisions)
{
    std::vector<std::vector<unsigned char>> exponentsPerOutput(cell.getNbOutputs());
    for(std::size_t output = 0; output < cell.getNbOutputs(); output++) {
        Float_T rescaleOutputsBy;
        if(nbDivisions == 1) {
            const auto singleDivApprox = approximateScalingWithPowerOf2Divs(scalingPerOutput[output], 1);

            exponentsPerOutput[output] = std::move(singleDivApprox.first);
            rescaleOutputsBy = 1/singleDivApprox.second;
        }
        else if(nbDivisions == 2) {
            const auto doubleDivApprox = approximateScalingWithPowerOf2Divs(scalingPerOutput[output], 2);

            exponentsPerOutput[output] = std::move(doubleDivApprox.first);
            rescaleOutputsBy = 1/doubleDivApprox.second;
        }
        else {
            throw std::runtime_error("Currently only an approximation with 1 or 2 divisions is supported.");
        }

        // Rescale the weights and biasses of the cell to compensate the lost precision
        // of the approximation.
        cell.processFreeParametersPerOutput([&](Float_T d){ 
                                                return rescaleOutputsBy*d; 
                                            }, output);
    }

    return exponentsPerOutput;
}

double N2D2::DeepNetQuantization::getCellThreshold(const std::string& cellName,
                                       const std::unordered_map<std::string, Histogram>& outputsHistogram,
                                       const std::unordered_map<std::string, RangeStats>& outputsRange,
                                       std::size_t nbBits, ClippingMode actClippingMode, double quantileValue) 
{
    switch(actClippingMode) {
        case ClippingMode::KL_DIVERGENCE:
            return outputsHistogram.at(cellName).calibrateKLDivergence(nbBits);
        case ClippingMode::MSE:
            return outputsHistogram.at(cellName).calibrateMSE(nbBits);
        case ClippingMode::QUANTILE:
            return outputsHistogram.at(cellName).getQuantileValue(quantileValue);
        default: {
            const auto& range = outputsRange.at(cellName);
            return Utils::max_abs(range.minVal(), range.maxVal());
        }
    }
}

void N2D2::DeepNetQuantization::approximateActivationScaling(Cell& cell, Activation& activation,
                                                             ScalingMode actScalingMode) 
{
    assert(activation.getActivationScaling().getMode() == ScalingMode::FLOAT_MULT);

    const std::vector<Float_T>& scalingPerOutput = activation.getActivationScaling()
                                                             .getFloatingPointScaling()
                                                             .getScalingPerOutput();

#ifdef VERBOSE_QUANT
    std::cout << "  - " << cell.getName() << ": " << scalingPerOutput[0]
        << std::endl;
#endif

    if ((actScalingMode == ScalingMode::SINGLE_SHIFT
        || actScalingMode == ScalingMode::DOUBLE_SHIFT)
            && !checkActivationScalingWithPowerOf2Divs(cell, scalingPerOutput))
    {
        std::cout << Utils::cwarning << "Scaling (" << scalingPerOutput[0]
            << ") > 1 for layer \"" << cell.getName() << "\" is "
            "not supported with Single/Double-shift scaling. "
            "Falling back to Fixed-point scaling for this layer."
            << Utils::cdef << std::endl;

        actScalingMode = ScalingMode::FIXED_MULT16;
    }

    if(actScalingMode == ScalingMode::FLOAT_MULT) {
        // Nothing to do.
    }
    else if(actScalingMode == ScalingMode::FIXED_MULT32
        || actScalingMode == ScalingMode::FIXED_MULT16)
    {
        auto scalingFixedPoint
            = approximateScalingWithFixedPoint(actScalingMode,
                                               scalingPerOutput);

        activation.setActivationScaling(
            Scaling::fixedPointScaling(actScalingMode,
                                       scalingFixedPoint.first,
                                       scalingFixedPoint.second, 
                                       false, std::vector<Float_T>(0)));

#ifdef VERBOSE_QUANT
        std::cout << "    FIXED_MULT"
            << ((actScalingMode == ScalingMode::FIXED_MULT32) ? 32 : 16)
            << ": " << scalingFixedPoint.second[0]
            << " x 2 ^ [- " << scalingFixedPoint.first << "]" << std::endl;
#endif
    }
    else if(actScalingMode == ScalingMode::SINGLE_SHIFT) {
        std::vector<unsigned char> shifts;
        for(const auto& powOf2Exponents: approximateActivationScalingWithPowerOf2Divs(cell, scalingPerOutput, 1)) {
            assert(powOf2Exponents.size() == 1);
            shifts.push_back(powOf2Exponents[0]);
        }

        activation.setActivationScaling(Scaling::singleShiftScaling(shifts, false, std::vector<Float_T>(0)));

#ifdef VERBOSE_QUANT
        std::cout << "    SINGLE_SHIFT: 2 ^ [- " << (int)shifts[0] << "]" << std::endl;
#endif
    }
    else if(actScalingMode == ScalingMode::DOUBLE_SHIFT) {
        std::vector<std::pair<unsigned char, unsigned char>> shifts;
        for(const auto& powOf2Exponents: approximateActivationScalingWithPowerOf2Divs(cell, scalingPerOutput, 2)) {
            assert(powOf2Exponents.size() == 2);
            assert(powOf2Exponents[0] <= powOf2Exponents[1]);
            shifts.push_back({powOf2Exponents[1] - powOf2Exponents[0], powOf2Exponents[1]});
        }
        std::vector<std::pair<unsigned char, unsigned char>> vClipping;
        vClipping.push_back(std::make_pair(0,0));

        activation.setActivationScaling(Scaling::doubleShiftScaling(shifts, false, vClipping));

#ifdef VERBOSE_QUANT
        std::cout << "    DOUBLE_SHIFT: 2 ^ [- " << (int)shifts[0].first << "] "
            << " + 2 ^ [- " << (int)shifts[0].second << "]" << std::endl;
#endif
    }
    else {
        throw std::runtime_error("Unsupported scaling mode.");
    }
}

std::string N2D2::DeepNetQuantization::getCellModelType(const Cell& cell) {
    const Cell_Frame_Top& cellFrameTop = dynamic_cast<const Cell_Frame_Top&>(cell);
    if(cellFrameTop.isCuda()) {
        return Cell_Frame_Top::FRAME_CUDA_TYPE;
    }
    else {
        return Cell_Frame_Top::FRAME_TYPE;
    }
}
