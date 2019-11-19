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

#ifndef N2D2_DEEP_NET_QUANTIZATION_H
#define N2D2_DEEP_NET_QUANTIZATION_H

#include <string>
#include <unordered_map>
#include <vector>

#include "Histogram.hpp"
#include "ScalingMode.hpp"

namespace N2D2 {

class Activation;
class Cell;
class DeepNet;
class ElemWiseCell;
class RangeStats;
class ScalingCell;

class DeepNetQuantization {
public:
    DeepNetQuantization(DeepNet& deepNet);

    void clipWeights(std::size_t nbBits, ClippingMode wtClippingMode);


    void normalizeFreeParameters(Float_T normFactor = 1.0);
    void normalizeFreeParametersPerOutputChannel(Float_T normFactor = 1.0);
    void rescaleAdditiveParameters(Float_T rescaleFactor);


    void reportOutputsRange(std::unordered_map<std::string, RangeStats>& outputsRange) const;
    void reportOutputsHistogram(std::unordered_map<std::string, Histogram>& outputsHistogram,
                                const std::unordered_map<std::string, RangeStats>& outputsRange,
                                std::size_t nbBits, ClippingMode actClippingMode) const;


    void normalizeOutputsRange(const std::unordered_map<std::string, Histogram>& outputsHistogram,
                               const std::unordered_map<std::string, RangeStats>& outputsRange,
                               std::size_t nbBits,
                               ClippingMode actClippingMode);


    void quantizeNormalizedNetwork(std::size_t nbBits, ScalingMode actScalingMode);
    
private:
    void fuseScalingCells();
    void fuseScalingCellWithParentActivation(const std::shared_ptr<ScalingCell>& scalingCell, 
                                             Activation& parentCellActivation);
    void fuseScalingCellWithParentScalingCell(const std::shared_ptr<ScalingCell>& scalingCell, 
                                              const std::shared_ptr<ScalingCell>& parentScalingCell);
    
    /**
     * If possible, move the ScalingCell with an ElemWiseCell parent above the ElemWiseCell.
     * It then become the parent of the ElemeWiseCell and the child of all the original
     * parents of the ElemWiseCell.
     * 
     * This can be done if the ElemWiseCell is a simple addition multiple inputs.
     */
    void moveScalingCellAboveParentElemWiseCell(const std::shared_ptr<ScalingCell>& scalingCell, 
                                                const std::shared_ptr<ElemWiseCell>& parentElemWiseCell);

    std::string getCellModelType(const Cell& cell);

    Float_T getMaxParentsScaling(const std::shared_ptr<Cell>& cell, 
                                 const std::unordered_map<std::string, Float_T>& scalingForCell) const;
    void rescaleParentsToScaling(const std::shared_ptr<Cell>& cell, 
                                 const std::unordered_map<std::string, Float_T>& scalingForCell,
                                 Float_T scaling);

    static void quantizeActivationScaling(Cell& cell, Activation& activation, 
                                          std::size_t nbBits, 
                                          ScalingMode actScalingMode);
    static void quantizeFreeParemeters(Cell& cell, std::size_t nbBits);

    static double getCellThreshold(const std::string& cellName,
                                   const std::unordered_map<std::string, Histogram>& outputsHistogram,
                                   const std::unordered_map<std::string, RangeStats>& outputsRange,
                                   std::size_t nbBits, ClippingMode actClippingMode);

    static void approximateActivationScaling(Cell& cell, Activation& activation,
                                             ScalingMode actScalingMode);
    
    static std::vector<std::vector<unsigned char>> approximateActivationScalingWithPowerOf2Divs(Cell& cell, 
                                                const std::vector<Float_T>& scalingPerOutput, 
                                                std::size_t nbDivisions);

    /**
     * Approximate the multiplicative scaling factor by an addition of power of two divisions.
     * 
     * The following multiplication '1231 * 0.0119 = 14.6489' can be approximate with 
     *
     * - one power of two divison:
     *       1231/128 = 9.6172 (precision of (1/0.0119)/128 = 0.6565)
     * 
     * - two power of two divisions:
     *       1231/128 + 1231/256 = 14.4258 (precision of (1/0.0119)/128 + (1/0.0119)/256 = 0.9848)
     * 
     * - three power of two divisions:
     *       1231/128 + 1231/256 + 1231/8192 = 14.5760 (precision  of (1/0.0119)/128 + 
     *                                                                (1/0.0119)/256 + 
     *                                                                (1/0.0119)/8192 = 0.9950)
     * 
     * 
     * Return a pair with a vector of exponents and the precision of the approximation.
     */
    static std::pair<std::vector<unsigned char>, double> approximateScalingWithPowerOf2Divs(
                                                Float_T scaling, std::size_t nbDivisions);

    static void approximateScalingCell(ScalingCell& cell, ScalingMode scalingCellMode, 
                                       std::size_t nbBits);

    /**
     * Throw and exception if a cell with 'cellName' already exists.
     */
    void errorIfCellExist(const std::string& cellName) const;

private:
    DeepNet& mDeepNet;
};

}

#endif