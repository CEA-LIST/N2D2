/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "utils/Pruning.hpp"
#include "utils/Utils.hpp"

#include "Cell/ConvCell.hpp"

N2D2::Pruning::Pruning(const N2D2::Pruning::Prune_T pruneName)
    : // Variables
      mName(pruneName)
{
    // ctor
}

N2D2::Pruning::Pruning(const std::string pruneStr)
{
    // ctor
    std::stringstream ss;
    ss << pruneStr;
    Utils::signChecked<Prune_T>(ss) >> mName;
}

void N2D2::Pruning::apply(std::shared_ptr<N2D2::DeepNet>& deepNet, 
                          std::vector<float> opts)
{
    switch (mName) {
    case Random:
        {
            float threshold = opts[0];
            prune_random(deepNet, threshold);
            break;
        }

    case None:
    default:
        throw std::runtime_error("Unknown pruning method");
    }
}


// ----------------------------------------------------------------------------
// ----------------------------- Pruning methods ------------------------------
// ----------------------------------------------------------------------------

void N2D2::prune_random(std::shared_ptr<DeepNet>& deepNet, 
                        const float threshold)
{
    std::shared_ptr<Cell> cell = deepNet->getCell(deepNet->getLayers()[1][0]);
    prune_random(cell, threshold);
}

void N2D2::prune_random(std::shared_ptr<Cell>& cell, 
                        const float threshold)
{
    const std::string cellType = cell->getType();

    if (cellType == "Conv") {

        // Get weights from 1st layer
        std::shared_ptr<ConvCell> convCell = std::dynamic_pointer_cast<ConvCell>(cell);
        Tensor<float> weights = tensor_cast<float>((*convCell->getWeights())[0]);
        std::cout << "Weights before random pruning" << std::endl;
        std::cout << weights << std::endl;

        for (unsigned int i = 0; i < weights.size(); ++i) {
            weights(i) = (Random::randUniform(0.0, 1.0) > threshold) ? weights(i) : 0.0f;
        }

        //set those new weights to conv and check
        //not good multiplication factor for now, just testing
        std::cout << "Weights after random pruning" << std::endl;
        std::cout << weights << std::endl;
    } 
    else {
        throw std::runtime_error("No parameters to prune in that cell");
    }
}
