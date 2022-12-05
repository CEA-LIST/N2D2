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
#include "Cell/FcCell.hpp"

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
    case Iter:
        {
            float threshold = opts[0];
            prune_iter_nonstruct(deepNet, threshold);
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
    for (std::map<std::string, std::shared_ptr<Cell> >::iterator
            itCells = deepNet->getCells().begin(),
            itCellsEnd = deepNet->getCells().end();
            itCells != itCellsEnd;
            ++itCells)
    {
        std::cout << "Pruning " << (*itCells).first << "..." << std::endl;
        prune_random((*itCells).second, threshold);
    }
}

void N2D2::prune_random(std::shared_ptr<Cell>& cell, 
                        const float threshold)
{
    const std::string cellType = cell->getType();

    if (cellType == "Conv") {

        std::shared_ptr<ConvCell> convCell = std::dynamic_pointer_cast<ConvCell>(cell);

        for (unsigned int output = 0; output < convCell->getNbOutputs(); ++output) {
            for (unsigned int channel = 0; channel < convCell->getNbChannels(); ++channel) {
                Tensor<float> kernel;
                convCell->getWeight(output, channel, kernel);

                for (unsigned int sx = 0; sx < convCell->getKernelWidth(); ++sx) {
                    for (unsigned int sy = 0; sy < convCell->getKernelHeight(); ++sy){
                        kernel(sx, sy) *= Random::randBernoulli(threshold);
                    }
                }

                convCell->setWeight(output, channel, kernel);
            }
        }
    } 
    else {
        std::cout << "No need to prune this " << cellType << " cell" << std::endl;
        // throw std::runtime_error("No parameters to prune in that cell");
    }
}

void N2D2::prune_iter_nonstruct(std::shared_ptr<DeepNet>& deepNet,
                        const float threshold)
{
    int count = 0;
    for (std::map<std::string, std::shared_ptr<Cell> >::iterator
            itCells = deepNet->getCells().begin(),
            itCellsEnd = deepNet->getCells().end();
            itCells != itCellsEnd;
            ++itCells)
    {
        /*
        count++;
        if(count > 1){
            break;
        }
        */
        std::cout << "Pruning " << (*itCells).first << "..." << std::endl;
        prune_iter_nonstruct((*itCells).second, threshold);
    }
}

/*
TODO in this method:
we need 2 params : sparsity thershold (set by user) and delta for a step from 0

get all weights in layer
loop over each weight and if weight is < 0 +/- delta, set 0 in mask tensor
!!! note : mask tensor has to be accessible during the fine-tuning procedure (when we call learn-epoch)

when we are done with all weights in the layer, check if number of 0th is >= the sparsity threshold
if it is smaller, we add increase the range to 2*delta and we repeat the procedure in a range 0 +/- 2*delta
etc till the criterion on sparsity level is met

here we need to fill the mask with 0th in right places

the last step is a fine-tuning, need to access and apply mask on weights and their gradients at each iteration
*/

void N2D2::prune_iter_nonstruct(std::shared_ptr<Cell>& cell,
                        const float threshold)
{
    const std::string cellType = cell->getType();

    if (cellType == "Conv") {
        std::shared_ptr<ConvCell> convCell = std::dynamic_pointer_cast<ConvCell>(cell);
        Tensor<float> weights = tensor_cast<float>((*convCell->getWeights())[0]);

        //std::cout << "weights init = " << weights << std::endl;

        float delta_step = 0.01;
        int zero_count= 0;
        unsigned int iter_delta = 0;
        Tensor<float> mask(weights.dims());
        bool maskFilled = false;

        while(!maskFilled){
            //check if the sparsity (within the range) is equal or above the requested thershold
            if( ((float)zero_count)/weights.size() < threshold){
                zero_count = 0;
                for (unsigned int i = 0; i < weights.size(); ++i) {
                    //check if the weight is inside the range
                    if (abs(weights(i)) < delta_step*iter_delta) {
                        zero_count++;
                    }
                }
            }
            else{
                //sparsity is ok, fill the mask
                for (unsigned int i = 0; i < weights.size(); ++i) {
                    if (abs(weights(i)) < delta_step*iter_delta) {
                        mask(i) = 0;
                    }
                    else{
                        mask(i) = 1;
                    }
                }
                //TODO :: here save the mask into right tensor (using method setMaskWeights from convCell_Frame_CUDA?)
                maskFilled = true;
            }
            //increase the range for sparsity
            iter_delta++;
        }

        //std::cout << "mask = " << mask << std::endl;

        // apply the mask to weights
        for (unsigned int i = 0; i < weights.size(); ++i) {
            weights(i) *= mask(i);
        }

        Tensor<float> weights_pruned = tensor_cast<float>((*convCell->getWeights())[0]);
        //std::cout << "weights pruned = " << weights_pruned << std::endl;

        //have to set weight for conv, if not pruned weights = init weights when we export them
        int i = 0;
        for (unsigned int output = 0; output < convCell->getNbOutputs(); ++output) {
            for (unsigned int channel = 0; channel < convCell->getNbChannels(); ++channel) {
                Tensor<float> kernel;
                convCell->getWeight(output, channel, kernel);

                for (unsigned int sy = 0; sy < convCell->getKernelHeight(); ++sy){
                    for (unsigned int sx = 0; sx < convCell->getKernelWidth(); ++sx) {
                        //std::cout << "output, channel, sy, sx, i = " << output << "," << channel << "," << sy << ",//"<< sx << "," << i << ": kernel, mask = " <<  kernel(sx, sy) << " ,  " << mask(i) << std::endl;
                        //here the swap sx and sy is correct, we need to do it to get correct matching between indexes
                        kernel(sx, sy) *= mask(i);
                        i++;
                    }
                }
                convCell->setWeight(output, channel, kernel);
            }
        }
    } 
    else {
        std::cout << "No need to prune this " << cellType << " cell" << std::endl;
        // throw std::runtime_error("No parameters to prune in that cell");
    }
}

