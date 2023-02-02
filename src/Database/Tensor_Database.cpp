/*
    (C) Copyright 2023 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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

#include "Database/Tensor_Database.hpp"

N2D2::Tensor_Database::Tensor_Database(int stimuliDepth=-16):
    mStimuliDepth(stimuliDepth)
{
    // TODO : maybe -16 is not the best default value ...
    // ctor
}

void N2D2::Tensor_Database::load(
    std::vector<Tensor<float>>& inputs,
    std::vector<int>& labels)
{
    assert(inputs.size() == labels.size());
    // Check there is no disperency with stimuli before method
    assert(mStimuli.size() == mStimuliData.size());
    assert(mStimuli.size() == mStimuliSets(Unpartitioned).size() + 
                              mStimuliSets(Test).size() + 
                              mStimuliSets(Validation).size() + 
                              mStimuliSets(Learn).size());
    // assert(mStimuli.size() == mStimuliLabelsData.size());
    // assert(mStimuli.size() == mStimuliTargetData.size());

    unsigned int nbStimuliToLoad = inputs.size();
    unsigned int oldNbStimuli = mStimuli.size();

    mStimuli.reserve(mStimuli.size() + nbStimuliToLoad);
    mStimuliData.reserve(mStimuliData.size() + nbStimuliToLoad);
    mStimuliSets(Unpartitioned).reserve(mStimuliSets(Unpartitioned).size() + nbStimuliToLoad);
    // mStimuliLabelsData.reserve(mStimuliLabelsData.size() + nbStimuliToLoad);
    // mStimuliTargetData.reserve(mStimuliTargetData.size() + nbStimuliToLoad);

    // TODO : for loop to fill mStimuliData, mStimuli, mStimulilabel, mStimuliTarget
    for(unsigned int i = 0; i < nbStimuliToLoad; ++i){
        mStimuliData.push_back((cv::Mat)inputs[i]); // TODO : Clone the cv mat ?
        std::ostringstream nameStr;
        nameStr << "RandomName[" << mStimuli.size() << "]";
        mStimuli.push_back(Stimulus(nameStr.str(), labels[i]));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
    }

    // Check there is no disperency with stimuli after method
    assert(mStimuli.size() == nbStimuliToLoad + oldNbStimuli);
    assert(mStimuli.size() == mStimuliData.size());
    assert(mStimuli.size() == mStimuliSets(Unpartitioned).size() + 
                              mStimuliSets(Test).size() + 
                              mStimuliSets(Validation).size() + 
                              mStimuliSets(Learn).size());
    // assert(mStimuli.size() == mStimuliLabelsData.size());
    // assert(mStimuli.size() == mStimuliTargetData.size());
}