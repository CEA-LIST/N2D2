/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Database/Relational_Database.hpp"

N2D2::Relational_Database::Relational_Database(double validation)
    : Database(), mValidation(validation)
{
    // ctor
}

// TODO: Extend this to stimuli from a database
void N2D2::Relational_Database::load(const std::string& /*dataPath*/,
                                    const std::string& /*labelPath*/,
                                    bool /*extractROIs*/)
{
    throw std::runtime_error("Relational_Database::load not implemented");
}



std::vector<std::pair<std::vector<double>, double>>
N2D2::Relational_Database::loadRelationSample(double* triple)
{
    double maxValue = 1.0;
    unsigned int stimulusSize = 100;
    unsigned int numberVariables = 3;
    std::vector<std::pair<std::vector<double>, double>> sample;
    for (unsigned int k=0; k<numberVariables; k++){

        std::pair<std::vector<double>, double> stimulusVariable;
        double variableValue = Random::randUniform(0.0, 1.0);
        if (!(triple==nullptr)) {
            variableValue = triple[k];
        }
        if (k == numberVariables-1){
            double sum = 0;
            for (unsigned int p=0; p<numberVariables-1; p++){
                sum += sample[p].second;
            }
            variableValue = sum;
        }
        if (variableValue >= 1.0){
            variableValue = variableValue - 1.0;
        }
        std::cout << "Var " << k << " " << variableValue << std::endl;
        stimulusVariable = std::make_pair(std::vector<double>(stimulusSize), variableValue);

        double slope = 2*maxValue/stimulusSize;
        double centerVal = variableValue*stimulusSize;

        for (unsigned int x=0; x<stimulusSize; x++){
            double diffVal = (double)x - centerVal;
            double yVal = maxValue - slope*std::fabs(diffVal);
            yVal =  yVal < 0 ? -yVal : yVal;
            stimulusVariable.first[x] = yVal;
        }

        sample.push_back(stimulusVariable);

    }
    return sample;

}




