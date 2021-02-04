/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Generator/ActivationGenerator.hpp"

std::shared_ptr<N2D2::Activation>
N2D2::ActivationGenerator::generate(IniParser& iniConfig,
                                    const std::string& section,
                                    const std::string& model,
                                    const DataType& dataType,
                                    const std::string& name,
                                    const std::shared_ptr
                                    <Activation>& defaultActivation,
                                    bool nullIfDefault)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    if (iniConfig.isProperty(name)) {
        const std::string type = iniConfig.getProperty<std::string>(name);

        std::shared_ptr<Activation> activation
            = Registrar<ActivationGenerator>::create(type)(
                iniConfig, section, model, dataType, name);

        activation->setPrefixedParameters(iniConfig.getSection(section),
                                          name + ".");
        std::shared_ptr<QuantizerActivation> quantizer 
            = QuantizerActivationGenerator::generate(iniConfig,
                                                     section,
                                                     model,
                                                     dataType, 
                                                     "QAct");

        if (quantizer) {
            activation->setQuantizer(quantizer);

            std::shared_ptr<Solver> quantizerSolver
                = SolverGenerator::generate(iniConfig, 
                                            section, 
                                            model, 
                                            dataType, 
                                            "QActSolver");
            std::cout << "Added " <<  activation->getQuantizer()->getType() << 
                " quantizer to " << type << " Activation " << std::endl; 

            if (quantizerSolver) {
                activation->getQuantizer()->setSolver(quantizerSolver);
            }
        }
        return activation;
    }
    else {
        if(!nullIfDefault) {
            return defaultActivation;
        } else {
            return nullptr;
        }
    }
}

void N2D2::ActivationGenerator::generateParams( const std::shared_ptr<Cell_Frame_Top>& cell,
                                                    IniParser& iniConfig,
                                                    const std::string& section,
                                                    const std::string& model,
                                                    const DataType& dataType)
{
    std::shared_ptr<Cell_Frame_Top> cellFrame
        = std::dynamic_pointer_cast<Cell_Frame_Top>(cell);

    std::shared_ptr<Activation> activation
        = generate( iniConfig,
                    section,
                    model,
                    dataType,
                    "ActivationFunction",
                    nullptr,
                    true );

    if (activation) {
        std::cout << "Modify Activation in Cell_Frame_Top" << std::endl;
        cell->setActivation(activation);
    }   

}
