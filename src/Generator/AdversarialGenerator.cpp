/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)

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

#include "Generator/AdversarialGenerator.hpp"

std::shared_ptr<N2D2::Adversarial> 
N2D2::AdversarialGenerator::generate(IniParser& iniConfig, const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const Adversarial::Attack_T attackName 
        = iniConfig.getProperty<Adversarial::Attack_T>("Attack", Adversarial::Attack_T::None);

    std::shared_ptr<Adversarial> adv(new Adversarial(attackName));

    if (adv->getAttackName() != Adversarial::Attack_T::None) {
        std::cout << "Adversarial attack: " << adv->getAttackName() << std::endl;
    }

    adv->setEps(iniConfig.getProperty<float>("Eps", adv->getEps()));
    adv->setNbIterations(iniConfig.getProperty<unsigned int>("NbIterations", adv->getNbIterations()));
    adv->setRandomStart(iniConfig.getProperty<bool>("RandomStart", adv->getRandomStart()));
    adv->setTargeted(iniConfig.getProperty<bool>("Targeted", adv->getTargeted()));

    return adv;
}