/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include "Network.hpp"
#include "env.hpp"

std::size_t N2D2::Network::inputHeight() const {
    return ENV_SIZE_Y;
}

std::size_t N2D2::Network::inputWidth() const {
    return ENV_SIZE_X;
}

std::size_t N2D2::Network::inputNbChannels() const {
    return ENV_NB_OUTPUTS;
}

std::size_t N2D2::Network::inputSize() const {
    return inputHeight()*inputWidth()*inputNbChannels();
}


std::size_t N2D2::Network::outputHeight(std::size_t index) const {
    return OUTPUTS_HEIGHT[index];
}

std::size_t N2D2::Network::outputWidth(std::size_t index) const {
    return OUTPUTS_WIDTH[index];
}

std::size_t N2D2::Network::outputNbOutputs(std::size_t index) const {
    return NB_OUTPUTS[index];
}

std::size_t N2D2::Network::outputSize(std::size_t index) const {
    return outputHeight(index)*outputWidth(index)*outputNbOutputs(index);
}
