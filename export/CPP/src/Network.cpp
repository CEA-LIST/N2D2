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



#if NETWORK_TARGETS != 1
#error "Only one target is supported for now"
#endif
std::size_t N2D2::Network::outputHeight() const {
    return OUTPUTS_HEIGHT[0];
}

std::size_t N2D2::Network::outputWidth() const {
    return OUTPUTS_WIDTH[0];
}

std::size_t N2D2::Network::outputNbOutputs() const {
    return NB_OUTPUTS[0];
}

std::size_t N2D2::Network::outputSize() const {
    return outputHeight()*outputWidth()*outputNbOutputs();
}
