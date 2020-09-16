/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <stdexcept>
#include <string>

#include "DeepNet.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell_Frame.hpp"


static const N2D2::Registrar<N2D2::CPP_ConcatCell> registrar(
                    "Frame", N2D2::CPP_ConcatCell_Frame::create);


std::shared_ptr<N2D2::CPP_ConcatCell> N2D2::CPP_ConcatCell_Frame::create(
                                        const DeepNet& deepNet, const std::string& name,
                                        unsigned int nbOutputs)
{
    return std::make_shared<CPP_ConcatCell_Frame>(deepNet, name, nbOutputs);
}

N2D2::CPP_ConcatCell_Frame::CPP_ConcatCell_Frame(const DeepNet& deepNet, const std::string& name,
                                                                 unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs),
      CPP_ConcatCell(deepNet, name, nbOutputs),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs)
{
}

void N2D2::CPP_ConcatCell_Frame::initialize() {
}

void N2D2::CPP_ConcatCell_Frame::propagate(bool /*inference*/) {
    throw std::runtime_error("propagate not supported yet.");
}

void N2D2::CPP_ConcatCell_Frame::backPropagate() {
    throw std::runtime_error("backPropagate not supported yet.");
}

void N2D2::CPP_ConcatCell_Frame::update() {
    throw std::runtime_error("update not supported yet.");
}

void N2D2::CPP_ConcatCell_Frame::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
}

std::pair<double, double> N2D2::CPP_ConcatCell_Frame::getOutputsRange() const {
    return CPP_ConcatCell::getOutputsRange();
}
