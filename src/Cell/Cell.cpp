/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "DeepNet.hpp"
#include "HeteroStimuliProvider.hpp"

unsigned int N2D2::Cell::mIdCnt = 1;

N2D2::Cell::Cell(const DeepNet& deepNet, const std::string& name, unsigned int nbOutputs)
    : mId(mIdCnt++),
      mDeepNet(deepNet),
      mName(name),
      mOutputsDims(std::vector<size_t>({1U, 1U, nbOutputs})),
      mGroupMap(0),
      mGroupMapInitialized(false)
{
    // ctor
}

void N2D2::Cell::addMultiscaleInput(HeteroStimuliProvider& sp,
                                    unsigned int x0,
                                    unsigned int y0,
                                    unsigned int width,
                                    unsigned int height,
                                    const Tensor<bool>& mapping)
{
    const unsigned int nbMaps = sp.size();

    if (!mapping.empty()) {
        unsigned int nbChannels = 0;

        for (unsigned int map = 0; map < nbMaps; ++map)
            nbChannels += sp[map]->getNbChannels();

        if (mapping.dimY() != nbChannels) {
            throw std::runtime_error("Cell::addMultiscaleInput(): number of "
                                     "mapping rows must be equal to the total "
                                     "number of input filters");
        }
    }

    if (width == 0)
        width = sp[0]->getSizeX() - x0;
    if (height == 0)
        height = sp[0]->getSizeY() - y0;

    unsigned int mappingOffset = 0;

    for (unsigned int map = 0; map < nbMaps; ++map) {
        const unsigned int nbChannels = sp[map]->getNbChannels();
        const double scale = sp[map]->getSizeX()
                             / (double)sp[0]->getSizeX(); // TODO: unsafe legacy
        // code adaptation
        const unsigned int scaledX0 = (unsigned int)(scale * x0);
        const unsigned int scaledY0 = (unsigned int)(scale * y0);
        const unsigned int scaledWidth = (unsigned int)(scale * width);
        const unsigned int scaledHeight = (unsigned int)(scale * height);

        addInput(
            *(sp[map]),
            scaledX0,
            scaledY0,
            scaledWidth,
            scaledHeight,
            (mapping.empty())
                ? Tensor<bool>()
                : mapping.rows(mappingOffset, nbChannels));
        mappingOffset += nbChannels;
    }
}

std::vector<std::shared_ptr<N2D2::Cell>> N2D2::Cell::getChildrenCells() const {
    return mDeepNet.getChildCells(mName);
}

std::vector<std::shared_ptr<N2D2::Cell>> N2D2::Cell::getParentsCells() const {
    return mDeepNet.getParentCells(mName);
}

void N2D2::Cell::save(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    // Save parameters
    std::ostringstream fileName;
    fileName << dirName << "/" << getType() << ".cfg";
    saveParameters(fileName.str());

    // Save free parameters
    fileName.str(std::string());
    fileName << dirName << "/" << getType() << ".syn";
    saveFreeParameters(fileName.str());
}

void N2D2::Cell::load(const std::string& dirName)
{
    // Load parameters
    std::ostringstream fileName;
    fileName << dirName << "/" << getType() << ".cfg";
    loadParameters(fileName.str());

    // Load free parameters
    fileName.str(std::string());
    fileName << dirName << "/" << getType() << ".syn";
    loadFreeParameters(fileName.str());
}

void N2D2::Cell::setInputsDims(std::initializer_list<size_t> dims)
{
    setInputsDims(std::vector<size_t>(dims));
}

void N2D2::Cell::setInputsDims(const std::vector<size_t>& dims)
{
    if (mInputsDims.empty())
        mInputsDims = dims;
    else if (dims.size() != mInputsDims.size()) {
        std::stringstream msgStr;
        msgStr << "Cell::setInputsDims(): trying to connect an input of"
            " dimension " << dims.size() << " (" << dims << "), but another"
            " input of dimension " << mInputsDims.size() << " (" << mInputsDims
            << ") already exists!" << std::endl;
        throw std::runtime_error(msgStr.str());
    }
    else {
        for (unsigned int n = 0, size = dims.size() - 1; n < size; ++n) {
            if (dims[n] != mInputsDims[n]) {
                std::stringstream msgStr;
                msgStr << "Cell::setInputsDims(): trying to connect an input of"
                    " dims (" << dims << "), but another input already exists"
                    " with dims (" << mInputsDims << ")" << std::endl;
                throw std::runtime_error(msgStr.str());
            }
        }

        mInputsDims.back() += dims.back();
    }
}

size_t N2D2::Cell::groupMap() const
{
    if (!mGroupMapInitialized) {
        mGroupMap = getNbGroups(mMapping);
        mGroupMapInitialized = true;
    }

    return mGroupMap;
}

size_t N2D2::Cell::getNbGroups(const Tensor<bool>& map) const
{
    const size_t nbOutputs = map.dimX();
    const size_t nbChannels = map.dimY();

    // Determine the number of groups
    size_t nbChannelsPerGroup = 0;
    for (; nbChannelsPerGroup < nbChannels && map(0, nbChannelsPerGroup);
        ++nbChannelsPerGroup) {}

    if (nbChannelsPerGroup == 0 || nbChannels % nbChannelsPerGroup != 0)
        return 0;

    const size_t nbGroups = nbChannels / nbChannelsPerGroup;

    if (nbOutputs % nbGroups != 0)
        return 0;

    const size_t nbOutputsPerGroup = nbOutputs / nbGroups;

    // Check that there are really only groups, with nbGroups groups
    size_t outputGroupOffset = 0;
    size_t channelGroupOffset = 0;

    for (size_t group = 0; group < nbGroups; ++group) {
        for (size_t output = outputGroupOffset;
            output < outputGroupOffset + nbOutputsPerGroup; ++output)
        {
            size_t channel = 0;

            for (; channel < channelGroupOffset; ++channel) {
                if (map(output, channel))
                    return 0;
            }

            for (; channel < channelGroupOffset + nbChannelsPerGroup;
                ++channel)
            {
                if (!map(output, channel))
                    return 0;
            }

            for (; channel < nbChannels; ++channel)
            {
                if (map(output, channel))
                    return 0;
            }
        }

        outputGroupOffset += nbOutputsPerGroup;
        channelGroupOffset += nbChannelsPerGroup;
    }

    return nbGroups;
}

std::pair<double, double> N2D2::Cell::getOutputsRangeParents() const {
    const auto& parentsCells = getParentsCells();
    if(parentsCells.empty()) {
        throw std::runtime_error("Cell " + getName() + " must have a parent.");
    }


    std::pair<double, double> range;
    for(auto it = parentsCells.begin(); it != parentsCells.end(); ++it) {
        const std::shared_ptr<Cell>& parent = *it;
        if(!parent) {
            throw std::runtime_error("Environment shouldn't be connected to cell " + getName() + ".");
        }

        const auto& parentFrame = dynamic_cast<const Cell_Frame_Top&>(*parent);
        const std::pair<double, double> parentOutputsRange = parentFrame.getOutputsRange();

        if(it == parentsCells.begin()) {
            range = parentOutputsRange;
        }
        else {
            range.first = std::min(range.first, parentOutputsRange.first);
            range.second = std::max(range.second, parentOutputsRange.second);
        }
    }

    return range;
}


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_Cell(py::module &m) {
    py::class_<Cell, std::shared_ptr<Cell>> cell(m, "Cell", py::multiple_inheritance());

    py::enum_<Cell::FreeParametersType>(cell, "FreeParametersType")
    .value("Additive", Cell::FreeParametersType::Additive)
    .value("Multiplicative", Cell::FreeParametersType::Multiplicative)
    .value("All", Cell::FreeParametersType::All)
    .export_values();

    cell.def("getId", &Cell::getId)
    .def("getName", &Cell::getName)
    .def("getType", &Cell::getType)
    .def("getNbChannels", &Cell::getNbChannels)
    .def("getChannelsWidth", &Cell::getChannelsWidth)
    .def("getChannelsHeight", &Cell::getChannelsHeight)
    .def("getInputsDim", &Cell::getInputsDim, py::arg("dim"))
    .def("getInputsDims", &Cell::getInputsDims)
    .def("getInputsSize", &Cell::getInputsSize)
    .def("getNbOutputs", &Cell::getNbOutputs)
    .def("getOutputsWidth", &Cell::getOutputsWidth)
    .def("getOutputsHeight", &Cell::getOutputsHeight)
    .def("getOutputsDim", &Cell::getOutputsDim, py::arg("dim"))
    .def("getOutputsDims", &Cell::getOutputsDims)
    .def("getOutputsSize", &Cell::getOutputsSize)
    .def("getStats", &Cell::getStats)
    .def("getReceptiveField", &Cell::getReceptiveField, py::arg("outputField") = std::vector<unsigned int>())
    .def("getAssociatedDeepNet", &Cell::getAssociatedDeepNet)
    .def("isConnection", &Cell::isConnection, py::arg("channel"), py::arg("output"))
    .def("addInput", (void (Cell::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const Tensor<bool>&)) &Cell::addInput, py::arg("sp"), py::arg("channel"), py::arg("x0"), py::arg("y0"), py::arg("width"), py::arg("height"), py::arg("mapping") = Tensor<bool>())
    .def("addInput", (void (Cell::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int, const Tensor<bool>&)) &Cell::addInput, py::arg("sp"), py::arg("x0") = 0, py::arg("y0") = 0, py::arg("width") = 0, py::arg("height") = 0, py::arg("mapping") = Tensor<bool>())
    .def("addMultiscaleInput", &Cell::addMultiscaleInput, py::arg("sp"), py::arg("x0") = 0, py::arg("y0") = 0, py::arg("width") = 0, py::arg("height") = 0, py::arg("mapping") = Tensor<bool>())
    .def("addInput", (void (Cell::*)(Cell*, const Tensor<bool>&)) &Cell::addInput, py::arg("cell"), py::arg("mapping") = Tensor<bool>())
    .def("addInput", (void (Cell::*)(Cell*, unsigned int, unsigned int, unsigned int, unsigned int)) &Cell::addInput, py::arg("cell"), py::arg("x0"), py::arg("y0"), py::arg("width") = 0, py::arg("height") = 0)
    .def("initialize", &Cell::initialize)
    .def("save", &Cell::save, py::arg("dirName"))
    .def("load", &Cell::load, py::arg("dirName"))
    .def("saveFreeParameters", &Cell::saveFreeParameters, py::arg("fileName"))
    .def("loadFreeParameters", &Cell::loadFreeParameters, py::arg("fileName"), py::arg("ignoreNotExists") = false)
    .def("exportFreeParameters", &Cell::exportFreeParameters, py::arg("fileName"))
    .def("importFreeParameters", &Cell::importFreeParameters, py::arg("fileName"), py::arg("ignoreNotExists") = false)
    .def("logFreeParameters", &Cell::logFreeParameters, py::arg("fileName"))
    .def("logFreeParametersDistrib", &Cell::logFreeParametersDistrib, py::arg("fileName"))
    .def("discretizeFreeParameters", &Cell::discretizeFreeParameters, py::arg("nbLevels"))
    .def("getFreeParametersRange", &Cell::getFreeParametersRange, py::arg("withAdditiveParameters"))
    .def("getFreeParametersRangePerOutput", &Cell::getFreeParametersRangePerOutput, py::arg("output"), py::arg("withAdditiveParameters"))
    .def("processFreeParameters", &Cell::processFreeParameters, py::arg("func"), py::arg("type"))
    .def("processFreeParametersPerOutput", &Cell::processFreeParametersPerOutput, py::arg("func"), py::arg("output"), py::arg("type"))
    .def("isFullMap", &Cell::isFullMap)
    .def("groupMap", &Cell::groupMap)
    .def("isUnitMap", &Cell::isUnitMap);
}
}
#endif
