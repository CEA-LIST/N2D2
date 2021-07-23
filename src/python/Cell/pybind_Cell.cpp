/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Cyril MOINEAU (cyril.moineau@cea.fr)

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

#ifdef PYBIND
#include "DeepNet.hpp"
#include "HeteroStimuliProvider.hpp"
#include "Cell/Cell.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_Cell(py::module &m) {


    py::class_<Cell, std::shared_ptr<Cell>, Parameterizable> cell(m, "Cell", py::multiple_inheritance());

    py::enum_<Cell::FreeParametersType>(cell, "FreeParametersType")
    .value("Additive", Cell::FreeParametersType::Additive)
    .value("Multiplicative", Cell::FreeParametersType::Multiplicative)
    .value("All", Cell::FreeParametersType::All)
    .export_values();

    cell.doc() = "Cell is the base object for any kind of layer composing a deep network. It provides the base interface required.";
    
    
    // TODO : Find a better method to add description to overloaded method
    // As mentionned here https://github.com/pybind/pybind11/issues/2619 pybind + shpinx have trouble generating docstring for overloaded method.
    // The current best fix is to disable function signatures, this seems to be currently acceptable. 

    py::options options;
    options.disable_function_signatures();
    cell.def("addInput", (void (Cell::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const Tensor<bool>&)) &Cell::addInput, py::arg("sp"), py::arg("channel"), py::arg("x0"), py::arg("y0"), py::arg("width"), py::arg("height"), py::arg("mapping") = Tensor<bool>(), 
    R"mydelimiter(
     Connect an input filter from the environment to the cell

     :param sp: N2D2 StimuliProvider object reference
     :type sp: :py:class:`N2D2.StimuliProvider`
     :param channel: Channel number in the environment
     :type channel: int
     :param x0: Left offset
     :type x0: int
     :param y0: Top offset
     :type y0: int
     :param width: Width
     :type width: int
     :param height: Height
     :type height: int
     :param mapping: Connection between the environment map and the cell output maps (size of the vector = number of output maps in the cell)
     :type mapping: :py:class:`N2D2.Tensor_bool`, optional
     )mydelimiter")

    .def("addInput", (void (Cell::*)(StimuliProvider&, unsigned int, unsigned int, unsigned int, unsigned int, const Tensor<bool>&)) &Cell::addInput, py::arg("sp"), py::arg("x0") = 0, py::arg("y0") = 0, py::arg("width") = 0, py::arg("height") = 0, py::arg("mapping") = Tensor<bool>(),
    R"mydelimiter(
     Connect all the input maps from the environment to the cell

     :param sp: N2D2 StimuliProvider object reference
     :type sp: :py:class:`N2D2.StimuliProvider`
     :param x0: Left offset
     :type x0: int, optional
     :param y0: Top offset
     :type y0: int, optional
     :param width: Width
     :type width: int, optional
     :param height: Height
     :type height: int, optional
     :param mapping: Connection between the environment map filters and the cell output maps (size of the matrix = number of output maps in the cell [cols] x total number of filters in the environment [rows])
     :type mapping: :py:class:`N2D2.Tensor_bool`, optional
     )mydelimiter")
    
    .def("addInput", (void (Cell::*)(Cell*, const Tensor<bool>&)) &Cell::addInput, py::arg("cell"), py::arg("mapping") = Tensor<bool>(),
    R"mydelimiter(
     Connect an input cell to the cell
     
     :param cell: Pointer to the input cell
     :type cell: :py:class:`N2D2.Cell`
     :param mapping: Connection between the input cell output maps (input channels) and the cell output maps (size of the matrix = number of output maps in the cell [cols] x number of input cell output maps (input channels) [rows])
     :type mapping: :py:class:`N2D2.Tensor_bool`, optional
     )mydelimiter")

    .def("addInput", (void (Cell::*)(Cell*, unsigned int, unsigned int, unsigned int, unsigned int)) &Cell::addInput, py::arg("cell"), py::arg("x0"), py::arg("y0"), py::arg("width") = 0, py::arg("height") = 0,
    R"mydelimiter(
     Connect an input cell to the cell
     
     :param cell: Pointer to the input cell
     :param x0: Left offset
     :type x0: int
     :param y0: Top offset
     :type y0: int
     :param width: Width
     :type width: int, optional
     :param height: Height
     :type height: int, optional
     )mydelimiter")
     
    .def("clearInputs", &Cell::clearInputs,
    R"mydelimiter(
     Clear input Cells
     )mydelimiter");
    options.enable_function_signatures();

    cell
    .def("addMultiscaleInput", &Cell::addMultiscaleInput, py::arg("sp"), py::arg("x0") = 0, py::arg("y0") = 0, py::arg("width") = 0, py::arg("height") = 0, py::arg("mapping") = Tensor<bool>(),
    R"mydelimiter(
     Connect all the input maps from the environment to the cell
     
     :param sp: N2D2 StimuliProvider object reference
     :type sp: :py:class:`N2D2.StimuliProvider`
     :param x0:  Left offset
     :type x0: int, optional
     :param y0: Top offset
     :type y0: int, optional
     :param width: Width
     :type width: int, optional
     :param height: Height
     :type height: int, optional
     :param mapping: Connection between the environment map filters and the cell output maps (size of the matrix = number of output maps in the cell [cols] x total number of filters in the environment [rows])    
     :type mapping: :py:class:`N2D2.Tensor_bool`, optional

     )mydelimiter")

    .def("initialize", &Cell::initialize,    
    R"mydelimiter(
     Initialize the state of the cell (e.g. weights random initialization)
    )mydelimiter")

    .def("setMapping", &Cell::setMapping, py::arg("mapping"))
    .def("initializeDataDependent", &Cell::initializeDataDependent)

    .def("save", &Cell::save, py::arg("dirName"),    
    R"mydelimiter(
     Save cell configuration and free parameters to a directory
     
     :param dirName: Destination directory
     :type dirName: str
     )mydelimiter")
    .def("load", &Cell::load, py::arg("dirName"),    
    R"mydelimiter(
     Load cell configuration and free parameters from a directory
     
     :param dirName: Source directory
     :type dirName: str
     )mydelimiter")

    .def("exportFreeParameters", &Cell::exportFreeParameters, py::arg("fileName"),    
    R"mydelimiter(
     Export cell free parameters to a file, in ASCII format compatible between
     the different cell models
     
     :param fileName: Destination file
     :type fileName: str
     )mydelimiter")
    .def("importFreeParameters", &Cell::importFreeParameters, py::arg("fileName"), py::arg("ignoreNotExists") = false,
     R"mydelimiter(
     Load cell free parameters from a file, in ASCII format compatible between
     the different cell models

     :param fileName: Source file
     :type fileName: str
     :param ignoreNotExists: If true, don't throw an error if the file doesn't exist
     :type ignoreNotExists: bool, optional
     )mydelimiter")
     .def("importActivationParameters", &Cell::importActivationParameters, py::arg("fileName"), py::arg("ignoreNotExists") = false,
     R"mydelimiter(
     Load activation parameters from a file

     :param fileName: Source file
     :type fileName: str
     :param ignoreNotExists: If true, don't throw an error if the file doesn't exist
     :type ignoreNotExists: bool, optional
     )mydelimiter")
    // .def("logFreeParameters", &Cell::logFreeParameters, py::arg("fileName"))
    // .def("logFreeParametersDistrib", &Cell::logFreeParametersDistrib, py::arg("fileName"), py::arg("type"))
    // .def("discretizeFreeParameters", &Cell::discretizeFreeParameters, py::arg("nbLevels"))
    // .def("getFreeParametersRange", &Cell::getFreeParametersRange, py::arg("withAdditiveParameters"))
    // .def("getFreeParametersRangePerOutput", &Cell::getFreeParametersRangePerOutput, py::arg("output"), py::arg("withAdditiveParameters"))
    // .def("processFreeParameters", &Cell::processFreeParameters, py::arg("func"), py::arg("type"))
    // .def("processFreeParametersPerOutput", &Cell::processFreeParametersPerOutput, py::arg("func"), py::arg("output"), py::arg("type"))
    // .def("isFullMap", &Cell::isFullMap)
    // .def("groupMap", &Cell::groupMap)
    // .def("isUnitMap", &Cell::isUnitMap) 
    .def("saveFreeParameters", &Cell::saveFreeParameters, py::arg("fileName"))
    // .def("loadFreeParameters", &Cell::loadFreeParameters, py::arg("fileName"), py::arg("ignoreNotExists") = false)
    // .def("getId", &Cell::getId)
    .def("getName", &Cell::getName,
      R"mydelimiter(
     Get the cell name

     )mydelimiter")
     .def("getType", &Cell::getType,
      R"mydelimiter(
     Get basic cell type

     )mydelimiter")
    .def("getNbChannels", &Cell::getNbChannels)
    // .def("getChannelsWidth", &Cell::getChannelsWidth)
    // .def("getChannelsHeight", &Cell::getChannelsHeight)
    // .def("getInputsDim", &Cell::getInputsDim, py::arg("dim"))
    .def("getInputsDims", &Cell::getInputsDims)
    .def("getInputsSize", &Cell::getInputsSize)
    .def("getNbOutputs", &Cell::getNbOutputs,
    R"mydelimiter(
     Returns number of output maps in the cell (or number of outputs for 1D cells)

     )mydelimiter")
    .def("getOutputsWidth", &Cell::getOutputsWidth,
    R"mydelimiter(
     Returns cell output maps width (returns 1 for 1D cells)

     )mydelimiter")
    .def("getOutputsHeight", &Cell::getOutputsHeight,
    R"mydelimiter(
     Returns cell output maps height (returns 1 for 1D cells)

     )mydelimiter")
    // .def("getOutputsDim", &Cell::getOutputsDim, py::arg("dim"))
    // .def("getOutputsDims", &Cell::getOutputsDims)
    // .def("getOutputsSize", &Cell::getOutputsSize)
    // .def("getStats", &Cell::getStats)
    // .def("getReceptiveField", &Cell::getReceptiveField, py::arg("outputField") = std::vector<unsigned int>())
    .def("getAssociatedDeepNet", &Cell::getAssociatedDeepNet)
    .def("getChildrenCells", &Cell::getChildrenCells)
    .def("getParentsCells", &Cell::getParentsCells)
    // .def("isConnection", &Cell::isConnection, py::arg("channel"), py::arg("output"))
    .def("getMapping", &Cell::getMapping)
    ;
}
}
#endif

