/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils/Helper.hpp"

using namespace N2D2_HELPER;
namespace py = pybind11;

namespace N2D2 {
    void init_helper(py::module &m) {
        py::class_<Options> (m, "Options")
        .def(py::init<>())
        .def_readwrite("seed", &Options::seed)
        .def_readwrite("log", &Options::log)
        .def_readwrite("log_epoch", &Options::logEpoch)
        .def_readwrite("report", &Options::report)
        .def_readwrite("learn", &Options::learn)
        .def_readwrite("learn_epoch", &Options::learnEpoch)
        .def_readwrite("pre_samples", &Options::preSamples)
        .def_readwrite("find_lr", &Options::findLr)
        .def_readwrite("valid_metric", &Options::validMetric)
        .def_readwrite("stop_valid", &Options::stopValid)
        .def_readwrite("test", &Options::test)
        .def_readwrite("test_QAT", &Options::testQAT)
        .def_readwrite("fuse", &Options::fuse)
        .def_readwrite("bench", &Options::bench)
        .def_readwrite("learn_stdp", &Options::learnStdp)
        .def_readwrite("present_time", &Options::presentTime)
        .def_readwrite("avg_window", &Options::avgWindow)
        .def_readwrite("test_index", &Options::testIndex)
        .def_readwrite("test_id", &Options::testId)
        .def_readwrite("test_adv", &Options::testAdv)
        .def_readwrite("check", &Options::check)
        .def_readwrite("log_outputs", &Options::logOutputs)
        .def_readwrite("log_JSON", &Options::logJSON)
        .def_readwrite("log_db_stats", &Options::logDbStats)
        .def_readwrite("log_kernels", &Options::logKernels)
        .def_readwrite("gen_config", &Options::genConfig)
        .def_readwrite("gen_export", &Options::genExport)
        .def_readwrite("nb_bits", &Options::nbBits)
        .def_readwrite("calibration", &Options::calibration)
        .def_readwrite("calibration_reload", &Options::calibrationReload)
        .def_readwrite("wt_clipping_mode", &Options::wtClippingMode)
        .def_readwrite("b_round_mode", &Options::bRoundMode)
        .def_readwrite("c_round_mode", &Options::cRoundMode)
        .def_readwrite("wt_round_mode", &Options::wtRoundMode)
        .def_readwrite("act_clipping_mode", &Options::actClippingMode)
        .def_readwrite("act_scaling_mode", &Options::actScalingMode)
        .def_readwrite("act_rescale_per_output", &Options::actRescalePerOutput)
        .def_readwrite("act_quantile_value", &Options::actQuantileValue)
        .def_readwrite("export_no_unsigned", &Options::exportNoUnsigned)
        .def_readwrite("export_no_cross_layer_equalization", &Options::exportNoCrossLayerEqualization)
        .def_readwrite("time_step", &Options::timeStep)
        .def_readwrite("save_test_set", &Options::saveTestSet)
        .def_readwrite("load", &Options::load)
        .def_readwrite("weights", &Options::weights)
        .def_readwrite("ignore_no_exist", &Options::ignoreNoExist)
        .def_readwrite("ban_multi_device", &Options::banMultiDevice)
        .def_readwrite("export_nb_stimuli_max", &Options::exportNbStimuliMax)
        .def_readwrite("qat_SAT", &Options::qatSAT)
        .def_readwrite("version", &Options::version)
        .def_readwrite("ini_config", &Options::iniConfig)
        ;
        m.def("learn_epoch", &learn_epoch, py::arg("opt"), py::arg("deepNet"));
        m.def("test", &test, py::arg("opt"), py::arg("deepNet"), py::arg("afterCalibration"));
        m.def("setCudaDeviceOption", &setCudaDeviceOption, py::arg("value"));
        m.def("setMultiDevices", &setMultiDevices, py::arg("cudaDev"));
    }
}

#endif
