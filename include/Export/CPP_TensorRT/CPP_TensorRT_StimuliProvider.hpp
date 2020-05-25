/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_CPP_TENSORRT_STIMULIPROVIDEREXPORT_H
#define N2D2_CPP_TENSORRT_STIMULIPROVIDEREXPORT_H

#include "CPP_TensorRT_CellExport.hpp"
#include "DeepNet.hpp"
#include "CPP_TensorRT_DeepNetExport.hpp"
#include "StimuliProvider.hpp"

namespace N2D2 {
class CPP_TensorRT_StimuliProvider {
public:
    static void generateCalibFiles(StimuliProvider& sp,
                                    const std::string& dirName,
                                    Database::StimuliSet set,
                                    DeepNet* deepNet = NULL);
};
}

#endif // N2D2_CPP_TENSORRT_STIMULIPROVIDEREXPORT_H

