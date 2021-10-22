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

#ifndef N2D2_ONNX_BATCHNORMCELLEXPORT_H
#define N2D2_ONNX_BATCHNORMCELLEXPORT_H

#ifdef ONNX

#include "Export/BatchNormCellExport.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/ONNX/ONNX_CellExport.hpp"

#include <onnx.pb.h>

namespace N2D2 {
/**
 * Class for methods of BatchNormCell for all ONNX exports type
 * BatchNormCell, ONNX_EXPORT
**/

class ONNX_BatchNormCellExport : public BatchNormCellExport, public ONNX_CellExport {
public:
    static std::unique_ptr<ONNX_BatchNormCellExport> getInstance(Cell& cell);
    void generateNode(onnx::GraphProto* graph,
                                 const DeepNet& deepNet,
                                 const Cell& cell);

private:
    static Registrar<BatchNormCellExport> mRegistrar;
    static Registrar<ONNX_CellExport> mRegistrarType;
};
}

#endif

#endif // N2D2_ONNX_BATCHNORMCELLEXPORT_H
