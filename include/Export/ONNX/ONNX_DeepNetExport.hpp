/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_ONNX_DEEPNETEXPORT_H
#define N2D2_ONNX_DEEPNETEXPORT_H

#ifdef ONNX
#include "Export/DeepNetExport.hpp"

#include <onnx.pb.h>

namespace N2D2 {
/**
 * Class for methods of DeepNet for all ONNX exports types
 * DeepNetExport, ONNX_EXPORT
**/
class ONNX_DeepNetExport : public DeepNetExport {
public:
    static void generate(DeepNet& deepNet, const std::string& dirName);
    static void saveModel(DeepNet& deepNet, const std::string& fileName);
    static onnx::ModelProto generateModel(DeepNet& deepNet);

    static void addBranchesCells(DeepNet& deepNet);

private:
    static std::string getCellModelType(const Cell& cell);
    static void setTensorProto(onnx::ValueInfoProto* info,
                               const BaseTensor& tensor);

    static Registrar<DeepNetExport> mRegistrar;
};
}

#endif

#endif // N2D2_ONNX_DEEPNETEXPORT_H
