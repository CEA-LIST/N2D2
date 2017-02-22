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

#ifndef N2D2_CPP_CUDNN_CELLEXPORT_H
#define N2D2_CPP_CUDNN_CELLEXPORT_H

#include "Export/CPP/CPP_CellExport.hpp"
#include "Cell/Cell.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_BatchNormCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_ConvCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_FMPCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_FcCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_PoolCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrarType@CPP_cuDNN_SoftmaxCellExport@N2D2@@0U?$Registrar@VCPP_cuDNN_CellExport@N2D2@@@2@A")
#endif

namespace N2D2 {
/**
 * Virtual base class for methods commun to every cell type for the CPP_cuDNN
 * export
 * ANY CELL, CPP_cuDNN EXPORT
*/
class CPP_cuDNN_CellExport {
public:
    typedef std::function
        <std::unique_ptr<CPP_cuDNN_CellExport>(Cell&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static void generateHeaderIncludes(Cell& cell, std::ofstream& header);

    inline static std::unique_ptr<CPP_cuDNN_CellExport> getInstance(Cell& cell);

    // Commun methods for all cells

    virtual void generateCellProgramDesc(Cell& cell, std::ofstream& prog) = 0;
    virtual void generateCellProgramGlobalDefinition(Cell& cell,
                                                     std::ofstream& prog) = 0;
    virtual void generateCellBuffer(const std::string& bufferName,
                                    std::ofstream& prog) = 0;

    virtual void generateCellProgramInitNetwork(Cell& cell,
                                       std::vector<std::string>& parentsName,
                                       std::ofstream& prog) = 0;

    virtual void generateCellProgramInitBuffer(Cell& cell,
                                               const std::string& bufferName,
                                               std::ofstream& prog) = 0;

    virtual void generateCellProgramFunction(Cell& cell,
                                             const std::string& inputName,
                                             const std::string& outputName,
                                             const std::string& output_pos,
                                             std::ofstream& prog,
                                             const std::string& funcProto = "")
        = 0;
    virtual void generateCellProgramOutputFunction(Cell& cell,
                                                   const std::string
                                                   & outputDataName,
                                                   const std::string
                                                   & outputName,
                                                   std::ofstream& prog) = 0;
    virtual void generateCellProgramFree(Cell& cell,
                                         std::vector<std::string>& parentsName,
                                         std::ofstream& prog) = 0;
};
}

std::unique_ptr<N2D2::CPP_cuDNN_CellExport>
N2D2::CPP_cuDNN_CellExport::getInstance(Cell& cell)
{
    return Registrar<CPP_cuDNN_CellExport>::create(cell.getType())(cell);
}

#endif // N2D2_CPP_CUDNN_CELLEXPORT_H
