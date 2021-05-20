/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Cell/ElemWiseCell.hpp"
#include "Export/ElemWiseCellExport.hpp"
#include "Export/C/C_ElemWiseCellExport.hpp"
#include "utils/Registrar.hpp"



static const N2D2::Registrar<N2D2::ElemWiseCellExport> registrar(
    "C", N2D2::C_ElemWiseCellExport::generate);

static const N2D2::Registrar<N2D2::C_CellExport> registrarType(
    N2D2::ElemWiseCell::Type, N2D2::C_ElemWiseCellExport::getInstance);




void N2D2::C_ElemWiseCellExport::generate(const ElemWiseCell& cell, const std::string& dirName) {
    Utils::createDirectories(dirName + "/include");

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const std::string fileName = dirName + "/include/" + identifier + ".h";
    std::ofstream header(fileName);

    C_CellExport::generateHeaderBegin(cell, header);
    C_CellExport::generateHeaderIncludes(cell, header);

    header << "\n";
    
    header << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n";
    header << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n";
    header << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n";

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n";
    header << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n";
    header << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n\n";

    C_CellExport::generateActivation(cell, header);
    C_CellExport::generateActivationScaling(cell, header);

    C_CellExport::generateHeaderEnd(cell, header);
}

std::unique_ptr<N2D2::C_ElemWiseCellExport> N2D2::C_ElemWiseCellExport::getInstance(Cell& /*cell*/) {
    return std::unique_ptr<C_ElemWiseCellExport>(new C_ElemWiseCellExport());
}

void N2D2::C_ElemWiseCellExport::generateCellData(Cell& cell,
                        const std::string& outputName,
                        const std::string& /*outputSizeName*/,
                        std::ofstream& prog) 
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    
    prog << "static DATA_T " << outputName 
             << "[" << prefix << "_NB_OUTPUTS]"
             << "[" << prefix << "_OUTPUTS_HEIGHT]"
             << "[" << prefix << "_OUTPUTS_WIDTH];\n";
}

void N2D2::C_ElemWiseCellExport::generateCellFunction(Cell& cell,
                            const std::vector<std::shared_ptr<Cell>>& parentCells,
                            const std::string& inputName,
                            const std::string& outputName,
                            const std::string& /*outputSizeName*/,
                            std::ofstream& prog,
                            bool isUnsigned,
                            const std::string& /*funcProto*/,
                            const std::string& /*memProto*/,
                            bool /*memCompact*/) 
{
    if(parentCells.size() > 2) {
        throw std::runtime_error("ElemWiseCell C Export only support 2 inputs parents");
    }

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    prog << "    " << "elemwise_" << ((isUnsigned) ? "u" : "") 
            << "propagate("
            << prefix << "_CHANNELS_HEIGHT, "
            << prefix << "_CHANNELS_WIDTH, "
            << prefix << "_NB_OUTPUTS, "
            << inputName << ", "
            << inputName << ", "
            << outputName << ", "
            << prefix << "_ACTIVATION, "
            << prefix << "_SHIFT);\n";

    // Save outputs
    prog << "#ifdef SAVE_OUTPUTS\n"
         << "    elemwisecell_outputs_save("
            << "\"" << identifier << ".txt\", "
            << prefix << "_NB_OUTPUTS, "
            << prefix << "_OUTPUTS_HEIGHT, "
            << prefix << "_OUTPUTS_WIDTH, "
            << outputName
         << ");\n"
         << "#endif\n";
}

void N2D2::C_ElemWiseCellExport::generateOutputFunction(Cell& cell,
                            const std::string& inputName,
                            const std::string& outputName,
                            std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    if ((cell.getOutputsWidth() == 1) && (cell.getOutputsHeight() == 1)) {
        prog << "    output_max(" 
                << prefix << "_NB_OUTPUTS, " 
                << inputName << ", " 
                << outputName 
            << ");\n";
    } else {
        prog << "    spatial_output_max(" 
                 << prefix << "_NB_OUTPUTS, " 
                 << prefix << "_OUTPUTS_HEIGHT, " 
                 << prefix << "_OUTPUTS_WIDTH, " 
                 << inputName << ", " 
                 << outputName 
             << ");\n";
    }

}
