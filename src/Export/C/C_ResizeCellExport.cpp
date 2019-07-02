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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Cell/ResizeCell.hpp"
#include "Export/ResizeCellExport.hpp"
#include "Export/C/C_ResizeCellExport.hpp"
#include "utils/Registrar.hpp"



static const N2D2::Registrar<N2D2::ResizeCellExport> registrar(
    "C", N2D2::C_ResizeCellExport::generate);

static const N2D2::Registrar<N2D2::C_CellExport> registrarType(
    N2D2::ResizeCell::Type, N2D2::C_ResizeCellExport::getInstance);




void N2D2::C_ResizeCellExport::generate(ResizeCell& cell, const std::string& dirName) {
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
    header << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n";

    header << "\n";

    if(cell.getMode() == ResizeCell::BilinearTF) {
        generateInterpolationData(cell, header);
    }

    header << "\n\n";    

    C_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::C_ResizeCellExport::generateInterpolationData(const ResizeCell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));


    const float scaleWidth = (cell.isAlignedCorner() && cell.getOutputsWidth() > 1) ? 
                      (cell.getChannelsWidth() - 1) / (float) (cell.getOutputsWidth() - 1)
                    : (cell.getChannelsWidth()) / (float) (cell.getOutputsWidth());

    const float scaleHeight = (cell.isAlignedCorner() && cell.getOutputsHeight() > 1) ? 
                      (cell.getChannelsHeight() - 1) / (float) (cell.getOutputsHeight() - 1)
                    : (cell.getChannelsHeight()) / (float) (cell.getOutputsHeight());


    header << "static const Interpolation " << 
             prefix << "_INTERPOLATION_WIDTH[" << prefix << "_OUTPUTS_WIDTH] = {\n";
    generateInterpolation(cell.getChannelsWidth(), cell.getOutputsWidth(), scaleWidth, header);
    header << "\n};\n";


    header << "static const Interpolation " << 
            prefix << "_INTERPOLATION_HEIGHT[" << prefix << "_OUTPUTS_HEIGHT] = {\n";
    generateInterpolation(cell.getChannelsHeight(), cell.getOutputsHeight(), scaleHeight, header);
    header << "\n};\n";
}

void N2D2::C_ResizeCellExport::generateInterpolation(
                                unsigned int inputsSize, unsigned int outputsSize,
                                float scale, std::ofstream& header) 
{
    for(unsigned int o = 0; o < outputsSize; o++) {
        const float in = o * scale;

        const unsigned int lowIndex = (unsigned int) in;
        const unsigned int highIndex = std::min(lowIndex+ 1, inputsSize - 1);
        const float interpolation = in - lowIndex;
        
        header << "{.lowIndex = " << lowIndex 
               << ", .highIndex = " << highIndex 
               << ", .interpolation = (float) " << interpolation << "}";
        
        if(o < outputsSize-1) {
            header << ", \n";
        }
    }
}





std::unique_ptr<N2D2::C_ResizeCellExport> N2D2::C_ResizeCellExport::getInstance(Cell& /*cell*/) {
    return std::unique_ptr<C_ResizeCellExport>(new C_ResizeCellExport());
}

void N2D2::C_ResizeCellExport::generateCellData(Cell& cell,
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

void N2D2::C_ResizeCellExport::generateCellFunction(Cell& cell,
                            const std::vector<std::shared_ptr<Cell>>& /*parentCells*/,
                            const std::string& inputName,
                            const std::string& outputName,
                            const std::string& /*outputSizeName*/,
                            std::ofstream& prog,
                            bool /*isUnsigned*/,
                            const std::string& /*funcProto*/,
                            const std::string& /*memProto*/,
                            bool /*memCompact*/) 
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    const ResizeCell& resizeCell = dynamic_cast<const ResizeCell&>(cell);
    if(resizeCell.getMode() == ResizeCell::BilinearTF) {
        prog << "    resize_bilinear_tf_propagete("
                << prefix << "_NB_CHANNELS, "
                << prefix << "_CHANNELS_HEIGHT, "
                << prefix << "_CHANNELS_WIDTH, "
                << inputName << ", "
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, "
                << prefix << "_OUTPUTS_WIDTH, "
                << outputName << ", "
                << prefix << "_INTERPOLATION_HEIGHT, "
                << prefix << "_INTERPOLATION_WIDTH"
            << ");\n";
    }
    else if(resizeCell.getMode() == ResizeCell::NearestNeighbor) {
        prog << "    resize_nearest_neighbor_propagete("
                << prefix << "_NB_CHANNELS, "
                << prefix << "_CHANNELS_HEIGHT, "
                << prefix << "_CHANNELS_WIDTH, "
                << inputName << ", "
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, "
                << prefix << "_OUTPUTS_WIDTH, "
                << outputName
            << ");\n";
    }
    else {
        throw std::runtime_error("The resize mode of the cell '" + cell.getName() + 
                                 "' isn't curently supported by the export.");
    }
}

void N2D2::C_ResizeCellExport::generateOutputFunction(Cell& cell,
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