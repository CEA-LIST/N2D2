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

#include "Export/C_HLS/C_HLS_DeepNetExport.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::DeepNetExport>
N2D2::C_HLS_DeepNetExport::mRegistrar("C_HLS",
                                      N2D2::C_HLS_DeepNetExport::generate);

void N2D2::C_HLS_DeepNetExport::generate(DeepNet& deepNet,
                                         const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");
    Utils::createDirectories(dirName + "/src");

    deepNet.fusePadding();  // probably already done, but make sure!
    DeepNetExport::generateCells(deepNet, dirName, "C_HLS");

    C_DeepNetExport::generateParamsHeader(dirName + "/include/params.h");
    C_DeepNetExport::generateEnvironmentHeader(deepNet,
                                               dirName + "/include/env.h");
    generateDeepNetHeader(deepNet, "network", dirName + "/include/network.h");

    generateDeepNetProgram(deepNet, "network", dirName + "/src/network.c");
    generateTcl(deepNet, dirName);
}

void N2D2::C_HLS_DeepNetExport::generateDeepNetHeader(DeepNet& deepNet,
                                                      const std::string& name,
                                                      const std::string
                                                      & fileName)
{
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C network file: "
                                 + fileName);

    C_DeepNetExport::generateHeaderBegin(deepNet, header, fileName);
    generateHeaderIncludes(deepNet, header);
    C_DeepNetExport::generateHeaderConstants(deepNet, header);
    C_DeepNetExport::generateHeaderFunction(deepNet, name, header);
    C_DeepNetExport::generateHeaderEnd(deepNet, header);
}

void N2D2::C_HLS_DeepNetExport::generateHeaderIncludes(DeepNet& deepNet,
                                                       std::ofstream& header)
{
    header << "#include \"n2d2_hls.h\"\n";

    C_DeepNetExport::generateHeaderIncludes(deepNet, header);
}

void N2D2::C_HLS_DeepNetExport::generateDeepNetProgram(DeepNet& deepNet,
                                                       const std::string& name,
                                                       const std::string
                                                       & fileName)
{
    std::ofstream prog(fileName.c_str());

    if (!prog.good())
        throw std::runtime_error("Could not create C network file: "
                                 + fileName);

    C_DeepNetExport::generateProgramBegin(deepNet, prog);
    C_DeepNetExport::generateProgramData(deepNet, prog);
    generateProgramPrototypes(deepNet, prog);
    generateProgramFunction(deepNet, name, prog);
    // C_DeepNetExport::generateProgramFunction(deepNet, name, prog);
}

void N2D2::C_HLS_DeepNetExport::generateProgramPrototypes(DeepNet& deepNet,
                                                          std::ofstream& prog)
{
    std::string inputsBuffer = "in_";
    std::string outputsBuffer = "output_";
    std::string input_buff;
    std::string output_buff;
    std::string output_size;

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        prog << "/************************************LAYER ("
             << std::distance(layers.begin(), itLayer) << ")***/\n";
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            input_buff
                = (itLayer == itLayerBegin)
                      ? inputsBuffer
                      : getCellInputName(deepNet,
                                         std::distance(layers.begin(), itLayer),
                                         std::distance(itBegin, it));
            output_buff = (itLayer >= itLayerEnd - 1)
                              ? outputsBuffer
                              : getCellOutputName(
                                    deepNet,
                                    std::distance(layers.begin(), itLayer),
                                    std::distance(itBegin, it));
            output_size = (itLayer >= itLayerEnd - 1)
                              ? "OUTPUTS_SIZE*NB_OUTPUTS"
                              : output_buff + "NB_OUTPUTS";

            C_HLS_CellExport::getInstance(*cell)
                ->generateCellPrototype(*cell,
                                        deepNet.getParentCells(cell->getName()),
                                        Utils::upperCase(output_size),
                                        prog,
                                        isCellInputsUnsigned(*cell, deepNet));
        }
    }
}

void N2D2::C_HLS_DeepNetExport::generateProgramFunction(DeepNet& deepNet,
                                                        const std::string& name,
                                                        std::ofstream& prog)
{
    prog << "\n"
            "void " << name
         << "(DATA_T in_data[ENV_NB_OUTPUTS][ENV_SIZE_Y][ENV_SIZE_X],"
            " uint32_t out_data[OUTPUTS_HEIGHT][OUTPUTS_WIDTH]) {\n";

    std::string inputsBuffer = "in_";
    std::string outputsBuffer = "output_";
    std::string input_buff;
    std::string output_buff;
    std::string output_size;

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        prog << "/************************************LAYER ("
             << std::distance(layers.begin(), itLayer) << ")***/\n";
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            input_buff
                = (itLayer == itLayerBegin)
                      ? inputsBuffer
                      : getCellInputName(deepNet,
                                         std::distance(layers.begin(), itLayer),
                                         std::distance(itBegin, it));
            bool isSpatial = ( ((*cell).getOutputsWidth() > 1) ||
                                ((*cell).getOutputsHeight() > 1)) ? true
                                : false;

            output_buff = (itLayer >= itLayerEnd - 1)
                              ? (isSpatial ? outputsBuffer + "spatial_"
                                 : outputsBuffer)
                              : getCellOutputName(
                                    deepNet,
                                    std::distance(layers.begin(), itLayer),
                                    std::distance(itBegin, it));
            output_size = (itLayer >= itLayerEnd - 1)
                              ? "OUTPUTS_SIZE*NB_OUTPUTS"
                              : output_buff + "NB_OUTPUTS";

            C_CellExport::getInstance(*cell)
                ->generateCellFunction(*cell,
                                       deepNet.getParentCells(cell->getName()),
                                       input_buff + "data",
                                       output_buff + "data",
                                       Utils::upperCase(output_size),
                                       prog,
                                       isCellInputsUnsigned(*cell, deepNet),
                                       Utils::CIdentifier(cell->getName()),
                                       "",
                                       !cell->isFullMap());
        }
        if (itLayer == itLayerEnd - 1) {
            const std::shared_ptr<Cell> cell
                = deepNet.getCell((*itLayer).at(0));
            C_CellExport::getInstance(*cell)->generateOutputFunction(
                *cell, output_buff + "data", "out_data", prog);
        }
    }
    prog << "}\n";
}

void N2D2::C_HLS_DeepNetExport::generateTcl(DeepNet& deepNet,
                                            const std::string& dirName)
{
    // Default solution script
    std::ofstream vivadoTcl((dirName + "/run_hls.tcl").c_str());

    if (!vivadoTcl.good())
        throw std::runtime_error(
            "Could not create Vivado script file: run_hls.tcl");

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    vivadoTcl << "# N2D2 auto-generated file.\n"
                 "# @ " << std::asctime(localNow)
              << "\n"; // std::asctime() already appends end of line

    vivadoTcl << "open_project -reset hls\n"
                 "set_top network\n"
                 "add_files include/utils.h -cflags \"-std=c99\"\n"
                 "add_files include/network.h -cflags \"-std=c99\"\n"
                 "add_files src/network.c -cflags \"-std=c99 -I./include/\"\n"
                 "add_files include/n2d2_hls.h -cflags \"-std=c99\"\n"
                 "add_files include/n2d2.h -cflags \"-std=c99\"\n"
                 "add_files src/n2d2.c -cflags \"-std=c99 -I./include/\"\n"
                 "add_files include/env.h -cflags \"-std=c99\"\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            vivadoTcl << "add_files include/"
                      << Utils::CIdentifier(cell->getName())
                      << ".h -cflags \"-std=c99\"\n";
        }
    }

    vivadoTcl << "add_files -tb n2d2_test.c -cflags \"-std=c99 -I./include/\"\n"
                 "add_files -tb stimuli\n\n";

    vivadoTcl << "set cells [dict create]\n";

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);
            const std::vector<std::shared_ptr<Cell> > parentCells
                = deepNet.getParentCells(cell.getName());
            const C_HLS_CellExport::TclDirectives directives
                = C_HLS_CellExport::getInstance(cell)
                      ->getTclDirectives(cell,
                                         parentCells,
                                         isCellInputsUnsigned(cell, deepNet));

            vivadoTcl << "dict set cells " << directives.funcName << " type "
                      << directives.typeName << "\n"
                      << "dict set cells " << directives.funcName
                      << " channel_dims {" << directives.nbChannels << " "
                      << directives.channelsWidth << " "
                      << directives.channelsHeight << "}\n"
                      << "dict set cells " << directives.funcName
                      << " output_dims {" << cell.getNbOutputs() << " "
                      << cell.getOutputsWidth() << " "
                      << cell.getOutputsHeight() << "}\n"
                      << "dict set cells " << directives.funcName
                      << " kernel_dims {" << directives.kernelWidth << " "
                      << directives.kernelHeight << "}\n";
        }
    }

    vivadoTcl << "\n"
                 "source \"./solutions.tcl\"\n"
                 "exit\n";
}
