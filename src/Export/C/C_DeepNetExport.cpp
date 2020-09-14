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

#include "Export/C/C_DeepNetExport.hpp"
#include "DeepNet.hpp"
#include "Export/CellExport.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::DeepNetExport>
N2D2::C_DeepNetExport::mRegistrar("C", N2D2::C_DeepNetExport::generate);

void N2D2::C_DeepNetExport::generate(DeepNet& deepNet,
                                     const std::string& dirName)
{
    Utils::createDirectories(dirName + "/include");
    Utils::createDirectories(dirName + "/src");

    deepNet.fusePadding();  // probably already done, but make sure!
    DeepNetExport::generateCells(deepNet, dirName, "C");

    generateParamsHeader(dirName + "/include/params.h");
    generateEnvironmentHeader(deepNet, dirName + "/include/env.h");
    generateDeepNetHeader(deepNet, "network", dirName + "/include/network.h");

    generateDeepNetProgram(deepNet, "network", dirName + "/src/network.c");
}

void N2D2::C_DeepNetExport::generateParamsHeader(const std::string& fileName)
{
    // Export parameters
    std::ofstream paramsHeader(fileName.c_str());

    if (!paramsHeader.good())
        throw std::runtime_error("Could not create C header file: params.h");

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    paramsHeader << "// N2D2 auto-generated file.\n"
                    "// @ " << std::asctime(localNow)
                 << "\n"; // std::asctime() already appends end of line

    paramsHeader << "#ifndef N2D2_EXPORTC_PARAMS_H\n"
                    "#define N2D2_EXPORTC_PARAMS_H\n\n";

    // Constants
    paramsHeader << "#define NB_BITS " << (int)CellExport::mPrecision << "\n"
        << "#define UNSIGNED_DATA " << DeepNetExport::mUnsignedData << "\n\n";

    paramsHeader << "#endif // N2D2_EXPORTC_PARAMS_H" << std::endl;
}

void N2D2::C_DeepNetExport::generateEnvironmentHeader(DeepNet& deepNet,
                                                      const std::string
                                                      & fileName)
{
    // Environment
    std::ofstream envHeader(fileName.c_str());

    if (!envHeader.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    envHeader << "// N2D2 auto-generated file.\n"
                 "// @ " << std::asctime(localNow)
              << "\n"; // std::asctime() already appends end of line

    envHeader << "#ifndef N2D2_EXPORTC_ENV_LAYER_H\n"
                 "#define N2D2_EXPORTC_ENV_LAYER_H\n\n";

    const std::shared_ptr<StimuliProvider> sp = deepNet.getStimuliProvider();

    // Constants
    envHeader
        << "#define ENV_SIZE_X " << sp->getSizeX() << "\n"
                                                      "#define ENV_SIZE_Y "
        << sp->getSizeY() << "\n"
                             "#define ENV_NB_OUTPUTS " << sp->getNbChannels()
        << "\n\n"
           "#define ENV_DATA_UNSIGNED " << mEnvDataUnsigned
        << "\n\n"
           "#define ENV_OUTPUTS_SIZE (ENV_NB_OUTPUTS*ENV_SIZE_X*ENV_SIZE_Y)\n\n";

    const std::shared_ptr<Cell> cell = deepNet.getTargetCell();

    envHeader << "#define OUTPUTS_HEIGHT " << cell->getOutputsHeight()
              << "\n"
                 "#define OUTPUTS_WIDTH " << cell->getOutputsWidth()
              << "\n"
                 "#define OUTPUTS_SIZE (OUTPUTS_WIDTH*OUTPUTS_HEIGHT)\n"
                 "#define NB_OUTPUTS " << cell->getNbOutputs()
              << "\n"
                 "#define NB_TARGETS "
              << ((cell->getNbOutputs() > 1) ? cell->getNbOutputs() : 2)
              << "\n\n";

    envHeader << "#endif // N2D2_EXPORTC_ENV_LAYER_H" << std::endl;
}

void N2D2::C_DeepNetExport::generateDeepNetHeader(DeepNet& deepNet,
                                                  const std::string& name,
                                                  const std::string& fileName)
{
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C network file: "
                                 + fileName);

    generateHeaderBegin(deepNet, header, fileName);
    generateHeaderIncludes(deepNet, header);
    generateHeaderConstants(deepNet, header);
    generateHeaderFunction(deepNet, name, header);
    generateHeaderEnd(deepNet, header);
}

void N2D2::C_DeepNetExport::generateHeaderBegin(DeepNet& /*deepNet*/,
                                                std::ofstream& header,
                                                const std::string& fileName)
{
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    header << "// N2D2 auto-generated file.\n"
              "// @ " << std::asctime(localNow)
           << "\n"; // std::asctime() already appends end of line

    const std::string guardName
        = Utils::upperCase(Utils::baseName(Utils::fileBaseName(fileName)));

    header << "#ifndef N2D2_EXPORTC_" << guardName << "_H\n"
                                                      "#define N2D2_EXPORTC_"
           << guardName << "_H\n\n";
}

void N2D2::C_DeepNetExport::generateHeaderIncludes(DeepNet& deepNet,
                                                   std::ofstream& header)
{
    // Includes
    header << "#include \"n2d2.h\"\n"
              "#include \"env.h\"\n";

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
            const std::string cellType = cell->getType();

            header << "#include \"" << Utils::CIdentifier(cell->getName())
                << ".h\"\n";
        }
    }
}

void N2D2::C_DeepNetExport::generateHeaderConstants(DeepNet& deepNet,
                                                    std::ofstream& header)
{
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 2,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance(itBegin, it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);

                const std::string prefix
                    = Utils::upperCase(Utils::CIdentifier(
                                            (*parentCells[0]).getName()));

                if (parentCells.size() > 1) {
                    std::stringstream outputDepth;
                    std::stringstream outputName;
                    std::string opPlus = " + ";

                    outputDepth << "(" << prefix << "_NB_OUTPUTS ";
                    outputName << prefix << "_";

                    header << "#define " << prefix << "_OUTPUT_OFFSET 0\n";

                    for (unsigned int i = 1; i < parentCells.size(); ++i) {
                        const std::string prefix_i
                            = Utils::upperCase(Utils::CIdentifier(
                                                (*parentCells[i]).getName()));

                        header << "#define " << prefix_i << "_OUTPUT_OFFSET ";
                        header << outputDepth.str() << ")\n";

                        outputName << prefix_i << "_";
                        outputDepth << opPlus << prefix_i << "_NB_OUTPUTS";
                        (i == parentCells.size() - 1) ? opPlus = " " : opPlus
                            = "+ ";
                    }
                    header << "#define " << outputName.str() << "NB_OUTPUTS ";
                    header << outputDepth.str() << ")\n";
                } else {
                    header << "#define " << prefix << "_OUTPUT_OFFSET 0\n";
                }
            }
            if (itLayer == itLayerEnd - 1) {
                const std::shared_ptr<Cell> cell
                    = deepNet.getCell((*itLayer).at(0));

                header << "#define "
                    << Utils::upperCase(Utils::CIdentifier(cell->getName()))
                    << "_OUTPUT_OFFSET 0\n";
            }
        }
    }
}

void N2D2::C_DeepNetExport::generateHeaderFunction(DeepNet& /*deepNet*/,
                                                   const std::string& name,
                                                   std::ofstream& header)
{
    header << "extern DATA_T "
            "output_data[NB_OUTPUTS*OUTPUTS_HEIGHT*OUTPUTS_WIDTH]; \n";

    header << "\n"
              "void " << name
           << "(DATA_T in_data[ENV_NB_OUTPUTS][ENV_SIZE_Y][ENV_SIZE_X],"
              " uint32_t out_data[OUTPUTS_HEIGHT][OUTPUTS_WIDTH]);\n";
}

void N2D2::C_DeepNetExport::generateHeaderEnd(DeepNet& /*deepNet*/,
                                              std::ofstream& header)
{
    header << "\n"
              "#endif" << std::endl;
    header.close();
}

void N2D2::C_DeepNetExport::generateDeepNetProgram(DeepNet& deepNet,
                                                   const std::string& name,
                                                   const std::string& fileName)
{
    std::ofstream prog(fileName.c_str());

    if (!prog.good())
        throw std::runtime_error("Could not create C network file: "
                                 + fileName);

    generateProgramBegin(deepNet, prog);
    generateProgramData(deepNet, prog);
    generateProgramFunction(deepNet, name, prog);
}

void N2D2::C_DeepNetExport::generateProgramBegin(DeepNet& /*deepNet*/,
                                                 std::ofstream& prog)
{
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    // Program
    prog << "// N2D2 auto-generated file.\n"
            "// @ " << std::asctime(localNow)
         << "\n" // std::asctime() already appends end of line
            "#include \"network.h\"\n"
            "\n";
}

void N2D2::C_DeepNetExport::generateProgramData(DeepNet& deepNet,
                                                std::ofstream& prog)
{
    prog << "//#define TIME_ANALYSIS\n"
            "//#define DATA_DYN_ANALYSIS\n"
            "//#define ACC_DYN_ANALYSIS\n"
            "#define ACC_DYN_REPORT CHW\n"
            "\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 2,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            std::stringstream outputName;

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance(itBegin, it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);
                // const std::shared_ptr<Cell> cell =
                // deepNet.getCell((*itLayer).at(std::distance((*itLayer).begin(),
                // it)));
                std::stringstream outputName;
                outputName << (*parentCells[0]).getName() << "_";

                for (unsigned int i = 1; i < parentCells.size(); ++i)
                    outputName << (*parentCells[i]).getName() << "_";

                const std::string identifier = Utils::CIdentifier(
                                                        outputName.str());

                C_CellExport::getInstance(*parentCells[0])->generateCellData(
                    *parentCells[0],
                    identifier + "data",
                    Utils::upperCase(identifier) + "NB_OUTPUTS",
                    prog);
            }
        }
    }
    prog << "DATA_T "
            "output_data[NB_OUTPUTS*OUTPUTS_HEIGHT*OUTPUTS_WIDTH]; \n";
    prog << "static DATA_T "
            "output_spatial_data[NB_OUTPUTS][OUTPUTS_HEIGHT][OUTPUTS_WIDTH]; "
            "\n";
}

void N2D2::C_DeepNetExport::generateProgramFunction(DeepNet& deepNet,
                                                    const std::string& name,
                                                    std::ofstream& prog)
{
    prog << "\n"
            "void " << name
         << "(DATA_T in_data[ENV_NB_OUTPUTS][ENV_SIZE_Y][ENV_SIZE_X],"
            " uint32_t out_data[OUTPUTS_HEIGHT][OUTPUTS_WIDTH]) {\n"
            "#ifdef SAVE_OUTPUTS\n"
            "    convcell_outputs_save(\"in_data.txt\", ENV_NB_OUTPUTS, ENV_SIZE_Y, ENV_SIZE_X, in_data);\n"
            "#endif\n"
            "\n"
            "#ifdef TIME_ANALYSIS\n"
            "    struct timeval start, end;\n"
            "#endif\n";

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
                                       isCellInputsUnsigned(*cell));
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
