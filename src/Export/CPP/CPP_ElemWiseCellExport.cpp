
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "DeepNet.hpp"
#include "Export/CPP/CPP_ElemWiseCellExport.hpp"

N2D2::Registrar<N2D2::ElemWiseCellExport>
N2D2::CPP_ElemWiseCellExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_ElemWiseCellExport::generate);

void N2D2::CPP_ElemWiseCellExport::generate(ElemWiseCell& cell,
                                             const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ElemWiseCellExport::generateHeaderConstants(ElemWiseCell& cell,
                                                            std::ofstream
                                                            & header)
{

    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));

    const auto parentsCells = cell.getParentsCells();
    header << "#define " << prefix << "_NB_INPUTS " << parentsCells.size() << "\n";

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n"
              "#define " << prefix << "_OUTPUTS_WIDTH "
           << cell.getOutputsWidth() << "\n"
              "#define " << prefix
           << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
                                                               "#define "
           << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth()
           << "\n"
              "#define " << prefix << "_CHANNELS_HEIGHT "
           << cell.getChannelsHeight() << "\n\n";

    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n\n";

    const std::vector<Float_T> weights = cell.getWeights();
    header << "static WDATA_T  " << prefix << "_WEIGHTS[";

    header << weights.size();

    header << "] = {";
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        //header << weights[i];
        CellExport::generateFreeParameter(weights[i], header);
        if(i < weights.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";


    const std::vector<Float_T> shifts = cell.getShifts();
    header << "static WDATA_T  " << prefix << "_SHIFTS[";

    header << shifts.size();

    header << "] = {";
    for(unsigned int i = 0; i < shifts.size(); ++i)
    {
        CellExport::generateFreeParameter(shifts[i], header);
        if(i < shifts.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";
    header <<  "static WDATA_T  " << prefix << "_POWER[";

    header << shifts.size();

    header << "] = {";
    for(unsigned int i = 0; i < shifts.size(); ++i)
    {
        header << "1";

        if(i < shifts.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";

    const ElemWiseCell::Operation elemOp = cell.getOperation();

    header << "#define " << prefix << "_ELEM_OP " << elemOp << "\n"
           << std::endl;


}

std::unique_ptr<N2D2::CPP_ElemWiseCellExport>
N2D2::CPP_ElemWiseCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr<CPP_ElemWiseCellExport>(new CPP_ElemWiseCellExport);
}

void N2D2::CPP_ElemWiseCellExport::generateCallCode(
    const DeepNet& deepNet,
    const Cell& cell, 
    std::stringstream& includes,
    std::stringstream& /*buffers*/, 
    std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    // includes
    includes << "#include \"" << identifier << ".hpp\"\n";

    // functionCalls: input/output buffers
    const std::string inputType = DeepNetExport::isCellInputsUnsigned(cell)
        ? "UDATA_T" : "DATA_T";

    const auto& parents = deepNet.getParentCells(cell.getName());

    std::ostringstream inputBufferStr;

    for (const auto& parentCell: parents) {
        inputBufferStr << ", " << Utils::CIdentifier(parentCell->getName()
                                                        + "_output");
    }

    const std::string inputBuffer = inputBufferStr.str();
    const std::string outputBuffer = Utils::CIdentifier(cell.getName() + "_output");

    generateBenchmarkStart(deepNet, cell, functionCalls);

    // functionCalls: propagate
    // elemWisePropagate is a variadic template
    functionCalls << "    elemWisePropagate<" 
                        << prefix << "_NB_INPUTS, "
                        << prefix << "_CHANNELS_HEIGHT, "
                        << prefix << "_CHANNELS_WIDTH, "
                        << prefix << "_NB_OUTPUTS, "
                        << prefix << "_OUTPUTS_HEIGHT, " 
                        << prefix << "_OUTPUTS_WIDTH, "
                        << prefix << "_ELEM_OP, "
                        << prefix << "_ACTIVATION, "
                        // Memory mapping: output
                        << prefix << "_MEM_CONT_OFFSET, "
                        << prefix << "_MEM_CONT_SIZE, "
                        << prefix << "_MEM_WRAP_OFFSET, " 
                        << prefix << "_MEM_WRAP_SIZE, " 
                        << prefix << "_MEM_STRIDE";

    for (const auto& parentCell: parents) {
        const std::string parentIdentifier
            = Utils::CIdentifier((parentCell) ? parentCell->getName() : "env");
        const std::string parentPrefix
            = N2D2::Utils::upperCase(parentIdentifier);

        // Memory mapping: inputs
        functionCalls << ", "
            << parentPrefix << "_NB_OUTPUTS, "
            << parentPrefix << "_MEM_CONT_OFFSET, "
            << parentPrefix << "_MEM_CONT_SIZE, "
            << parentPrefix << "_MEM_WRAP_OFFSET, "
            << parentPrefix << "_MEM_WRAP_SIZE, "
            << parentPrefix << "_MEM_STRIDE";
    }

    functionCalls << ">("
                        << outputBuffer
                        << inputBuffer << ", "
                        << prefix << "_SCALING"
                    << ");\n\n";

    generateBenchmarkEnd(deepNet, cell, functionCalls);
    generateSaveOutputs(deepNet, cell, functionCalls);
}
