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

#include "Export/CPP/CPP_DeepNetExport.hpp"

N2D2::Registrar<N2D2::DeepNetExport>
N2D2::CPP_DeepNetExport::mRegistrar("CPP", N2D2::CPP_DeepNetExport::generate);

void N2D2::CPP_DeepNetExport::generate(DeepNet& deepNet,
                                       const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");
    Utils::createDirectories(dirName + "/dnn/src");

    generateParamsHeader(dirName + "/include/params.h");
    generateEnvironmentHeader(deepNet, dirName + "/dnn/include/env.hpp");
}


void N2D2::CPP_DeepNetExport::generateParamsHeader(const std::string& fileName)
{
    // Export parameters
    std::ofstream paramsHeader(fileName.c_str());

    if (!paramsHeader.good())
        throw std::runtime_error("Could not create CPP header file: params.h");

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    paramsHeader << "// N2D2 auto-generated file.\n"
                    "// @ " << std::asctime(localNow)
                 << "\n"; // std::asctime() already appends end of line

    paramsHeader << "#ifndef N2D2_EXPORTC_PARAMS_H\n"
                    "#define N2D2_EXPORTC_PARAMS_H\n\n";

    // Constants
    paramsHeader << "#define NB_BITS " << (int)CellExport::mPrecision << "\n\n";

    paramsHeader << "#endif // N2D2_EXPORTC_PARAMS_H" << std::endl;
}

void N2D2::CPP_DeepNetExport::generateEnvironmentHeader(DeepNet& deepNet,
                                                      const std::string
                                                      & fileName)
{
    // Environment
    std::ofstream envHeader(fileName.c_str());

    if (!envHeader.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    envHeader << "// N2D2 auto-generated file.\n"
                 "// @ " << std::asctime(localNow)
              << "\n"; // std::asctime() already appends end of line

    envHeader << "#ifndef N2D2_EXPORTCPP_ENV_LAYER_H\n"
                 "#define N2D2_EXPORTCPP_ENV_LAYER_H\n\n";

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
           "#define ENV_OUTPUTS_SIZE (ENV_NB_OUTPUTS*ENV_SIZE_X*ENV_SIZE_Y)\n"
           "#define ENV_BUFFER_SIZE (ENV_OUTPUTS_SIZE)\n\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    envHeader << "#define NETWORK_TARGETS " << nbTarget << "\n";
    envHeader << "//Output targets network dimension definition:\n";
    envHeader << "static unsigned int OUTPUTS_HEIGHT[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << cell->getOutputsHeight();
    }
    envHeader << "};\n";

    envHeader << "static unsigned int OUTPUTS_WIDTH[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << cell->getOutputsWidth();
    }
    envHeader << "};\n";

    envHeader << "static unsigned int NB_OUTPUTS[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        std::shared_ptr<Cell_Frame_Top> targetCellTop = std::dynamic_pointer_cast
            <Cell_Frame_Top>(cell);

        const Tensor4d<Float_T>& values = targetCellTop->getOutputs();

        envHeader << cell->getNbOutputs()*(values.dimB()/sp->getBatchSize());
    }
    envHeader << "};\n";

    envHeader << "static unsigned int NB_TARGET[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << ((cell->getNbOutputs() > 1) ? cell->getNbOutputs() : 2);
    }
    envHeader << "};\n";

    envHeader << "static unsigned int OUTPUTS_SIZE[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << "(OUTPUTS_WIDTH[" << targetIdx << "]"
                  << "*OUTPUTS_HEIGHT["<< targetIdx << "])";
    }
    envHeader << "};\n";

    envHeader << "#endif // N2D2_EXPORTCPP_ENV_LAYER_H" << std::endl;
}

void N2D2::CPP_DeepNetExport::generateHeaderBegin(DeepNet& /*deepNet*/,
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

void N2D2::CPP_DeepNetExport::generateHeaderIncludes(DeepNet& deepNet,
                                                     const std::string typeStr,
                                                     std::ofstream& header)
{
    header << "#include \"n2d2" + typeStr + ".hpp\"\n"
              "#include \"env.hpp\"\n";
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
            header << "#include \"" << Utils::CIdentifier(cell->getName())
                << ".hpp\"\n";
        }
    }
}


void N2D2::CPP_DeepNetExport::generateHeaderEnd(DeepNet& /*deepNet*/,
                                              std::ofstream& header)
{
    header << "\n"
              "#endif" << std::endl;
    header.close();
}

void N2D2::CPP_DeepNetExport::generateHeaderUtils(std::ofstream& header)
{

    header << "#include \"../../include/typedefs.h\"\n";
    header << "#include \"../../include/utils.h\"\n";
    header << "\n";
    header << "void setProfiling();\n";
    header << "void reportProfiling(unsigned int nbIter);\n";
    header << "unsigned int getOutputNbTargets();\n";
    header << "unsigned int getOutputTarget(unsigned int target);\n";
    header << "unsigned int getOutputDimZ(unsigned int target);\n";
    header << "unsigned int getOutputDimY(unsigned int target);\n";
    header << "unsigned int getOutputDimX(unsigned int target);\n";
    header << "unsigned int getInputDimZ();\n";
    header << "unsigned int getInputDimY();\n";
    header << "unsigned int getInputDimX();\n";
    header << "\n";

}

void N2D2::CPP_DeepNetExport::generateProgramUtils(std::ofstream& prog)
{
    prog << "void setProfiling() { set_profiling(); }\n";
    prog << "void reportProfiling(unsigned int nbIter) { report_per_layer_profiling(nbIter); }\n";
    prog << "unsigned int getOutputNbTargets(){ return NETWORK_TARGETS; }\n";
    prog << "unsigned int getOutputTarget(unsigned int target){ return NB_TARGET[target]; }\n";
    prog << "unsigned int getOutputDimZ(unsigned int target){ return NB_OUTPUTS[target]; }\n";
    prog << "unsigned int getOutputDimY(unsigned int target){ return OUTPUTS_HEIGHT[target]; }\n";
    prog << "unsigned int getOutputDimX(unsigned int target){ return OUTPUTS_WIDTH[target]; }\n";
    prog << "unsigned int getInputDimZ(){ return ENV_NB_OUTPUTS; }\n";
    prog << "unsigned int getInputDimY(){ return ENV_SIZE_Y; }\n";
    prog << "unsigned int getInputDimX(){ return ENV_SIZE_X; }\n";

}
