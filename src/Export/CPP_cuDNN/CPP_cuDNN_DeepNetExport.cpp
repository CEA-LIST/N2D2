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

#include "Export/CPP_cuDNN/CPP_cuDNN_DeepNetExport.hpp"

N2D2::Registrar<N2D2::DeepNetExport> N2D2::CPP_cuDNN_DeepNetExport::mRegistrar(
    "CPP_cuDNN", N2D2::CPP_cuDNN_DeepNetExport::generate);

void N2D2::CPP_cuDNN_DeepNetExport::generate(DeepNet& deepNet,
                                             const std::string& dirName)
{
    CPP_DeepNetExport::generate(deepNet, dirName);

    generateDeepNetHeader(
        deepNet, "network_cudnn", dirName + "/dnn/include/network.hpp");

    generateDeepNetProgram(
        deepNet, "network_cudnn", dirName + "/dnn/src/network.cpp");
}

void N2D2::CPP_cuDNN_DeepNetExport::generateDeepNetHeader(
    DeepNet& deepNet, const std::string& name, const std::string& fileName)
{
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP network file: "
                                 + fileName);
    CPP_DeepNetExport::generateHeaderBegin(deepNet, header, fileName);
    CPP_DeepNetExport::generateHeaderUtils(header);
    generateHeaderInit(deepNet, name, header);
    generateHeaderFunction(deepNet, name, header);
    generateHeaderFree(deepNet, name, header);
    CPP_DeepNetExport::generateHeaderEnd(deepNet, header);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderConstants(DeepNet& deepNet,
                                                             std::ofstream
                                                             & header)
{

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 2,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);

                const std::string prefix
                    = Utils::upperCase(Utils::CIdentifier(
                                            (*parentCells[0]).getName()));

                if (parentCells.size() > 1) {
                    std::stringstream outputOffset;
                    std::stringstream outputDepth;
                    std::stringstream outputName;
                    std::string opPlus = " + ";


                    outputOffset << "(" << prefix << "_OUTPUTS_SIZE ";
                    outputDepth << "(" << prefix << "_NB_OUTPUTS ";
                    outputName << prefix << "_";

                    header << "#define "
                           << prefix
                           << "_OUTPUT_OFFSET 0\n";

                    for (unsigned int i = 1; i < parentCells.size(); ++i) {
                        const std::string prefix_i
                            = Utils::upperCase(Utils::CIdentifier(
                                                (*parentCells[i]).getName()));

                        header << "#define " << prefix_i << "_OUTPUT_OFFSET ";
                        header << i << "\n";

                        outputName << prefix_i << "_";
                        outputOffset << opPlus << prefix_i << "_OUTPUTS_SIZE";
                        outputDepth << opPlus << prefix_i << "_NB_OUTPUTS";
                        (i == parentCells.size() - 1) ? opPlus = " " : opPlus
                            = "+ ";
                    }
                    header << "#define " << outputName.str() << "NB_OUTPUTS ";
                    header << outputDepth.str() << ")\n";
                    header << "#define " << outputName.str() << "OUTPUTS_SIZE ";
                    header << outputOffset.str() << ")\n";
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

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderInit(DeepNet& /*deepNet*/,
                                                       const std::string& name,
                                                       std::ofstream& header)
{
    header << "\n"
              "void " << name << "_init(unsigned int batchSize, "
                              << "unsigned int devID);\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderFunction(
    DeepNet& /*deepNet*/, const std::string& name, std::ofstream& header)
{
    header << "void " << name << "_syncExe"
           << "(DATA_T* in_data,  unsigned int batchSize);";
    header <<"\n";
    header << "void " << name << "_output"
           << "(uint32_t* out_data, unsigned int batchSize, "
           << "unsigned int target);";
    header <<"\n";

}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderFree(DeepNet& /*deepNet*/,
                                                       const std::string& /*name*/,
                                                       std::ofstream& header)
{
    header << "\n"
           << "void free_memory();\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateDeepNetProgram(
    DeepNet& deepNet, const std::string& name, const std::string& fileName)
{
    std::ofstream prog(fileName.c_str());

    if (!prog.good())
        throw std::runtime_error("Could not create cuDNN network file: "
                                 + fileName);
    generateProgramBegin(deepNet, prog);
    generateProgramDesc(deepNet, prog);
    generateProgramGlobalDefinition(deepNet, prog);
    CPP_DeepNetExport::generateProgramUtils(prog);
    generateProgramInitNetwork(deepNet, name, prog);
    generateProgramFunction(deepNet, name, prog);
    generateOutputFunction(name, prog);
    generateProgramFree(deepNet, prog);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramBegin(DeepNet& deepNet,
                                                 std::ofstream& prog)
{
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    // Program
    prog << "// N2D2 auto-generated file.\n"
            "// @ " << std::asctime(localNow)
         << "\n" // std::asctime() already appends end of line
            "#include \"network.hpp\"\n"
            "\n"
            "//#define DATA_DYN_ANALYSIS\n"
            "\n";
    CPP_DeepNetExport::generateHeaderIncludes(deepNet, "_cudnn", prog);
    generateHeaderConstants(deepNet, prog);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramDesc(DeepNet& deepNet,
                                                        std::ofstream& prog)
{

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end(); itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end(); it != itEnd; ++it) {

            Cell& cell = *deepNet.getCell(*it);

            CPP_cuDNN_CellExport::getInstance(cell)->
                generateCellProgramDesc(cell, prog);
        }
        prog << "\n";
    }
    prog << "\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramGlobalDefinition(
    DeepNet& deepNet, std::ofstream& prog)
{

    prog << "std::vector<DATA_T *> in_buffer;\n"
        "std::vector<DATA_T *> output_buffer;\n"
        "DATA_T *ones_vector_buffer(NULL);\n\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    /**Weight & Bias memory objects definition**/
    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerEnd = layers.end();
        itLayer != itLayerEnd;
        ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);

            CPP_cuDNN_CellExport::getInstance(cell)->
                generateCellProgramGlobalDefinition(cell, prog);
        }
    }
    prog << "\n";

    /**Data buffer objects global definition**/
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 2,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator
             it = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);
                const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

                std::stringstream outputName;
                outputName << Utils::CIdentifier((*parentCells[0]).getName())
                    << "_";

                for (unsigned int i = 1; i < parentCells.size(); ++i) {
                    outputName << Utils::CIdentifier(
                                                (*parentCells[i]).getName())
                               << "_";
                }

                CPP_cuDNN_CellExport::getInstance(*cell)
                    ->generateCellBuffer(outputName.str() + "buffer", prog);
            }
        }
    }

    prog << "\n\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramInitNetwork(DeepNet& deepNet,
                                                               const std::string& name,
                                                               std::ofstream& prog)
{
    std::string outputsBuffer = "output_";
    std::string output_buff;

    prog << "void " << name << "_init"
        << "(unsigned int batchSize, unsigned int devID) {\n"
        << "    CudaContext::setDevice(devID);\n"
        << "    cudnnDataType_t context_dataType ="
        " CUDNN_DATA_FLOAT;\n"
        << "    cudnnTensorFormat_t context_tensorFormat ="
        " CUDNN_TENSOR_NCHW;\n\n";

    prog << "    CHECK_CUDA_STATUS( cudaMalloc(&ones_vector_buffer,"
        " batchSize*sizeof(DATA_T)) );\n"
        "    std::vector<DATA_T> onesVec(batchSize, 1.0);\n"
        "    CHECK_CUDA_STATUS( cudaMemcpy(ones_vector_buffer, &onesVec[0],"
        " batchSize*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n\n";

    prog << "    in_buffer.push_back(new DATA_T());\n"
        "    CHECK_CUDA_STATUS( cudaMalloc(&in_buffer.back(),"
        " batchSize*ENV_OUTPUTS_SIZE*sizeof(DATA_T)) );\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    /**Data buffer objects resizing**/
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 2,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator
             it = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);
                const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

                std::stringstream outputName;
                outputName << Utils::CIdentifier((*parentCells[0]).getName())
                    << "_";

                for (unsigned int i = 1; i < parentCells.size(); ++i) {
                    outputName << Utils::CIdentifier(
                                            (*parentCells[i]).getName())
                               << "_";
                }

                prog << "    " << outputName.str() << "buffer.resize("
                    << std::to_string((unsigned long long int)
                                      parentCells.size()) << ");\n";
            }
        }
    }

    prog << "    output_buffer.resize(NETWORK_TARGETS);";

    prog << "\n\n";

    /**Tensors initialization **/
    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerEnd = layers.end();
        itLayer != itLayerEnd;
        ++itLayer)
    {

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it)
        {
            std::vector<std::string> parentsName;
            const std::shared_ptr<Cell>
                cell = deepNet.getCell((*itLayer).
                    at(std::distance((*itLayer).begin(), it)));

            if (itLayer == layers.begin() + 1) {
                parentsName.push_back("env");
            }
            else {
                const std::vector<std::shared_ptr<Cell> >&
                    parentCells = deepNet.getParentCells(cell->getName());

                for(unsigned int k = 0; k < parentCells.size(); ++k) {
                    parentsName.push_back(Utils::CIdentifier(
                                                parentCells[k]->getName()));
                }
            }
            output_buff = getCellOutputName(deepNet,
                                        std::distance(layers.begin(),itLayer),
                                        std::distance((*itLayer).begin(), it));

            CPP_cuDNN_CellExport::getInstance(*cell)->
                generateCellProgramInitNetwork(*cell, parentsName, prog);

            if(!output_buff.empty())
                CPP_cuDNN_CellExport::getInstance(*cell)->
                    generateCellProgramInitBuffer(*cell, output_buff, prog);
        }
    }

    prog << "\n\n";
    prog << "   " << "set_output( NETWORK_TARGETS );\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    prog << "//Initialization of the " << nbTarget << " network targets:\n";

    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);

        prog << "   " << "CHECK_CUDA_STATUS( cudaMalloc(&output_buffer["
                      << targetIdx << "], " // Added 1 for stride the input buffer
                      << "sizeof(DATA_T)*batchSize"
                      << "*NB_OUTPUTS[" << targetIdx << "]"
                      << "*OUTPUTS_HEIGHT[" << targetIdx << "]"
                      << "*OUTPUTS_WIDTH[" << targetIdx << "]"
                      << "));\n";
    }
    prog << "\n\n";


    prog << "}\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramFunction(DeepNet& deepNet,
                                                            const std::string
                                                            & name,
                                                            std::ofstream& prog)
{
    std::string inputsBuffer = "in_";
    std::string outputsBuffer = "output_";
    std::string input_buff;
    std::string output_buff;


    prog << "\n"
        "void " << name << "_syncExe"
            "(DATA_T* in_data, "
            "unsigned int batchSize) {\n\n";

    prog <<  "/*******INPUT DATA TRANSFER TO DEVICE*********/\n" ;
    prog << "    CHECK_CUDA_STATUS( cudaMemcpy(" << inputsBuffer << "buffer[0]"
        << ", in_data, batchSize*ENV_BUFFER_SIZE*sizeof(DATA_T),"
        << " cudaMemcpyHostToDevice) );\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();
    unsigned int targetIdx = 0;

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerBegin = layers.begin() + 1,
        itLayerEnd = layers.end();
        itLayer != itLayerEnd;
        ++itLayer)
    {
        prog << "/** LAYER ("
            << std::distance(layers.begin(), itLayer) << ") **/\n" ;

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {

            const std::shared_ptr<Cell>
                cell = deepNet.getCell((*itLayer).
                    at(std::distance((*itLayer).begin(), it)));

            std::string outputOffset
                 = Utils::upperCase(Utils::CIdentifier((*cell).getName())
                                    + "_output_offset");

            input_buff = (itLayer == itLayerBegin) ? inputsBuffer :
                getCellInputName(deepNet,
                                 std::distance(layers.begin(), itLayer),
                                 std::distance((*itLayer).begin(), it));

            output_buff = getCellOutputName(deepNet,
                                        std::distance(layers.begin(), itLayer),
                                        std::distance((*itLayer).begin(), it));

            if(output_buff.empty())
            {
                if(targetIdx >= outputTargets.size())
                    throw std::runtime_error("CPP_cuDNN_DeepNetExport::generateProgramFunction(): "
                                             "targetIdx cannot be superior to the number of outputs network targets");
                std::stringstream targetIdxStr;
                targetIdxStr << targetIdx;

                CPP_cuDNN_CellExport::getInstance(*cell)
                    ->generateCellProgramFunction(*cell, input_buff + "buffer",
                    "output_buffer", targetIdxStr.str(), prog, "");
                ++targetIdx;
            }
            else
                CPP_cuDNN_CellExport::getInstance(*cell)
                ->generateCellProgramFunction(*cell, input_buff + "buffer",
                output_buff + "buffer", outputOffset, prog, "");
        }
        prog << "\n";
    }
    prog << "}\n";
}

void N2D2::CPP_cuDNN_DeepNetExport
    ::generateOutputFunction(const std::string& name,
                            std::ofstream& prog)
{
    prog << "void " << name << "_output(uint32_t* out_data, unsigned int batchSize, unsigned int target) {\n";
    prog << "\n";
	prog << "   " << "spatial_output_generation(batchSize,\n"
         << "       " << "NB_OUTPUTS[target],\n"
         << "       " << "OUTPUTS_HEIGHT[target],\n"
         << "       " << "OUTPUTS_WIDTH[target],\n"
         << "       " << "output_buffer[target],\n"
         << "       " << "out_data);\n";
    prog << "}";
    prog <<"\n";

}


void N2D2::CPP_cuDNN_DeepNetExport::generateProgramFree(DeepNet& deepNet,
                                                        std::ofstream& prog)
{

    prog << "\n"
        "void free_memory(){\n\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerEnd = layers.end();
        itLayer != itLayerEnd;
        ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it) {

            std::vector<std::string> parentsName;
            const std::shared_ptr<Cell>
                cell = deepNet.getCell((*itLayer).
                    at(std::distance((*itLayer).begin(), it)));

            if(itLayer == layers.begin() + 1){
                parentsName.push_back("env");
            }
            else {
                const std::vector<std::shared_ptr<Cell> >&
                    parentCells = deepNet.getParentCells(cell->getName());

                for(unsigned int k = 0; k < parentCells.size(); ++k) {
                    parentsName.push_back(Utils::CIdentifier(
                                                parentCells[k]->getName()));
                }
            }

            CPP_cuDNN_CellExport::getInstance(*cell)
                ->generateCellProgramFree(*cell, parentsName, prog);
        }
    }

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    prog << "//Destruction of the " << nbTarget << " network targets:\n";

    for(int targetIdx = nbTarget - 1; targetIdx >= 0; --targetIdx)
        prog << "    CHECK_CUDA_STATUS( cudaFree(output_buffer["
             << targetIdx << "]) );\n";
    prog << "}\n";
}
