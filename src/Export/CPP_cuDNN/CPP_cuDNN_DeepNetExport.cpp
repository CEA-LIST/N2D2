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
        deepNet, "network_cudnn", dirName + "/include/network.hpp");

    generateDeepNetProgram(
        deepNet, "network_cudnn", dirName + "/src/network.cpp");
}

void N2D2::CPP_cuDNN_DeepNetExport::generateDeepNetHeader(
    DeepNet& deepNet, const std::string& name, const std::string& fileName)
{
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP network file: "
                                 + fileName);
    CPP_DeepNetExport::generateHeaderBegin(deepNet, header, fileName);
    CPP_DeepNetExport::generateHeaderIncludes(deepNet, "_cudnn", header);
    generateHeaderConstants(deepNet, header);
    generateHeaderInit(deepNet, header);
    generateHeaderFunction(deepNet, name, header);
    generateHeaderFree(deepNet, header);
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
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);

                if (parentCells.size() > 1) {

                    std::stringstream outputOffset;
                    std::stringstream outputDepth;
                    std::stringstream outputName;
                    std::string opPlus = " + ";

                    outputOffset << "(" << Utils::upperCase(
                                               (*parentCells[0]).getName())
                                 << "_OUTPUTS_SIZE ";
                    outputDepth << "("
                                << Utils::upperCase((*parentCells[0]).getName())
                                << "_NB_OUTPUTS ";
                    outputName << Utils::upperCase((*parentCells[0]).getName())
                               << "_";

                    header << "#define "
                           << Utils::upperCase((*parentCells[0]).getName())
                           << "_OUTPUT_OFFSET 0\n";

                    for (unsigned int i = 1; i < parentCells.size(); ++i) {

                        header << "#define "
                               << Utils::upperCase((*parentCells[i]).getName())
                               << "_OUTPUT_OFFSET ";
                        header << i << "\n";

                        outputName << Utils::upperCase(
                                          (*parentCells[i]).getName()) << "_";
                        outputOffset << opPlus
                                     << Utils::upperCase(
                                            (*parentCells[i]).getName())
                                     << "_OUTPUTS_SIZE";
                        outputDepth << opPlus << Utils::upperCase((
                                                     *parentCells[i]).getName())
                                    << "_NB_OUTPUTS";
                        (i == parentCells.size() - 1) ? opPlus = " " : opPlus
                            = "+ ";
                    }
                    header << "#define " << outputName.str() << "NB_OUTPUTS ";
                    header << outputDepth.str() << ")\n";
                    header << "#define " << outputName.str() << "OUTPUTS_SIZE ";
                    header << outputOffset.str() << ")\n";
                } else {
                    header << "#define "
                           << Utils::upperCase((*parentCells[0]).getName())
                           << "_OUTPUT_OFFSET 0\n";
                }
            }
            if (itLayer == itLayerEnd - 1) {
                const std::shared_ptr<Cell> cell
                    = deepNet.getCell((*itLayer).at(0));

                header << "#define " << Utils::upperCase(cell->getName())
                       << "_OUTPUT_OFFSET 0\n";
            }
        }
    }
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderInit(DeepNet& /*deepNet*/,
                                                       std::ofstream& header)
{
    header << "\n"
              "void init_network(cudnnHandle_t& context_handle, unsigned int "
              "batchSize);\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderFunction(
    DeepNet& /*deepNet*/, const std::string& name, std::ofstream& header)
{
    header << "\n"
              "void " << name
           << "(DATA_T* in_data, uint32_t* out_data, cudnnHandle_t& "
              "context_handle, cublasHandle_t& context_cublasHandle, unsigned "
              "int batchSize);\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderFree(DeepNet& /*deepNet*/,
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
        throw std::runtime_error("Could not create C network file: "
                                 + fileName);
    generateProgramBegin(deepNet, prog);
    generateProgramDesc(deepNet, prog);
    generateProgramGlobalDefinition(deepNet, prog);
    generateProgramInitNetwork(deepNet, prog);
    generateProgramFunction(deepNet, name, prog);
    generateProgramFree(deepNet, prog);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramBegin(DeepNet& /*deepNet*/,
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
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramDesc(DeepNet& deepNet,
                                                        std::ofstream& prog)
{

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerBegin = layers.begin() + 1,
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
        itLayerBegin = layers.begin() + 1,
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
             itBegin = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);
                const std::shared_ptr<Cell> cell = deepNet.getCell(
                    (*itLayer).at(std::distance((*itLayer).begin(), it)));
                std::stringstream outputName;
                outputName << (*parentCells[0]).getName() << "_";

                for (unsigned int i = 1; i < parentCells.size(); ++i)
                    outputName << (*parentCells[i]).getName() << "_";

                CPP_cuDNN_CellExport::getInstance(*cell)
                    ->generateCellBuffer(outputName.str() + "buffer", prog);
            }
        }
    }

    prog << "\n\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramInitNetwork(DeepNet& deepNet,
                                                               std::ofstream
                                                               & prog)
{
    std::string outputsBuffer = "output_";
    std::string output_buff;

    prog << "void init_network(cudnnHandle_t& context_handle,"
        "unsigned int batchSize) {\n"
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
             itBegin = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {

            if (!isSharedInput(deepNet,
                               std::distance(layers.begin(), itLayer),
                               std::distance((*itLayer).begin(), it))) {
                const std::vector<std::shared_ptr<Cell> > parentCells
                    = deepNet.getParentCells(*it);
                const std::shared_ptr<Cell> cell = deepNet.getCell(
                    (*itLayer).at(std::distance((*itLayer).begin(), it)));
                std::stringstream outputName;
                outputName << (*parentCells[0]).getName() << "_";

                for (unsigned int i = 1; i < parentCells.size(); ++i)
                    outputName << (*parentCells[i]).getName() << "_";

                prog << "    " << outputName.str() << "buffer.resize("
                    << std::to_string((unsigned long long int)
                                      parentCells.size()) << ");\n";

                if(itLayer == itLayerEnd - 1) {
                    prog << "    output_buffer.resize("
                        << std::to_string((unsigned long long int)
                                          parentCells.size()) << ");\n";
                }
            }
        }
    }

    prog << "\n\n";

    /**Tensors initialization **/
    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerEnd = layers.end();
        itLayer != itLayerEnd;
        ++itLayer)
    {

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
             itBegin = (*itLayer).begin(),
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

                for(unsigned int k = 0; k < parentCells.size(); ++k)
                    parentsName.push_back(parentCells[k]->getName());
            }
            output_buff = (itLayer >= itLayerEnd - 1) ? outputsBuffer :
                getCellOutputName(deepNet,
                                  std::distance(layers.begin(),itLayer),
                                  std::distance((*itLayer).begin(), it));

            CPP_cuDNN_CellExport::getInstance(*cell)->
                generateCellProgramInitNetwork(*cell, parentsName, prog);
            CPP_cuDNN_CellExport::getInstance(*cell)->
                generateCellProgramInitBuffer(*cell, output_buff, prog);
        }
    }
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
        "void " << name << "(DATA_T* in_data, uint32_t* out_data,"
            " cudnnHandle_t& context_handle,"
            " cublasHandle_t& context_cublasHandle,"
            " unsigned int batchSize) {\n\n";

    prog <<  "/*******INPUT DATA TRANSFER TO DEVICE*********/\n" ;
    prog << "    CHECK_CUDA_STATUS( cudaMemcpy(" << inputsBuffer << "buffer[0]"
        << ", in_data, batchSize*ENV_BUFFER_SIZE*sizeof(DATA_T),"
        << " cudaMemcpyHostToDevice) );\n";

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
            itBegin = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {

            const std::shared_ptr<Cell>
                cell = deepNet.getCell((*itLayer).
                    at(std::distance((*itLayer).begin(), it)));

            std::string outputOffset
                 = Utils::upperCase((*cell).getName() + "_output_offset");

            input_buff = (itLayer == itLayerBegin) ? inputsBuffer :
                getCellInputName(deepNet,
                                 std::distance(layers.begin(), itLayer),
                                 std::distance((*itLayer).begin(), it));

            output_buff = (itLayer >= itLayerEnd - 1) ? outputsBuffer :
                getCellOutputName(deepNet,
                                  std::distance(layers.begin(), itLayer),
                                  std::distance((*itLayer).begin(), it));

            CPP_cuDNN_CellExport::getInstance(*cell)
                ->generateCellProgramFunction(*cell, input_buff + "buffer",
                output_buff + "buffer", outputOffset, prog, "");

            if(itLayer == itLayerEnd - 1) {
                const std::shared_ptr<Cell>
                    cell = deepNet.getCell((*itLayer).at(0));

                CPP_cuDNN_CellExport::getInstance(*cell)
                    ->generateCellProgramOutputFunction(*cell, output_buff
                    + "buffer[0]", "out_data", prog);
            }
        }
        prog << "\n";
    }
    prog << "}\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramFree(DeepNet& deepNet,
                                                        std::ofstream& prog)
{

    prog << "\n"
        "void free_memory(){\n\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator
        itLayer = layers.begin() + 1,
        itLayerBegin = layers.begin() + 1,
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

                for(unsigned int k = 0; k < parentCells.size(); ++k)
                    parentsName.push_back(parentCells[k]->getName());
            }

            CPP_cuDNN_CellExport::getInstance(*cell)
                ->generateCellProgramFree(*cell, parentsName, prog);
        }
    }

    prog << "}\n";
}
