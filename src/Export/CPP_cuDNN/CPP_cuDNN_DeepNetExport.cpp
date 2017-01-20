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
    Utils::createDirectories(dirName + "/include");
    Utils::createDirectories(dirName + "/src");

    C_DeepNetExport::generateParamsHeader(dirName + "/include/params.h");
    C_DeepNetExport::generateEnvironmentHeader(deepNet,
                                               dirName + "/include/env.hpp");
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
    C_DeepNetExport::generateHeaderBegin(deepNet, header, fileName);
    generateHeaderIncludes(deepNet, header);
    CPP_OpenCL_DeepNetExport::generateHeaderConstants(deepNet, header);
    generateHeaderInit(deepNet, header);
    generateHeaderFunction(deepNet, name, header);
    generateHeaderFree(deepNet, header);
    C_DeepNetExport::generateHeaderEnd(deepNet, header);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateHeaderIncludes(DeepNet& deepNet,
                                                           std::ofstream
                                                           & header)
{
    header << "#include \"n2d2_cudnn.hpp\"\n"
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
            header << "#include \"" << cell->getName() << ".h\"\n";
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
    C_DeepNetExport::generateProgramBegin(deepNet, prog);
    generateProgramDesc(deepNet, prog);
    generateProgramGlobalDefinition(deepNet, prog);
    generateProgramInitNetwork(deepNet, prog);
    generateProgramFunction(deepNet, name, prog);
    generateProgramFree(deepNet, prog);
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramDesc(DeepNet& deepNet,
                                                        std::ofstream& prog)
{

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);
            CPP_cuDNN_CellExport::getInstance(cell)
                ->generateCellProgramDesc(cell, prog);
        }
        prog << "\n";
    }
    prog << "\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramGlobalDefinition(
    DeepNet& deepNet, std::ofstream& prog)
{
    prog << "DATA_T *in_buffer(NULL);\n"
            "DATA_T *output_buffer(NULL);\n"
            "DATA_T *ones_vector_buffer(NULL);\n\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    /**Weight & Bias memory objects definition**/
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);
            CPP_cuDNN_CellExport::getInstance(cell)
                ->generateCellProgramGlobalDefinition(cell, prog);
        }
    }
    prog << "\n";

    /**Data buffer objects definition**/
    unsigned int layerNumber = 1;
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        std::vector<unsigned int> archNet = getMapLayer(deepNet, layerNumber);

        std::stringstream outputName;
        unsigned int cellNumber = 0;

        for (unsigned int i = 0; i < archNet.size(); i++) {
            outputName.str("");

            for (unsigned int j = 0; j < archNet[i]; j++) {
                const std::shared_ptr<Cell> cell
                    = deepNet.getCell((*itLayer).at(cellNumber));
                const std::string prefix = cell->getName();

                outputName << prefix + "_";

                if (j == archNet[i] - 1)
                    CPP_cuDNN_CellExport::getInstance(*cell)
                        ->generateCellBuffer(outputName.str() + "buffer", prog);

                ++cellNumber;
            }
        }
        layerNumber++;
    }
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramInitNetwork(DeepNet& deepNet,
                                                               std::ofstream
                                                               & prog)
{
    prog << "void init_network(cudnnHandle_t& context_handle, unsigned int "
            "batchSize) {\n"
            " cudnnDataType_t context_dataType        = CUDNN_DATA_FLOAT;\n"
            " cudnnTensorFormat_t context_tensorFormat    = "
            "CUDNN_TENSOR_NCHW;\n\n";

    prog << " CHECK_CUDA_STATUS( cudaMalloc(&ones_vector_buffer, "
            "batchSize*sizeof(DATA_T)) );\n"
            " std::vector<DATA_T> onesVec(batchSize, 1.0);\n"
            " CHECK_CUDA_STATUS( cudaMemcpy(ones_vector_buffer, &onesVec[0], "
            "batchSize*sizeof(DATA_T), cudaMemcpyHostToDevice) );\n\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);
            CPP_cuDNN_CellExport::getInstance(cell)
                ->generateCellProgramInitNetwork(cell, prog);
        }
    }

    unsigned int layerNumber = 1;
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        std::vector<unsigned int> archNet = getMapLayer(deepNet, layerNumber);
        std::stringstream outputName;
        unsigned int cellNumber = 0;

        if (itLayer == itLayerBegin)
            prog << " CHECK_CUDA_STATUS(cudaMalloc(&"
                 << "in_buffer, sizeof(DATA_T)*"
                 << "ENV_OUTPUTS_SIZE*batchSize));\n";

        for (unsigned int i = 0; i < archNet.size(); i++) {
            outputName.str("");

            for (unsigned int j = 0; j < archNet[i]; j++) {
                const std::shared_ptr<Cell> cell
                    = deepNet.getCell((*itLayer).at(cellNumber));
                const std::string prefix = cell->getName();

                outputName << prefix + "_";

                if (j == archNet[i] - 1)
                    CPP_cuDNN_CellExport::getInstance(*cell)
                        ->generateCellProgramInitBuffer(outputName.str(), prog);

                ++cellNumber;
            }
        }
        if (itLayer + 1 == itLayerEnd)
            prog << " CHECK_CUDA_STATUS(cudaMalloc(&"
                 << "output_buffer, sizeof(DATA_T)*"
                 << Utils::upperCase(outputName.str())
                 << "OUTPUTS_SIZE*batchSize));\n";

        layerNumber++;
    }
    prog << "}\n\n";
}

void N2D2::CPP_cuDNN_DeepNetExport::generateProgramFunction(DeepNet& deepNet,
                                                            const std::string
                                                            & name,
                                                            std::ofstream& prog)
{
    std::string inputsData = "in_buffer";
    std::string outputsData = "output_buffer";

    prog << "\n"
            "void " << name
         << "(DATA_T* in_data, uint32_t* out_data, cudnnHandle_t& "
            "context_handle, cublasHandle_t& context_cublasHandle, unsigned "
            "int batchSize) {\n\n";

    prog << "/************************************INPUT DATA TRANSFER TO "
            "DEVICE***************************************************/\n";
    prog << " CHECK_CUDA_STATUS( cudaMemcpy(" << inputsData
         << ", in_data, batchSize*ENV_BUFFER_SIZE*sizeof(DATA_T), "
            "cudaMemcpyHostToDevice) );\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();
    unsigned int layerNumber = 1;

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        std::vector<unsigned int> archNet = getMapLayer(deepNet, layerNumber);
        std::stringstream outputName;

        unsigned int cellNumber = 0;
        prog << "/************************************LAYER "
             << (layerNumber - 1)
             << "***************************************************/\n";
        if (layerNumber < layers.size()) {
            for (unsigned int i = 0; i < archNet.size(); i++) {
                outputName.str("");
                for (unsigned int j = 0; j < archNet[i]; j++) {

                    const std::shared_ptr<Cell> cell
                        = deepNet.getCell((*itLayer).at(j + cellNumber));
                    const std::string prefix = cell->getName();
                    outputName << prefix + "_";
                }

                for (unsigned int k = 0; k < archNet[i]; k++) {
                    const std::shared_ptr<Cell> cell
                        = deepNet.getCell((*itLayer).at(k + cellNumber));
                    std::string input_buff
                        = getCellInputName(deepNet, layerNumber, k);

                    CPP_cuDNN_CellExport::getInstance(*cell)
                        ->generateCellProgramFunction(*cell,
                                                      input_buff + "buffer",
                                                      outputName.str()
                                                      + "buffer",
                                                      prog,
                                                      "");
                }
                cellNumber += archNet[i];
            }
            if (layerNumber == layers.size() - 1) {
                prog << "/************************************OUTPUT "
                        "LAYER*************************************************"
                        "**/\n";
                const std::shared_ptr<Cell> cell
                    = deepNet.getCell((*itLayer).at(0));
                std::string input_buff
                    = getCellInputName(deepNet, layerNumber, 0);
                CPP_cuDNN_CellExport::getInstance(*cell)
                    ->generateCellProgramOutputFunction(
                          *cell, outputName.str() + "buffer", "out_data", prog);
            }
        }

        layerNumber++;
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

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            Cell& cell = *deepNet.getCell(*it);
            CPP_cuDNN_CellExport::getInstance(cell)
                ->generateCellProgramFree(cell, prog);
        }
    }

    prog << "}\n";
}
