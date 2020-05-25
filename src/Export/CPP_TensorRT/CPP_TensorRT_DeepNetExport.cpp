/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Export/CPP_TensorRT/CPP_TensorRT_DeepNetExport.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::DeepNetExport> N2D2::CPP_TensorRT_DeepNetExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_DeepNetExport::generate);

void N2D2::CPP_TensorRT_DeepNetExport::generate(DeepNet& deepNet,
                                             const std::string& dirName)
{
    CPP_DeepNetExport::generate(deepNet, dirName);

    generateDeepNetHeader(
        deepNet, "network_tensorRT", dirName + "/dnn/include/network.hpp");

    generateDeepNetProgram(
        deepNet, "network_tensorRT", dirName + "/dnn/src/network.cpp");

    generateStimuliCalib(deepNet, dirName);
}

void N2D2::CPP_TensorRT_DeepNetExport::generateDeepNetHeader(
    DeepNet& deepNet, const std::string& name, const std::string& fileName)
{
    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP network file: "
                                 + fileName);
    CPP_DeepNetExport::generateHeaderBegin(deepNet, header, fileName);
    //CPP_DeepNetExport::generateHeaderIncludes(deepNet, "_tensorRT", header);
    //generateHeaderConstants(deepNet, header);
    CPP_DeepNetExport::generateHeaderUtils(header);
    generateHeaderFunction(deepNet, name, header);
    generateHeaderFree(deepNet, header);
    CPP_DeepNetExport::generateHeaderEnd(deepNet, header);

}

void N2D2::CPP_TensorRT_DeepNetExport::generateHeaderConstants(DeepNet& deepNet,
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
                    std::stringstream outputDepth;
                    std::stringstream outputName;
                    std::string opPlus = " + ";


                    outputName << prefix << "_";

                    header << "#define "
                           << prefix
                           << "_OUTPUT_OFFSET 0\n";

                    for (unsigned int i = 1; i < parentCells.size(); ++i) {
                        const std::string prefix_i
                            = Utils::upperCase(Utils::CIdentifier(
                                                (*parentCells[i]).getName()));

                        outputName << prefix_i << "_";
                        outputDepth << opPlus << prefix_i << "_NB_OUTPUTS";
                        (i == parentCells.size() - 1) ? opPlus = " " : opPlus
                            = "+ ";
                    }
                    header << "#define " << outputName.str() << "NB_OUTPUTS ";
                    header << outputDepth.str() << ")\n";

                }
            }
        }
    }
}

void N2D2::CPP_TensorRT_DeepNetExport::generateHeaderFunction(
    DeepNet& /*deepNet*/, const std::string& name, std::ofstream& header)
{
    header << "void "
        << name << "_init"
        << "(unsigned int batchSize, unsigned int devID, unsigned int iterBuild, int bitPrecision, std::string inputEngine = \"\", std::string outputEngine = \"\", bool useINT8 = false);\n";
    header << "void "
        << name << "_def"
        << "(unsigned int batchSize, unsigned int iterBuild);\n";
    header << "void "
        << name << "_asyncExe"
        << "(DATA_T* in_data, unsigned int batchSize);\n";
    header << "void "
        << name << "_syncExe"
        << "(DATA_T* in_data, unsigned int batchSize);\n";
    header << "void "
        << name << "_syncGPUExe"
        << "(float** externalInOut, unsigned int batchSize);\n";
    header << "void "
        << name << "_output"
        << "(uint32_t* out_data, unsigned int batchSize, unsigned int target);\n";
    header << "void "
        << name << "_fcnn_output"
        << "(uint32_t* out_data, unsigned int batchSize, unsigned int target, float threshold);\n";
    header << "void "
        << name << "_log_output"
        << "(DATA_T* out_data, unsigned int batchSize, unsigned int target);\n";
    header << "void "
        << name << "_overlay_input"
        << "(unsigned char* overlay_data, unsigned int batchSize, unsigned int target, float alpha);\n";
    header << "void* "
        << name << "_get_device_ptr"
        << "(unsigned int target);\n";

}

void N2D2::CPP_TensorRT_DeepNetExport::generateHeaderFree(DeepNet& /*deepNet*/,
                                                       std::ofstream& header)
{
    header << "\n"
           << "void free_memory();\n";
}

void N2D2::CPP_TensorRT_DeepNetExport::generateStimuliCalib(DeepNet& deepNet,
                                                        const std::string& dirName)
{

    CPP_TensorRT_StimuliProvider::generateCalibFiles(*deepNet.getStimuliProvider(),
                                                        dirName + "/batches_calib",
                                                        Database::Test,
                                                        &deepNet);
}


void N2D2::CPP_TensorRT_DeepNetExport::generateDeepNetProgram(
    DeepNet& deepNet, const std::string& name, const std::string& fileName)
{
    std::ofstream prog(fileName.c_str());

    if (!prog.good())
        throw std::runtime_error("Could not create CPP_TensorRT network file: "
                                 + fileName);
    generateProgramBegin(deepNet, prog);
    generateProgramGlobalDefinition(deepNet, prog);
    CPP_DeepNetExport::generateProgramUtils(prog);
    generateProgramInitNetwork(deepNet, name, prog);
    generateProgramFunction(deepNet, name, prog);
    generateGetDevicePtr(name, prog);
    generateProgramFree(deepNet, prog);
}

void N2D2::CPP_TensorRT_DeepNetExport::generateProgramBegin(DeepNet& deepNet,
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
    CPP_DeepNetExport::generateHeaderIncludes(deepNet, "_tensorRT", prog);
    generateHeaderConstants(deepNet, prog);

}

void N2D2::CPP_TensorRT_DeepNetExport::generateProgramDesc(DeepNet& deepNet,
                                                        std::ofstream& prog)
{
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end(); itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end(); it != itEnd; ++it)
        {

            if (itLayer > layers.begin() + 1)
            {
                if (!isSharedInput(deepNet,
                                   std::distance(layers.begin(), itLayer),
                                   std::distance((*itLayer).begin(), it)))
                {
                    const std::vector<std::shared_ptr<Cell> > parentCells
                        = deepNet.getParentCells(*it);

                    if (parentCells.size() > 1)
                    {
                        std::stringstream concatName;

                        for (unsigned int i = 0; i < parentCells.size(); ++i)
                        {
                            const std::string prefix_i =
                                Utils::CIdentifier((*parentCells[i]).getName());

                            concatName << prefix_i << "_";
                        }
                        prog << "nvinfer1::IConcatenationLayer* "
                             << concatName.str() << "layer;\n";

                    }
                }
            }

            Cell& cell = *deepNet.getCell(*it);

            CPP_TensorRT_CellExport::getInstance(cell)->
                generateCellProgramDescriptors(cell, prog);

        }
        prog << "\n";
    }
    prog << "\n";
}

void N2D2::CPP_TensorRT_DeepNetExport::generateProgramGlobalDefinition(
    DeepNet& deepNet, std::ofstream& prog)
{
/*
    if((int)CellExport::mPrecision == -32)
        prog << "nvinfer1::DataType dT = nvinfer1::DataType::kFLOAT;\n";
    else if ((int)CellExport::mPrecision == -16)
        prog << "nvinfer1::DataType dT = nvinfer1::DataType::kHALF;\n";
    else if ((int)CellExport::mPrecision == 8)
        prog << "nvinfer1::DataType dT = nvinfer1::DataType::kINT8;\n";
    else {
        std::stringstream precision;
        precision << (int)CellExport::mPrecision;

        throw std::runtime_error("Could not generate a TensorRT export on "
                                  + precision.str() + " bits precision");
    }
*/
    //prog << "nvinfer1::DataType dT;\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();
    const unsigned int nbTarget = outputTargets.size();
    // Add 1 for the input buffer (must be unique
    prog << "void* inout_buffer["<< nbTarget + 1 << "];\n";
    // Workspace GPU use for add_weighted in segmentation
    prog << "unsigned char* workspace_gpu;\n";
    prog << "cudaStream_t dataStream;\n";
    prog << "PluginFactory pluginFactory;\n";
    prog << "bool mUseDLA = false;\n";
    prog << "\n";
}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateProgramInitNetwork(DeepNet& deepNet,
                                 const std::string& name,
                                 std::ofstream& prog)
{
    std::string outputsBuffer = "out";
    std::vector<std::string>  pNameFactory;

    prog << "void "
         << name
         << "_init(unsigned int batchSize, unsigned int devID,"
         << "unsigned int iterBuild, int bitPrecision, std::string inputEngine, std::string outputEngine, bool useINT8) {\n";
    prog << "   tsrRTHandles.dT = (bitPrecision == -32) ? nvinfer1::DataType::kFLOAT : (bitPrecision == -16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;\n";
    prog << "   cudaSetDevice(devID);\n";
	prog << "   CHECK_CUDA_STATUS(cudaStreamCreate(&dataStream));\n";
    prog << "   CHECK_CUDA_STATUS( cudaMalloc(&inout_buffer[0],"
        " batchSize*ENV_OUTPUTS_SIZE*sizeof(DATA_T)) );\n";
    prog << "   CHECK_CUDA_STATUS( cudaMalloc(&workspace_gpu,"
        " batchSize*ENV_SIZE_X*ENV_SIZE_Y*3*sizeof(unsigned char)) );\n";

    prog << "   if(inputEngine.empty()){\n";
    prog << "       tsrRTHandles.netBuilder = nvinfer1::createInferBuilder(gLogger);\n";
    prog << "       tsrRTHandles.netDef.push_back(tsrRTHandles.netBuilder->createNetwork());\n";
    prog << "#if NV_TENSORRT_MAJOR > 4\n";
    prog << "       if(tsrRTHandles.dT == nvinfer1::DataType::kHALF)\n";
    prog << "              tsrRTHandles.netBuilder->setFp16Mode(true);\n";
    prog << "#endif\n";
    prog << "       network_tensorRT_def(batchSize, devID);\n";
    prog << "   }\n";

    prog << "   createContext(batchSize, iterBuild, pluginFactory, inputEngine, outputEngine, useINT8);\n\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    prog << "//Initialization of the " << nbTarget << " network targets:\n";

    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);

        CPP_TensorRT_CellExport::getInstance(*cell)->
            generateCellProgramAllocateMemory(targetIdx, prog);
    }

    prog << "}\n";

    prog << "void "
         << name
         << "_def(unsigned int batchSize, unsigned int devID){\n";

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    /*prog << "   auto in_tensor = netDef->addInput(\"ENV_INPUT\","
         << " dT, nvinfer1::DimsCHW{ENV_NB_OUTPUTS,"
         << " ENV_SIZE_Y, ENV_SIZE_X});\n";*/

    prog << "   std::vector<nvinfer1::ITensor *> in_tensor;\n";
    prog << "   in_tensor.push_back(tsrRTHandles.netDef.back()->addInput(\"ENV_INPUT\","
         << " tsrRTHandles.dT, nvinfer1::DimsCHW{ENV_NB_OUTPUTS,"
         << " ENV_SIZE_Y, ENV_SIZE_X}));\n";
    prog << "   in_tensor.back()->setType(nvinfer1::DataType::kFLOAT);\n"
        << "\n\n";

    /** Network instantiation **/
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
                parentsName.push_back("in");

            }
            else {
                const std::vector<std::shared_ptr<Cell> >&
                    parentCells = deepNet.getParentCells(cell->getName());

                for(unsigned int k = 0; k < parentCells.size(); ++k) {
                    parentsName.push_back(Utils::CIdentifier(
                                                parentCells[k]->getName()));


                }

                if(parentsName.size() > 1 ) {
                    std::stringstream pName;
                    bool isConcat = true;

                    for(unsigned int k = 1; k < parentCells.size(); ++k)
                        if( (parentCells[k]
                                ->getOutputsWidth() !=
                             parentCells[k-1]
                                ->getOutputsWidth()) ||
                            (parentCells[k]
                                ->getOutputsHeight() !=
                             parentCells[k-1]->getOutputsHeight()))
                            isConcat = false;

                    for(unsigned int i = 0; i < parentsName.size(); ++i)
                        pName << parentsName[i];

                    std::vector<std::string>::iterator
                        itNameFactory = std::find ( pNameFactory.begin(),
                                                    pNameFactory.end(),
                                                    pName.str());

                    if(itNameFactory == pNameFactory.end())
                    {
                        CPP_TensorRT_CellExport
                            ::generateTensor(*cell, parentsName, prog);

                        if(isConcat)
                            CPP_TensorRT_CellExport
                                ::generateAddConcat(*cell, parentsName, prog);
                    }
                    pNameFactory.push_back(pName.str());
                }
            }/*
            std::string output_buff = (itLayer >= itLayerEnd - 1) ? outputsBuffer :
                getCellOutputName(deepNet,
                                  std::distance(layers.begin(),itLayer),
                                  std::distance((*itLayer).begin(), it));
*/
            CPP_TensorRT_CellExport::getInstance(*cell)->
                generateCellProgramInstanciateLayer(*cell, parentsName, prog);

       }
    }
    prog << "//Initialization of the " << nbTarget << " network targets:\n";

    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);

        CPP_TensorRT_CellExport::getInstance(*cell)->
            generateCellProgramInstanciateOutput((*cell),
                                                 targetIdx,
                                                 prog);
    }

    prog << "\n\n";

    prog << "\n\n";
    prog << "}\n";
}


void N2D2::CPP_TensorRT_DeepNetExport
    ::generateProgramFunction(DeepNet& /*deepNet*/,
                              const std::string& name,
                              std::ofstream& prog)
{
    generateAsyncFunction(name, prog);
    generateSyncFunction(name, prog);
    generateSyncGPUFunction(name, prog);
    generateOutputFunction(name, prog);
}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateAsyncFunction(const std::string& name,
                            std::ofstream& prog)
{
    prog << "void " << name << "_asyncExe(DATA_T* in_data, unsigned int batchSize) {\n";
    prog << "\n";
	prog << "   " << "CHECK_CUDA_STATUS(cudaMemcpyAsync(inout_buffer[0],\n"
         << "       " << "in_data,\n"
         << "       " << "batchSize*ENV_BUFFER_SIZE*sizeof(DATA_T),\n"
         << "       " << "cudaMemcpyHostToDevice,\n"
         << "       " << "dataStream));\n";
    prog <<"\n";
	prog << "   " << "tsrRTHandles.context->enqueue(batchSize, inout_buffer, dataStream, nullptr);\n";
    prog << "}";
    prog <<"\n";
}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateSyncFunction(const std::string& name,
                            std::ofstream& prog)
{
    prog << "void " << name << "_syncExe(DATA_T* in_data, unsigned int batchSize) {\n";
    prog << "\n";
	prog << "   " << "CHECK_CUDA_STATUS(cudaMemcpy(inout_buffer[0],\n"
         << "       " << "in_data,\n"
         << "       " << "batchSize*ENV_BUFFER_SIZE*sizeof(DATA_T),\n"
         << "       " << "cudaMemcpyHostToDevice));\n";
    prog <<"\n";
	prog << "   " << "tsrRTHandles.context->execute(batchSize, inout_buffer);\n";
    prog << "}";
    prog <<"\n";
}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateSyncGPUFunction(const std::string& name,
                            std::ofstream& prog)
{
    prog << "void " << name << "_syncGPUExe(float** externalInOut, unsigned int batchSize) {\n";
    prog << "\n";
	prog << "   " << "tsrRTHandles.context->execute(batchSize, reinterpret_cast<void**>(externalInOut));\n";
    prog << "}";
    prog <<"\n";
}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateOutputFunction(const std::string& name,
                            std::ofstream& prog)
{
    prog << "void " << name << "_output(uint32_t* out_data, unsigned int batchSize, unsigned int target) {\n";
    prog << "\n";
	prog << "   " << "spatial_output_generation(batchSize,\n"
         << "       " << "NB_OUTPUTS[target],\n"
         << "       " << "OUTPUTS_HEIGHT[target],\n"
         << "       " << "OUTPUTS_WIDTH[target],\n"
         << "       " << "inout_buffer[target + 1],\n"
         << "       " << "out_data,\n"
         << "       " <<  "dataStream"
         << ");\n";

    prog << "}";
    prog <<"\n";

    prog << "void " << name << "_fcnn_output(uint32_t* out_data, unsigned int batchSize, unsigned int target, float threshold) {\n";
    prog << "\n";
	prog << "   " << "spatial_output_generation(batchSize,\n"
         << "       " << "NB_OUTPUTS[target],\n"
         << "       " << "OUTPUTS_HEIGHT[target],\n"
         << "       " << "OUTPUTS_WIDTH[target],\n"
         << "       " << "inout_buffer[target + 1],\n"
         << "       " << "out_data,\n"
         << "       " <<  "dataStream,\n"
         << "       " <<  "threshold,\n"
         << "       " <<  "true"
         << ");\n";

    prog << "}";
    prog <<"\n";

    prog << "void " << name << "_log_output(DATA_T* out_data, unsigned int batchSize, unsigned int target) {\n";
    prog << "\n";
	prog << "   " << "get_output(batchSize,\n"
         << "       " << "NB_OUTPUTS[target],\n"
         << "       " << "OUTPUTS_HEIGHT[target],\n"
         << "       " << "OUTPUTS_WIDTH[target],\n"
         << "       " << "inout_buffer[target + 1],\n"
         << "       " << "out_data);\n";
    prog << "}";
    prog <<"\n";

    prog << "void " << name << "_overlay_input(unsigned char* overlay_data, unsigned int batchSize, unsigned int target, float alpha) {\n";
    prog << "\n";
	prog << "   " << "add_weighted(batchSize,\n"
         << "       " << "NB_OUTPUTS[target],\n"
         << "       " << "OUTPUTS_HEIGHT[target],\n"
         << "       " << "OUTPUTS_WIDTH[target],\n"
         << "       " << "reinterpret_cast<DATA_T *>(inout_buffer[target + 1]),\n"
         << "       " << "ENV_NB_OUTPUTS,\n"
         << "       " << "ENV_SIZE_Y,\n"
         << "       " << "ENV_SIZE_X,\n"
         << "       " << "reinterpret_cast<DATA_T *>(inout_buffer[0]),\n"
         << "       " << "overlay_data,\n"
         << "       " << "workspace_gpu,\n"
         << "       " << "alpha,\n"
         << "       " << "dataStream);\n";
    prog << "}";
    prog <<"\n";

}

void N2D2::CPP_TensorRT_DeepNetExport::generateGetDevicePtr(const std::string& name,
                                                            std::ofstream& prog)
{
    prog << "void* " << name << "_get_device_ptr(unsigned int target) {\n\n";

    prog << "    return inout_buffer[target + 1];\n";
    prog << "}\n";

}


void N2D2::CPP_TensorRT_DeepNetExport::generateProgramFree(DeepNet& /*deepNet*/,
                                                        std::ofstream& prog)
{
    prog << "\n"
        "void free_memory(){\n\n";

    prog << "    CHECK_CUDA_STATUS( cudaFree(inout_buffer[1]) );\n";
    prog << "    CHECK_CUDA_STATUS( cudaFree(inout_buffer[0]) );\n";

    prog << "}\n";

}
