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

#include "Export/CPP_TensorRT/CPP_TensorRT_BatchNormCellExport.hpp"

N2D2::Registrar<N2D2::BatchNormCellExport>
N2D2::CPP_TensorRT_BatchNormCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_BatchNormCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_BatchNormCellExport::mRegistrarType(
    BatchNormCell::Type, N2D2::CPP_TensorRT_BatchNormCellExport::getInstance);

void N2D2::CPP_TensorRT_BatchNormCellExport::generate(BatchNormCell& cell,
                                              const std::string& dirName)
{
    //N2D2::CPP_BatchNormCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_TensorRT_CellExport::generateHeaderIncludes(cell, header);
    CPP_BatchNormCellExport::generateHeaderConstants(cell, header);
    CPP_BatchNormCellExport::generateHeaderFreeParameters(cell, header);
    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);

    //CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
}

std::unique_ptr<N2D2::CPP_TensorRT_BatchNormCellExport>
N2D2::CPP_TensorRT_BatchNormCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_BatchNormCellExport>(new CPP_TensorRT_BatchNormCellExport);
}

void N2D2::CPP_TensorRT_BatchNormCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_BatchNormCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateBatchNormProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_BatchNormCellExport
    ::generateBatchNormProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);
    bool isActivated = false;

    if (cellFrame != NULL)
    {
        if(cellFrame->getActivation()) {
            std::string actType = cellFrame->getActivation()->getType();
            
            if(actType.compare("Linear") != 0) 
                isActivated = true;
        }    
    }
    /*    if(cellFrame->getActivation())
            isActivated = true;*/
    std::string activationStr = isActivated ?
                                    "LayerActivation(true, " + prefix + "_ACTIVATION_TENSORRT)"
                                    : "LayerActivation(false)";

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        input_name << parentsName[i] << "_";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_batchnorm(tsrRTHandles.netDef.back(),\n"
         << "       " << "tsrRTHandles.netBuilder,\n"
         << "       " << "mUseDLA,\n"
         << "       " << "pluginFactory,\n"
         << "       " << "tsrRTHandles.dT,\n"
         << "       " << "\"BatchNorm_NATIVE_" << identifier << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << "batchSize,\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_OUTPUTS_HEIGHT,\n"
         << "       " << prefix << "_OUTPUTS_WIDTH,\n"
         << "       " << input_name.str() << "tensor,\n"
         << "       " << identifier << "_scales,\n"
         << "       " << identifier << "_biases,\n"
         << "       " << identifier << "_means,\n"
         << "       " << identifier << "_variances,\n"
         << "       " << prefix << "_EPSILON);\n";
}

void N2D2::CPP_TensorRT_BatchNormCellExport
    ::generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog)
{

    prog << "   " << "CHECK_CUDA_STATUS( cudaMalloc(&inout_buffer["
                  << targetIdx + 1 << "], " // Added 1 for stride the input buffer
                  << "sizeof(DATA_T)*batchSize"
                  << "*NB_OUTPUTS[" << targetIdx << "]"
                  << "*OUTPUTS_HEIGHT[" << targetIdx << "]"
                  << "*OUTPUTS_WIDTH[" << targetIdx << "]"
                  << "));\n";
}

void N2D2::CPP_TensorRT_BatchNormCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "   " << "add_target(tsrRTHandles.netDef.back(), " << identifier << "_tensor, "
                  << targetIdx << ");\n";

}

