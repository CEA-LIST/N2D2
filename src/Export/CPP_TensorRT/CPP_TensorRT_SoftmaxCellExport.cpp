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

#include "Export/CPP_TensorRT/CPP_TensorRT_SoftmaxCellExport.hpp"

N2D2::Registrar<N2D2::SoftmaxCellExport>
N2D2::CPP_TensorRT_SoftmaxCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_SoftmaxCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_SoftmaxCellExport::mRegistrarType(
    SoftmaxCell::Type, N2D2::CPP_TensorRT_SoftmaxCellExport::getInstance);

void N2D2::CPP_TensorRT_SoftmaxCellExport::generate(SoftmaxCell& cell,
                                                 const std::string& dirName)
{
    CPP_SoftmaxCellExport::generate(cell, dirName);

}

std::unique_ptr<N2D2::CPP_TensorRT_SoftmaxCellExport>
N2D2::CPP_TensorRT_SoftmaxCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_SoftmaxCellExport>(new CPP_TensorRT_SoftmaxCellExport);
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateCellProgramDescriptors(Cell& cell,std::ofstream& prog)
{
    generateSoftMaxProgramTensorDesc(cell, prog);
    generateSoftMaxProgramLayerDesc(cell, prog);
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateSoftMaxProgramTensorDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::ITensor* " << identifier << "_tensor;\n";
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateSoftMaxProgramLayerDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::ISoftMaxLayer* " << identifier << "_layer;\n";
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateCellProgramInstanciateLayer( Cell& cell,
                                           std::vector<std::string>& parentsName,
                                           std::ofstream& prog)
{
    generateSoftMaxProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateSoftMaxProgramAddLayer(Cell& cell,
                                std::vector<std::string>& parentsName,
                                std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    const SoftmaxCell* softCell = dynamic_cast<SoftmaxCell*>(&cell);
    const unsigned int groupSize = softCell->getGroupSize();
    if(groupSize > 0)
    {
        std::cout << "-> Add a reshape layer on top of " << cell.getName()
            << " (Specific requirement for grouped-softmax on TensorRT)"<< std::endl;

        std::stringstream input_name;
        for(unsigned int i = 0; i < parentsName.size(); ++i)
            input_name << parentsName[i] << "_";

        prog << "   " << "std::vector< nvinfer1::ITensor *> "
            << identifier << "_reshape_tensor;\n";

        prog << "   " << identifier << "_reshape_tensor = " << "add_reshape(tsrRTHandles.netDef.back(),\n"
            << "       " << "tsrRTHandles.dT,\n"
            << "       " << "\"Reshape_NATIVE_" << prefix << "\",\n"
            << "       " << "GROUP_SIZE_" << prefix << ",\n"
            << "       " << "false,\n"
            << "       " << input_name.str() << "tensor);\n";

        prog << "   " << "std::vector< nvinfer1::ITensor *> "
            << identifier << "_softmax_tensor;\n";

        prog << "   " << identifier << "_softmax_tensor = " << "add_softmax(tsrRTHandles.netDef.back(),\n"
            << "       " << "tsrRTHandles.dT,\n"
            << "       " << "\"Softmax_NATIVE_" << prefix << "\",\n"
            << "       " << identifier << "_reshape_tensor);\n";

        prog << "   " << "std::vector< nvinfer1::ITensor *> "
            << identifier << "_tensor;\n";

        prog << "   " << identifier << "_tensor = " << "add_reshape(tsrRTHandles.netDef.back(),\n"
            << "       " << "tsrRTHandles.dT,\n"
            << "       " << "\"RestoreShape_NATIVE_" << prefix << "\",\n"
            << "       " << "1,\n"
            << "       " << "true,\n"
            << "       " << identifier << "_softmax_tensor);\n";

    }
    else
    {
        std::stringstream input_name;
        for(unsigned int i = 0; i < parentsName.size(); ++i)
            input_name << parentsName[i] << "_";

        prog << "   " << "std::vector< nvinfer1::ITensor *> "
            << identifier << "_tensor;\n";

        prog << "   " << identifier << "_tensor = " << "add_softmax(tsrRTHandles.netDef.back(),\n"
         << "       " << "tsrRTHandles.netBuilder,\n"
            << "       " << "tsrRTHandles.dT,\n"
            << "       " << "\"Softmax_NATIVE_" << prefix << "\",\n"
            << "       " << input_name.str() << "tensor);\n";
    }
}

void N2D2::CPP_TensorRT_SoftmaxCellExport
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

void N2D2::CPP_TensorRT_SoftmaxCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "   " << "add_target(tsrRTHandles.netDef.back(), " << identifier << "_tensor, "
                  << targetIdx << ");\n";

}
