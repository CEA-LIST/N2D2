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

#include "Export/CPP_TensorRT/CPP_TensorRT_ROIPoolingCellExport.hpp"

N2D2::Registrar<N2D2::ROIPoolingCellExport>
N2D2::CPP_TensorRT_ROIPoolingCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_ROIPoolingCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_ROIPoolingCellExport::mRegistrarType(
    ROIPoolingCell::Type, N2D2::CPP_TensorRT_ROIPoolingCellExport::getInstance);

void N2D2::CPP_TensorRT_ROIPoolingCellExport::generate(ROIPoolingCell& cell,
                                              const std::string& dirName)
{
    N2D2::CPP_ROIPoolingCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
}

std::unique_ptr<N2D2::CPP_TensorRT_ROIPoolingCellExport>
N2D2::CPP_TensorRT_ROIPoolingCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_ROIPoolingCellExport>(new CPP_TensorRT_ROIPoolingCellExport);
}

void N2D2::CPP_TensorRT_ROIPoolingCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_ROIPoolingCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateROIPoolingProgramAddLayer(cell, parentsName, prog);
}


void N2D2::CPP_TensorRT_ROIPoolingCellExport
    ::generateROIPoolingProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;

    const unsigned int nbInputFeatures = parentsName.size() - 1;

    prog << "   unsigned int " << identifier << "_channels["
         << nbInputFeatures << "] = \n"
         << "         " << "{";

    for(unsigned int k = 1; k < parentsName.size(); ++k)
    {
        prog << Utils::upperCase(parentsName[k]) << "_NB_OUTPUTS";
        if(k < nbInputFeatures)
            prog << " ,\n         ";
        else
            prog << "};\n";
    }

    prog << "   unsigned int " << identifier << "_height["
         << nbInputFeatures << "] = \n"
         << "         " << "{";

    for(unsigned int k = 1; k < parentsName.size(); ++k)
    {
        prog << Utils::upperCase(parentsName[k]) << "_OUTPUTS_HEIGHT";
        if(k < nbInputFeatures)
            prog << " ,\n         ";
        else
            prog << "};\n";
    }


    prog << "   unsigned int " << identifier << "_width["
         << nbInputFeatures << "] = \n"
         << "         " << "{";

    for(unsigned int k = 1; k < parentsName.size(); ++k)
    {
        prog << Utils::upperCase(parentsName[k]) << "_OUTPUTS_WIDTH";
        if(k < nbInputFeatures)
            prog << " ,\n         ";
        else
            prog << "};\n";
    }

    for(unsigned int k = 0; k < parentsName.size(); ++k)
        input_name << parentsName[k] << "_";
/*
    prog << "   std::vector< nvinfer1::ITensor *>const " << input_name.str()
         << "tensor[" << parentsName.size() << "] = {"
         << parentsName[0] << "_tensor,\n";


    for(unsigned int k = 1; k < parentsName.size(); ++k)
        {
            prog << parentsName[k] << "_tensor";

            if( k < parentsName.size()-1)
                prog << ",\n";
        }
    prog << "};\n";*/
    prog << "   " << "std::vector<std::vector<nvinfer1::ITensor *>*> "
            <<  input_name.str() << "tensor;\n";

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        prog << "   " << input_name.str() << "tensor.push_back(&"
                        << parentsName[i] << "_tensor);\n";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_ROIpooling(tsrRTHandles.netDef.back(),\n"
         << "       " << "pluginFactory,\n"
         << "       " << "tsrRTHandles.dT,\n"
         << "       " << "\"ROIPooling_GPU_" << identifier << "\",\n"
         << "       " << "batchSize,\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_OUTPUTS_HEIGHT,\n"
         << "       " << prefix << "_OUTPUTS_WIDTH,\n"
         << "       " << input_name.str() << "tensor,\n"
         << "         " << "ENV_SIZE_Y,\n"
         << "         " << "ENV_SIZE_X,\n"
         << "         " << nbInputFeatures << ",\n"
         << "         " << "&" << identifier << "_channels[0],\n"
         << "         " << "&" << identifier << "_height[0],\n"
         << "         " << "&" << identifier << "_width[0],\n"
         << "         " << prefix << "_POOLING_TYPE,\n"
         << "         " << prefix << "_PARENT_PROPOSALS,\n"
         << "         " << prefix << "_IGNORE_PADDING,\n"
         << "         " << prefix << "_FLIP);\n";

}

void N2D2::CPP_TensorRT_ROIPoolingCellExport
    ::generateCellProgramAllocateMemory(unsigned int targetIdx, std::ofstream& prog)
{
    prog << "   " << "CHECK_CUDA_STATUS( cudaMalloc(&inout_buffer["
                  << targetIdx + 1 << "], " // Added 1 for stride the input buffer
                  << "sizeof(DATA_T)*batchSize"
                  //<< "*" << prefix << "_PARENT_PROPOSALS"
                  << "*NB_OUTPUTS[" << targetIdx << "]"
                  << "*OUTPUTS_HEIGHT[" << targetIdx << "]"
                  << "*OUTPUTS_WIDTH[" << targetIdx << "]"
                  << "));\n";
}

void N2D2::CPP_TensorRT_ROIPoolingCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    prog << "   " << "add_target(tsrRTHandles.netDef.back(), " << identifier << "_tensor, "
                  << targetIdx << ");\n";

}




