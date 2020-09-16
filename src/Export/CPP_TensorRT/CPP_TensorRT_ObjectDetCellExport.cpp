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

#include "Export/CPP_TensorRT/CPP_TensorRT_ObjectDetCellExport.hpp"

N2D2::Registrar<N2D2::ObjectDetCellExport>
N2D2::CPP_TensorRT_ObjectDetCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_ObjectDetCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_ObjectDetCellExport::mRegistrarType(
    ObjectDetCell::Type, N2D2::CPP_TensorRT_ObjectDetCellExport::getInstance);

void N2D2::CPP_TensorRT_ObjectDetCellExport::generate(ObjectDetCell& cell,
                                              const std::string& dirName)
{
    N2D2::CPP_ObjectDetCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
}

std::unique_ptr<N2D2::CPP_TensorRT_ObjectDetCellExport>
N2D2::CPP_TensorRT_ObjectDetCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_ObjectDetCellExport>(new CPP_TensorRT_ObjectDetCellExport);
}

void N2D2::CPP_TensorRT_ObjectDetCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_ObjectDetCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateObjectDetProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_ObjectDetCellExport
    ::generateObjectDetProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;
    for(unsigned int k = 0; k < parentsName.size(); ++k)
        input_name << parentsName[k] << "_";

    prog << "   " << "std::vector<std::vector<nvinfer1::ITensor *>*> "
            <<  input_name.str() << "tensor_packed;\n";

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        prog << "   " << input_name.str() << "tensor_packed.push_back(&"
                        << parentsName[i] << "_tensor);\n";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_objectdetect("
         << "       " << "\"ObjectDet_GPU" << identifier << "\",\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_OUTPUTS_HEIGHT,\n"
         << "       " << prefix << "_OUTPUTS_WIDTH,\n"
         << "       " << prefix << "_CHANNELS_HEIGHT,\n"
         << "       " << prefix << "_CHANNELS_WIDTH,\n"
         << "         " << "ENV_SIZE_X,\n"
         << "         " << "ENV_SIZE_Y,\n"
         << "         " << prefix << "_FM_WIDTH,\n"
         << "         " << prefix << "_FM_HEIGHT,\n"
         << "       " << prefix << "_NB_PROPOSALS,\n"
         << "         " << prefix << "_NB_CLASS,\n"
         << "         " << prefix << "_NB_ANCHORS,\n"
         << "       " << input_name.str() << "tensor_packed,\n"
         << "         " << prefix << "_NMS_IUO_THRESHOLD,\n"
         << "         " << identifier << "_score_threshold,\n"
         << "         " << prefix << "_MAX_PARTS,\n"
         << "         " << prefix << "_MAX_TEMPLATES,\n"
         << "         " << prefix << "_PARTS,\n"
         << "         " << prefix << "_TEMPLATES,\n"
         << "         " << identifier << "_anchors[0][0]);\n";

}

void N2D2::CPP_TensorRT_ObjectDetCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";
}


