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

#include "Export/CPP_TensorRT/CPP_TensorRT_AnchorCellExport.hpp"

N2D2::Registrar<N2D2::AnchorCellExport>
N2D2::CPP_TensorRT_AnchorCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_AnchorCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_AnchorCellExport::mRegistrarType(
    AnchorCell::Type, N2D2::CPP_TensorRT_AnchorCellExport::getInstance);

void N2D2::CPP_TensorRT_AnchorCellExport::generate(AnchorCell& cell,
                                              const std::string& dirName)
{
    N2D2::CPP_AnchorCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
}

std::unique_ptr<N2D2::CPP_TensorRT_AnchorCellExport>
N2D2::CPP_TensorRT_AnchorCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_AnchorCellExport>(new CPP_TensorRT_AnchorCellExport);
}

void N2D2::CPP_TensorRT_AnchorCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_AnchorCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateAnchorProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_AnchorCellExport
    ::generateAnchorProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        input_name << parentsName[i] << "_";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_anchors("
         << "       " << "\"Anchor_GPU_" << identifier << "\",\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_OUTPUTS_HEIGHT,\n"
         << "       " << prefix << "_OUTPUTS_WIDTH,\n"
         << "       " << input_name.str() << "tensor,\n"
         << "         " << "ENV_SIZE_Y,\n"
         << "         " << "ENV_SIZE_X,\n"
         << "         " << prefix << "_FM_WIDTH,\n"
         << "         " << prefix << "_FM_HEIGHT,\n"
         << "         " << prefix << "_SCORE_CLS,\n"
         << "         " << prefix << "_IS_COORD_ANCHORS,\n"
         << "         " << prefix << "_FLIP,\n"
         << "         " << prefix << "_NB_ANCHORS,\n"
         << "         " << identifier << "_anchors[0]);\n";
}

void N2D2::CPP_TensorRT_AnchorCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";

}


