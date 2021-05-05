/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Export/CPP_TensorRT/CPP_TensorRT_ActivationCellExport.hpp"

N2D2::Registrar<N2D2::ActivationCellExport>
N2D2::CPP_TensorRT_ActivationCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_ActivationCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_ActivationCellExport::mRegistrarType(
    ActivationCell::Type, N2D2::CPP_TensorRT_ActivationCellExport::getInstance);

void N2D2::CPP_TensorRT_ActivationCellExport::generate(ActivationCell& cell,
                                              const std::string& dirName)
{
    N2D2::CPP_ActivationCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
}

std::unique_ptr<N2D2::CPP_TensorRT_ActivationCellExport>
N2D2::CPP_TensorRT_ActivationCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_ActivationCellExport>(new CPP_TensorRT_ActivationCellExport);
}

void N2D2::CPP_TensorRT_ActivationCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_ActivationCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    generateActivationProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_ActivationCellExport
    ::generateActivationProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog)
{

    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;
    for(unsigned int k = 0; k < parentsName.size(); ++k)
        input_name << parentsName[k] << "_";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);
    bool isActivated = false;

    if (cellFrame != NULL) {
        if(cellFrame->getActivation()) {
            const std::string actType = cellFrame->getActivation()->getType();

            if(actType != "Linear")
                isActivated = true;
        }
    }

    std::string activationStr = isActivated ?
                                    "LayerActivation(true, " 
                                    + prefix + "_ACTIVATION_TENSORRT, " 
                                    + prefix + "_ALPHA_ACTIVATION_TENSORRT, "
                                    + prefix + "_BETA_ACTIVATION_TENSORRT)"
                                    : "LayerActivation(false)";

    prog << "   " << identifier << "_tensor = " << "add_activation_cell("
         <<"\"Activation_" << identifier << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << input_name.str() << "tensor"
         << ");\n";

}

void N2D2::CPP_TensorRT_ActivationCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";
}


