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

#include "Export/CPP_TensorRT/CPP_TensorRT_PoolCellExport.hpp"

N2D2::Registrar<N2D2::PoolCellExport>
N2D2::CPP_TensorRT_PoolCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_PoolCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_PoolCellExport::mRegistrarType(
    PoolCell::Type, N2D2::CPP_TensorRT_PoolCellExport::getInstance);

void N2D2::CPP_TensorRT_PoolCellExport::generate(PoolCell& cell,
                                              const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_TensorRT_CellExport::generateHeaderIncludes(cell, header);
    CPP_PoolCellExport::generateHeaderConstants(cell, header);
    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
    generateHeaderTensorRTConstants(cell, header);
    CPP_PoolCellExport::generateHeaderConnections(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_TensorRT_PoolCellExport
        ::generateHeaderTensorRTConstants(PoolCell& cell, std::ofstream& header)
{
    std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    std::string poolType;

    if(cell.getPooling() == N2D2::PoolCell::Max)
        poolType = "MAX";
    else if (cell.getPooling() == N2D2::PoolCell::Average)
        poolType = "AVERAGE";

    header << "#define "
        << prefix << "_POOLING_TENSORRT nvinfer1::PoolingType::k" << poolType
        << "\n\n";
}


std::unique_ptr<N2D2::CPP_TensorRT_PoolCellExport>
N2D2::CPP_TensorRT_PoolCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_PoolCellExport>(new CPP_TensorRT_PoolCellExport);
}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generateCellProgramDescriptors(Cell& cell,std::ofstream& prog)
{
    generatePoolProgramTensorDesc(cell, prog);
    generatePoolProgramLayerDesc(cell, prog);
}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generatePoolProgramTensorDesc(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::ITensor* " << identifier << "_tensor;\n";
}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generatePoolProgramLayerDesc(Cell& cell,std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "nvinfer1::IPoolingLayer* " << identifier << "_layer;\n";
}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generateCellProgramInstanciateLayer( Cell& cell,
                                           std::vector<std::string>& parentsName,
                                           std::ofstream& prog)
{
    generatePoolProgramAddLayer(cell, parentsName, prog);
}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generatePoolProgramAddLayer(Cell& cell,
                                std::vector<std::string>& parentsName,
                                std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);
    std::stringstream input_name;

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        input_name << parentsName[i] << "_";

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
                                    "LayerActivation(true, " + prefix + "_ACTIVATION_TENSORRT)"
                                    : "LayerActivation(false)";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << "_tensor;\n";

    prog << "   " << identifier << "_tensor = " << "add_pooling(\n"
         << "       " << "\"Pooling_NATIVE_" << identifier << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << prefix << "_POOL_HEIGHT,\n"
         << "       " << prefix << "_POOL_WIDTH,\n"
         << "       " << prefix << "_STRIDE_X,\n"
         << "       " << prefix << "_STRIDE_Y,\n"
         << "       " << prefix << "_PADDING_X,\n"
         << "       " << prefix << "_PADDING_Y,\n"
         << "       " << input_name.str() << "tensor,\n"
         << "       " << prefix << "_POOLING_TENSORRT);\n";

}

void N2D2::CPP_TensorRT_PoolCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";
}
