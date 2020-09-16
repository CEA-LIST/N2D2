/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Export/CPP_TensorRT/CPP_TensorRT_ElemWiseCellExport.hpp"

N2D2::Registrar<N2D2::ElemWiseCellExport>
N2D2::CPP_TensorRT_ElemWiseCellExport::mRegistrar(
    "CPP_TensorRT", N2D2::CPP_TensorRT_ElemWiseCellExport::generate);

N2D2::Registrar<N2D2::CPP_TensorRT_CellExport>
N2D2::CPP_TensorRT_ElemWiseCellExport::mRegistrarType(
    ElemWiseCell::Type, N2D2::CPP_TensorRT_ElemWiseCellExport::getInstance);

void N2D2::CPP_TensorRT_ElemWiseCellExport::generate(ElemWiseCell& cell,
                                              const std::string& dirName)
{
    //N2D2::CPP_ElemWiseCellExport::generate(cell, dirName);
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str(), std::ios::app);

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    CPP_ElemWiseCellExport::generateHeaderConstants(cell, header);
    generateHeaderTensorRTConstants(cell, header);
    CPP_TensorRT_CellExport::generateHeaderTensorRTConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);

}

void N2D2::CPP_TensorRT_ElemWiseCellExport
        ::generateHeaderTensorRTConstants(ElemWiseCell& cell, std::ofstream& header)
{
    std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    std::string opType;

    if(cell.getOperation() == N2D2::ElemWiseCell::Sum)
        opType = "SUM";
    else if (cell.getOperation() == N2D2::ElemWiseCell::Prod)
        opType = "PROD";
    else if (cell.getOperation() == N2D2::ElemWiseCell::Max)
        opType = "MAX";


    header << "#define "
        << prefix << "_ELEM_OP_TENSORRT nvinfer1::ElementWiseOperation::k" << opType
        << "\n\n";
}

std::unique_ptr<N2D2::CPP_TensorRT_ElemWiseCellExport>
N2D2::CPP_TensorRT_ElemWiseCellExport::getInstance(Cell& /*cell*/)
{
    return std::unique_ptr
        <CPP_TensorRT_ElemWiseCellExport>(new CPP_TensorRT_ElemWiseCellExport);
}

void N2D2::CPP_TensorRT_ElemWiseCellExport
    ::generateCellProgramDescriptors(Cell&/* cell*/, std::ofstream& /*prog*/)
{

}

void N2D2::CPP_TensorRT_ElemWiseCellExport
    ::generateCellProgramInstanciateLayer(Cell& cell,
                                          std::vector<std::string>& parentsName,
                                          std::ofstream& prog)
{
    if(parentsName.size() > 2)
    {
        //throw std::runtime_error("CPP_TensorRT_ElemWiseCellExport.cpp:generateCellProgramInstanciateLayer()::"
        //                         " TensorRT ElementWise support only two inputs layers in " + cell.getName());
        const bool isOdd =  parentsName.size() % 2;
        if( !isOdd)
        {   
            const std::vector<std::string> initialParentsName = parentsName;       
            std::vector<std::string> variableParentsName = parentsName;

            size_t nbStage = (size_t) std::floor( (float)parentsName.size() / 4.0f);

            for(unsigned int stage = 0; stage < nbStage; ++stage)
            {
                const size_t nbOP = (size_t) std::floor( (float)variableParentsName.size() 
                                                            / 2.0f);

                for(unsigned int op = 0; op < nbOP; ++op)
                {
                    std::string adderStr = "_" 
                                            + std::to_string(stage) + "_" 
                                            + std::to_string(op);

                    std::vector<std::string> pName(2);

                    std::copy(variableParentsName.begin() + op*2, 
                              variableParentsName.begin() + op*2 + 2,
                              pName.begin());

                    generateElemWiseProgramAddLayer( cell, 
                                                     pName, 
                                                     prog, 
                                                     adderStr);

                }
                variableParentsName.resize(initialParentsName.size() / ((stage + 1)*2));

                for(unsigned int op = 0; op < variableParentsName.size(); ++op)
                {
                    variableParentsName[op] = Utils::CIdentifier(cell.getName()) 
                                                    + "_" + std::to_string(stage) + "_" 
                                                    + std::to_string(op);
                }

            }
            generateElemWiseProgramAddLayer( cell, 
                                             variableParentsName, 
                                             prog);


        }
        else
        {
            throw std::runtime_error("CPP_TensorRT_ElemWiseCellExport.cpp:generateCellProgramInstanciateLayer()::"
                                    " TensorRT ElementWise support only pair inputs layers in " + cell.getName());
        }

    }
    else
    {
        generateElemWiseProgramAddLayer(cell, parentsName, prog);
    }
}

void N2D2::CPP_TensorRT_ElemWiseCellExport
    ::generateElemWiseProgramAddLayer(Cell& cell,
                                   std::vector<std::string>& parentsName,
                                   std::ofstream& prog,
                                   std::string adderTreeStr)
{

    const std::string identifier = Utils::CIdentifier(cell.getName()) ;
    const std::string prefix = Utils::upperCase(identifier);
    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);
    bool isActivated = false;
    std::stringstream input_name;

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

    for(unsigned int k = 0; k < parentsName.size(); ++k)
        input_name << parentsName[k] << "_";

    prog << "   " << "std::vector<std::vector<nvinfer1::ITensor *>> "
            <<  input_name.str()
            << identifier << adderTreeStr << "_tensor_agregate;\n";

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        prog << "   " << input_name.str()
                    << identifier << adderTreeStr << "_tensor_agregate.push_back("
                    << parentsName[i] << "_tensor);\n";

    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << identifier << adderTreeStr << "_tensor;\n";

    prog << "   " << identifier << adderTreeStr << "_tensor = " << "add_elementwise(\n"
         << "       " << "\"ElemWise_NATIVE_" << identifier << adderTreeStr << "\",\n"
         << "       " << activationStr << ",\n"
         << "       " << prefix << "_NB_OUTPUTS,\n"
         << "       " << prefix << "_OUTPUTS_HEIGHT,\n"
         << "       " << prefix << "_OUTPUTS_WIDTH,\n"
         << "       " << input_name.str()
                        << identifier << adderTreeStr << "_tensor_agregate,\n"
         << "         " << prefix << "_ELEM_OP_TENSORRT,\n"
         << "       " << prefix << "_WEIGHTS,\n"
         << "       " << prefix << "_SHIFTS,\n"
         << "         " << prefix << "_POWER);\n";

}

void N2D2::CPP_TensorRT_ElemWiseCellExport
    ::generateCellProgramInstanciateOutput(Cell& cell,
                                           unsigned int targetIdx,
                                           std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    prog << "   " << "add_target(" << identifier << "_tensor, "
                  << targetIdx << ");\n";

}


