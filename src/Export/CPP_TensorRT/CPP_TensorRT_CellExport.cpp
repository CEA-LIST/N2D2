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

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Export/CPP_TensorRT/CPP_TensorRT_CellExport.hpp"

void N2D2::CPP_TensorRT_CellExport::generateHeaderIncludes(Cell& cell,
                                                        std::ofstream& header)
{
    CPP_CellExport::generateHeaderIncludes(cell, header);

}

void N2D2::CPP_TensorRT_CellExport::
    generateHeaderTensorRTConstants(Cell& cell,std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL) {
        if(cellFrame->getActivation()) {
            const std::string actType = cellFrame->getActivation()->getType();

            if(actType != "Linear")  {
                double alpha = 0.0;
                double beta = 0.0;
                header << "#define " << prefix << "_ACTIVATION_TENSORRT ";

                header << "nvinfer1::ActivationType::"
                    << ((actType == "Rectifier") ? "kRELU" :
                            (actType == "Logistic") ? "kSIGMOID" :
                            (actType == "LogisticWithLoss") ? "kSIGMOID" :
                            (actType == "Tanh") ? "kTANH": 
                            (actType == "SoftPlus") ? "kSOFTPLUS " : "")
                    << "\n";

                if(actType == "Rectifier")  {
                    alpha = cellFrame->getActivation()
                        ->getParameter<double>("LeakSlope");
                    beta = cellFrame->getActivation()
                        ->getParameter<double>("Clipping");
                }
                if(actType == "SoftPlus")  {
                    alpha = 1.0;
                    beta = 1.0;
                }
                header << "#define " << prefix << "_ALPHA_ACTIVATION_TENSORRT " 
                    << alpha << "\n";
                header << "#define " << prefix << "_BETA_ACTIVATION_TENSORRT " 
                    << beta << "\n";
            }
        }
    }

}

void N2D2::CPP_TensorRT_CellExport
            ::generateProgramAddActivation(Cell& cell, std::ofstream& prog)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    const Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(&cell);

    if (cellFrame != NULL)
    {
        if(cellFrame->getActivation())
        {
            std::string actType = cellFrame->getActivation()->getType();
            prog << "   auto activation_" << identifier
                << " = netDef->addActivation(*"
                << identifier << "->getOutput(0), "
                << prefix << "_ACTIVATION_TENSORRT);\n";

            prog << "   " << "assert(activation_" << identifier
                << " != nullptr);\n";

            prog << "   nvinfer1::ITensor *" << identifier << "_tensor = "
                << "activation_" << identifier << "->getOutput(0);\n";

            prog << "   activation_" << identifier << "->setName(\""
                 << prefix << "_ACTIVATION\");\n";
        }
        else {
            prog << "   nvinfer1::ITensor *" << identifier << "_tensor = "
                << identifier << "->getOutput(0);\n";
        }
    }
    prog << "\n";
}
void N2D2::CPP_TensorRT_CellExport
            ::generateTensor(Cell& /*cell*/,
                               std::vector<std::string>& parentsName,
                               std::ofstream& prog)
{
    std::stringstream concatName;

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        concatName << parentsName[i] << "_";

    prog << "   " << "std::vector<std::vector<nvinfer1::ITensor *>*> "
            <<  concatName.str() << "in_tensor;\n";

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        prog << "   " << concatName.str() << "in_tensor.push_back(&"
                        << parentsName[i] << "_tensor);\n";


/*
    prog << "   " << "std::vector<nvinfer1::ITensor *> const "
            <<  concatName.str() << "in_tensor[" << parentsName.size() << "]= "
            << " {" << parentsName[0] << "_tensor";

    for(unsigned int i = 1; i < parentsName.size(); ++i)
        prog << ", " << parentsName[i] << "_tensor";
    prog << "};\n";
*/

/*
    prog << "   " << "nvinfer1::ITensor *const " << concatName.str() << "in_tensor["
         << parentsName.size() << "] = {" << parentsName[0] << "_tensor";

    for(unsigned int i = 1; i < parentsName.size(); ++i)
        prog << ", " << parentsName[i] << "_tensor";
    prog << "};\n";
*/
}

void N2D2::CPP_TensorRT_CellExport
            ::generateAddConcat(Cell& /*cell*/,
                                std::vector<std::string>& parentsName,
                                std::ofstream& prog)
{
    std::stringstream concatName;

    for(unsigned int i = 0; i < parentsName.size(); ++i)
        concatName << parentsName[i] << "_";


    prog << "   " << "std::vector< nvinfer1::ITensor *> "
         << concatName.str() << "tensor;\n";

    prog << "   " << concatName.str() << "tensor = " << "add_concat("
         << "       " << "\"Concat_NATIVE_" << concatName.str() << "\",\n"
         << "       " << parentsName.size() << ",\n"
         << "       " << concatName.str() << "in_tensor);\n";

/*
    prog << "   " << "auto " << concatName.str() << "= \n"
         << "   " << "netDef->addConcatenation("
         << concatName.str() << "in_tensor, "
         << parentsName.size() << ");\n";
    prog << "   " << "auto " << concatName.str() << "tensor = "
         << concatName.str() << "->getOutput(0);\n";
*/

}

