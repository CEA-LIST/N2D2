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

    generateDeepNetProgram(
        deepNet, "N2D2::Network::", dirName + "/dnn/src/network.cpp");

    generateStimuliCalib(deepNet, dirName);
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
    generateIncludes(deepNet, prog);
    generateProgramInitNetwork(deepNet, name, prog);
}

void N2D2::CPP_TensorRT_DeepNetExport::generateIncludes(DeepNet& deepNet,
                                                        std::ofstream& prog)
{
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();
    prog << "#include \"../include/env.hpp\n";

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);
            prog << "#include \"../include/" << Utils::CIdentifier(cell->getName())
                << ".hpp\"\n";
        }
    }
}



void N2D2::CPP_TensorRT_DeepNetExport::generateProgramBegin(DeepNet& /*deepNet*/,
                                                 std::ofstream& prog)
{
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    // Program
    prog << "// N2D2 auto-generated file.\n"
            "// @ " << std::asctime(localNow)
         << "\n" // std::asctime() already appends end of line
            "#include \"../../include/NetworkTensorRT.hpp\"\n"
            "\n";

}

void N2D2::CPP_TensorRT_DeepNetExport
    ::generateProgramInitNetwork(DeepNet& deepNet,
                                 const std::string& name,
                                 std::ofstream& prog)
{
    std::vector<std::string>  pNameFactory;
    const std::vector<std::shared_ptr<Target> > 
            outputTargets =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();

    prog << "void " << name << "networkDefinition() {\n";
    const std::vector<std::vector<std::string> >& 
            layers = deepNet.getLayers();

    prog << "//Initialization of the network input (Float32):\n";

    prog << "   std::vector<nvinfer1::ITensor *> in_tensor;\n";
    prog << "   in_tensor.push_back(mNetDef.back()->addInput(\"ENV_INPUT\","
         << " nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{ENV_NB_OUTPUTS,"
         << " ENV_SIZE_Y, ENV_SIZE_X}));\n"
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