/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Export/DeepNetExport.hpp"
#include "DeepNet.hpp"
#include "N2D2.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/PaddingCell.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Export/CellExport.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"

bool N2D2::DeepNetExport::mUnsignedData = true;
bool N2D2::DeepNetExport::mEnvDataUnsigned = false;
std::string N2D2::DeepNetExport::mExportParameters = "";

void N2D2::DeepNetExport::generate(DeepNet& deepNet,
                                   const std::string& dirName,
                                   const std::string& type)
{
    if (!Registrar<DeepNetExport>::exists(type)) {
        std::cout << Utils::cwarning << "Error: \"" << type << "\" export"
                  << " is not available (additional modules may be required)"
                  << Utils::cdef << std::endl;
        return;
    }

    std::cout << "Generating " << type << " export to \"" << dirName << "\":"
        << std::endl;

    Utils::createDirectories(dirName);

// Copy export core sources
#ifdef WIN32
    const std::string cmd = "XCOPY /E /Y \""
                            + std::string(N2D2_PATH("export/" + type)) + "\" \""
                            + dirName + "\"";
#else
    const std::string cmd = "cp -R -L "
                            + std::string(N2D2_PATH("export/" + type + "/*"))
                            + " " + dirName;
#endif
    const int ret = Utils::exec(cmd);

    if (ret != 0) {
        throw std::runtime_error("Warning: could not import files for " + type + " export "
                                 "(copy return code: " + std::to_string(ret) + ").");
    }

    std::cout << "-> Generating network" << std::endl;
    Registrar<DeepNetExport>::create(type)(deepNet, dirName);

    std::cout << "Done!" << std::endl;
}

void N2D2::DeepNetExport::generateCells(const DeepNet& deepNet,
                                        const std::string& dirName,
                                        const std::string& type)
{
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it)
        {
            CellExport::generate(*deepNet.getCell(*it), dirName, type);
        }
    }
}

void N2D2::DeepNetExport::setExportParameters(
    const std::string& exportParameters)
{
    if (!exportParameters.empty()
        && !std::ifstream(exportParameters.c_str()).good())
    {
        throw std::runtime_error("DeepNetExport::setExportParameters(): "
            "unable to read export parameter file: " + exportParameters);
    }

    mExportParameters = exportParameters;
}

std::string N2D2::DeepNetExport::getLayerName(DeepNet& deepNet,
                                              const std::vector
                                              <std::string>& layer)
{
    std::stringstream layerNameStr;

    for (std::vector<std::string>::const_iterator it = layer.begin(),
                                                  itEnd = layer.end();
         it != itEnd;
         ++it) {
        const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

        if (!layerNameStr.str().empty())
            layerNameStr << "_";

        layerNameStr << cell->getName();
    }

    return Utils::CIdentifier(layerNameStr.str());
}

bool N2D2::DeepNetExport::isSharedOutput(DeepNet& deepNet,
                                         const unsigned int layerNumber,
                                         const unsigned int cellNumber)
{
    bool isShared = false;

    if (layerNumber < deepNet.getLayers().size() - 1) {
        const std::vector<std::string>& nextLayer
            = deepNet.getLayer(layerNumber + 1);
        const std::vector<std::string>& currentLayer
            = deepNet.getLayer(layerNumber);
        Cell& cell = *deepNet.getCell(currentLayer[cellNumber]);

        for (std::vector<std::string>::const_iterator it = nextLayer.begin(),
                                                      itEnd = nextLayer.end();
             it != itEnd;
             ++it) {
            const std::vector<std::shared_ptr<Cell> > parentCells
                = deepNet.getParentCells(*it);

            if (parentCells.size() > 1) {
                for (unsigned int i = 1; i < parentCells.size(); ++i) {

                    if (cell.getName() == (*parentCells[i]).getName())
                        isShared = true;
                }
            }
        }
    }
    return isShared;
}

bool N2D2::DeepNetExport::isSharedInput(DeepNet& deepNet,
                                        const unsigned int layerNumber,
                                        const unsigned int cellNumber)
{
    bool isShared = false;
    const std::vector<std::string>& currentLayer
        = deepNet.getLayer(layerNumber);

    const std::vector<std::shared_ptr<Cell> > parentCurrentCells
        = deepNet.getParentCells(currentLayer[cellNumber]);

    if ((currentLayer.size() - 1) > cellNumber) {
        for (unsigned int i = cellNumber + 1; i < currentLayer.size(); ++i) {

            const std::vector<std::shared_ptr<Cell> > parentNextCell
                = deepNet.getParentCells(currentLayer[i]);

            if ((*parentCurrentCells[0]).getName()
                == (*parentNextCell[0]).getName())
                isShared = true;
        }
    }
    return isShared;
}

std::string
N2D2::DeepNetExport::getCellInputName(DeepNet& deepNet,
                                      const unsigned int layerNumber,
                                      const unsigned int cellNumber)
{
    std::string buffer_name;
    std::stringstream tmp;

    if (layerNumber == 1)
        tmp << "in_";
    else {
        const std::vector<std::string>& layer = deepNet.getLayer(layerNumber);
        const std::vector<std::shared_ptr<Cell> > parentCells
            = deepNet.getParentCells(layer[cellNumber]);
        for (unsigned int i = 0; i < parentCells.size(); ++i) {
            std::string prefix = parentCells.at(i)->getName();

            tmp << prefix + "_";
        }
    }
    buffer_name = tmp.str();

    return Utils::CIdentifier(buffer_name);
}

std::string
N2D2::DeepNetExport::getCellOutputName(DeepNet& deepNet,
                                       const unsigned int layerNumber,
                                       const unsigned int cellNumber)
{
    std::string buffer_name;
    std::stringstream tmp;
    bool isHandle = true;
    const std::vector<std::string>& inputLayer
        = deepNet.getLayers().at(layerNumber);
    const std::shared_ptr<Cell> inputCell
        = deepNet.getCell(inputLayer.at(cellNumber));

    const std::string cellName = inputCell->getName();

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);

        if(cell->getName() == cellName)
            return "";
    }

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + layerNumber + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        for (std::vector<std::string>::const_iterator itCell
             = (*itLayer).begin(),
             itEnd = (*itLayer).end();
             itCell != itEnd;
             ++itCell) {
            const std::vector<std::shared_ptr<Cell> > parentCells
                = deepNet.getParentCells(*itCell);

            if (isHandle) {
                tmp.str(std::string());
                for (unsigned int i = 0; i < parentCells.size(); ++i) {
                    std::string prefix = parentCells.at(i)->getName();

                    tmp << prefix + "_";

                    if (prefix == cellName)
                        isHandle = false;
                }
            }
        }
    }
    buffer_name = tmp.str();

    return Utils::CIdentifier(buffer_name);
}

std::vector<unsigned int>
N2D2::DeepNetExport::getMapLayer(DeepNet& deepNet,
                                 const unsigned int layerNumber)
{
    std::vector<unsigned int> mapping;

    if (layerNumber < deepNet.getLayers().size() - 1) {
        const std::vector<std::string>& nextLayer
            = deepNet.getLayer(layerNumber + 1);

        for (std::vector<std::string>::const_iterator it = nextLayer.begin(),
                                                      itEnd = nextLayer.end();
             it != itEnd;
             ++it) {
            const std::vector<std::shared_ptr<Cell> > parentCells
                = deepNet.getParentCells(*it);

            mapping.push_back(parentCells.size());
        }
    } else
        mapping.push_back(1); // for the last layer (target)

    return mapping;
}

bool N2D2::DeepNetExport::isCellInputsUnsigned(const Cell& cell) {
    if (CellExport::mPrecision <= 0 || !DeepNetExport::mUnsignedData) {
        // Unsigned cells are not allowed
        return false;
    }

    const std::vector<std::shared_ptr<Cell>>& parentsCells = cell.getParentsCells();
    
    bool unsignedInputs = false;
    for (auto it = parentsCells.begin(); it != parentsCells.end(); ++it) {
        const bool unsignedInput = (*it)?isCellOutputUnsigned(**it):DeepNetExport::mEnvDataUnsigned;

        if (it == parentsCells.begin()) {
            unsignedInputs = unsignedInput;
        }
        else if (unsignedInput != unsignedInputs) {
            throw std::runtime_error("Unsupported: cell " + cell.getName() + 
                                     " mixes signed and unsigned inputs. "
                                     "Try setting DeepNetExport::mUnsignedData to false");
        }
    }

    return unsignedInputs;
}

bool N2D2::DeepNetExport::isCellOutputUnsigned(const Cell& cell)  {
    if (CellExport::mPrecision <= 0 || !DeepNetExport::mUnsignedData) {
        // Unsigned cells are not allowed
        return false;
    }

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    if (cell.getType() == PaddingCell::Type) {
        // Special case for PaddingCell because getOutputsRange() throw an
        // exception if the parent cell is the env (which can happen for 
        // padding).
        // FIXME: find a better generic alternative...
        return isCellInputsUnsigned(cell);
    }
    else {
        return cellFrame.getOutputsRange().first >= 0.0;
    }
}