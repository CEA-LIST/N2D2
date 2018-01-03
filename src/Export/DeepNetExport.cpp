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
    const int ret = system(cmd.c_str());

    if (ret < 0) {
        std::cout << Utils::cwarning << "Warning: could not import files for "
                  << type << " export "
                             "(return code: " << ret << ")" << Utils::cdef
                  << std::endl;
    }

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
            std::cout << "-> Generating cell " << (*it) << std::endl;
            CellExport::generate(*deepNet.getCell(*it), dirName, type);
        }
    }

    std::cout << "-> Generating network" << std::endl;
    Registrar<DeepNetExport>::create(type)(deepNet, dirName);

    std::cout << "Done!" << std::endl;
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

bool N2D2::DeepNetExport::isCellUnsigned(DeepNet& deepNet, Cell& cell)
{
    if (CellExport::mPrecision <= 0 || !DeepNetExport::mUnsignedData) {
        // Unsigned cells are not allowed
        return false;
    }

    const std::vector<std::shared_ptr<Cell> > parentCells
        = deepNet.getParentCells(cell.getName());
    bool unsignedInputs = false;

    for (std::vector<std::shared_ptr<Cell> >::const_iterator it
            = parentCells.begin(), itBegin = parentCells.begin(),
        itEnd = parentCells.end(); it != itEnd; ++it)
    {
        bool unsignedInput = false;

        if (!(*it)) {
            // Parent is the environment
            unsignedInput = DeepNetExport::mEnvDataUnsigned;
        }
        else {
            const std::shared_ptr<Cell_Frame_Top> cellFrame
                = std::dynamic_pointer_cast<Cell_Frame_Top>(*it);

            if (cellFrame) {
                if (cellFrame->getActivation()
                    && cellFrame->getActivation()->getType()
                        == std::string("Rectifier"))
                {
                    // Rectifier is unsigned
                    unsignedInput = true;
                }

                if ((*it)->getType() == PoolCell::Type
                    && !cellFrame->getActivation())
                {
                    // PoolCell without activation (linear):
                    // Its output is unsigned if the cell is unsigned
                    // (i.e. has unsigned inputs)
                    unsignedInput = isCellUnsigned(deepNet, *(*it));
                }
            }
        }

        if (it == itBegin)
            unsignedInputs = unsignedInput;
        else if (unsignedInput != unsignedInputs) {
            throw std::runtime_error("Unsupported: cell " + cell.getName()
                                     + " mixes signed and unsigned inputs."
                                     " Try setting DeepNetExport::mUnsignedData"
                                     " to false");
        }
    }

    return unsignedInputs;
}
