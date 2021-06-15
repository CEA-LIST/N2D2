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

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include "DeepNet.hpp"
#include "DrawNet.hpp"
#include "StimuliProvider.hpp"
#include "Cell/ConvCell.hpp"
#include "Cell/FcCell.hpp"
#include "Cell/PoolCell.hpp"
#include "Cell/ElemWiseCell.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Target/TargetScore.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/CPP/CPP_Config.hpp"
#include "Export/CPP/CPP_DeepNetExport.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"
#include "utils/IniParser.hpp"
#include "utils/Registrar.hpp"

N2D2::Registrar<N2D2::DeepNetExport>
N2D2::CPP_DeepNetExport::mRegistrar(
    {"CPP", "CPP_ASMP", "CPP_STM32", "CPP_HLS"},
    N2D2::CPP_DeepNetExport::generate);

void N2D2::CPP_DeepNetExport::generate(DeepNet& deepNet,
                                       const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");
    Utils::createDirectories(dirName + "/dnn/src");

    generateParamsHeader(dirName + "/include/params.h");
    generateEnvironmentHeader(deepNet, dirName + "/dnn/include/env.hpp");

    deepNet.fusePadding();  // probably already done, but make sure!
    addBranchesCells(deepNet);

    IniParser exportParams;

    if(!DeepNetExport::mExportParameters.empty())
        exportParams.load(DeepNetExport::mExportParameters);

    const bool wrapAroundBuffer = exportParams.getProperty(
        CPP_Config::OPTIMIZE_BUFFER_MEMORY,
        CPP_Config::OPTIMIZE_BUFFER_MEMORY_DEFAULT);

    const bool noBranchConcatOpt = exportParams.getProperty(
        CPP_Config::OPTIMIZE_NOBRANCH_CONCAT,
        CPP_Config::OPTIMIZE_NOBRANCH_CONCAT_DEFAULT);

    const bool includeInputInBuffer = exportParams.getProperty(
        CPP_Config::INCLUDE_INPUT_IN_BUFFER,
        CPP_Config::INCLUDE_INPUT_IN_BUFFER_DEFAULT);

    const int memoryAlignment = exportParams.getProperty(
        CPP_Config::MEMORY_ALIGNMENT,
        CPP_Config::MEMORY_ALIGNMENT_DEFAULT);

    MemoryManager memManager = generateMemory(deepNet, wrapAroundBuffer,
                    noBranchConcatOpt, includeInputInBuffer, memoryAlignment);

    DrawNet::drawGraph(deepNet, dirName + "/graph");

    memManager.optimize(exportParams.getProperty<MemoryManager::OptimizeStrategy>
        (CPP_Config::MEMORY_MANAGER_STRATEGY,
        CPP_Config::MEMORY_MANAGER_STRATEGY_DEFAULT));

    memManager.log(dirName + "/memory_mapping.log");

    DeepNetExport::generateCells(deepNet, dirName, "CPP");

    generateMemoryInfoHeader(deepNet, dirName + "/dnn/include/mem_info.hpp", 
                             memManager, memoryAlignment);
    generateNetworkPropagateFile(deepNet,
                                 dirName + "/src/NetworkPropagate.cpp");
    printStats(deepNet, memManager);
}

N2D2::MemoryManager N2D2::CPP_DeepNetExport::generateMemory(
    DeepNet& deepNet,
    bool wrapAroundBuffer,
    bool noBranchConcatOpt,
    bool includeInputInBuffer,
    int memoryAlignment)
{
    MemoryManager memManager;

    if (includeInputInBuffer) {
        // Create memory manager and allocate input channels
        const std::shared_ptr<StimuliProvider>& sp = deepNet.getStimuliProvider();
        const unsigned int nbChannelsAligned = (sp->getNbChannels() > 1)
            ? memoryAlignment * (unsigned int)std::ceil(sp->getNbChannels()
                                                    / (double)memoryAlignment) : 1;

        memManager.allocate(std::shared_ptr<Cell>(),
            nbChannelsAligned,
            deepNet.getChildCells("env"),
            nbChannelsAligned,
            sp->getSizeX(),
            sp->getSizeY());
        memManager.tick();
    }

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    std::map<std::shared_ptr<Cell>, MemoryManager::MemoryPlane> noBranchConcats;
    std::vector<std::shared_ptr<Cell> > excludedAllocableCells;

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end(); itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            if (cell->getType() == CPP_ConcatCell::Type
                && noBranchConcats.find(cell) != noBranchConcats.end())
            {
                continue;
            }

            std::vector<std::shared_ptr<Cell> > childs
                = deepNet.getChildCells(cell->getName());

            unsigned int size = (cell->getNbOutputs() > 1)
                ? memoryAlignment * (unsigned int)std::ceil(cell->getNbOutputs()
                                                / (double)memoryAlignment) : 1;
            unsigned int stride = size;
            unsigned int length = cell->getOutputsWidth();
            unsigned int count = cell->getOutputsHeight();

            bool isWrappable = true;
            std::vector<std::shared_ptr<Cell> > allocableCells;
            std::shared_ptr<Cell> concatCell;

            // Check if next layer is concatenation without branch
            // => if it is the case, allocate the concatenated memory directly
            // to avoid memory copies
            if (noBranchConcatOpt && childs.size() == 1
                && childs.back()->getType() == CPP_ConcatCell::Type)
            {
                bool noBranchConcat = true;

                std::vector<std::shared_ptr<Cell> > concatParents
                    = deepNet.getParentCells(childs.back()->getName());

                for (std::vector<std::shared_ptr<Cell> >::const_iterator
                    itCell = concatParents.begin(),
                    itCellEnd = concatParents.end();
                    itCell != itCellEnd; ++itCell)
                {
                    const std::vector<std::shared_ptr<Cell> > parentChilds
                        = deepNet.getChildCells((*itCell)->getName());

                    if (parentChilds.size() > 1)
                        noBranchConcat = false;

                    if (!((*itCell)->getType() == ConvCell::Type
                        || (*itCell)->getType() == PoolCell::Type
                        || (*itCell)->getType() == ElemWiseCell::Type
                        || (*itCell)->getType() == ScalingCell::Type))
                    {
                        isWrappable = false;
                    }
                }

                if (noBranchConcat) {
                    concatCell = childs.back();
                    allocableCells.swap(concatParents);

                    // In this case, the memory alignment is on the concatenated
                    // size (stride), not each cell size
                    size = cell->getNbOutputs();
                    stride = (concatCell->getNbOutputs() > 1)
                            ? memoryAlignment * (unsigned int)
                                std::ceil(concatCell->getNbOutputs()
                                / (double)memoryAlignment)
                            : 1;

                    assert(concatCell->getNbOutputs() > cell->getNbOutputs());
                    assert(concatCell->getOutputsWidth()
                        == cell->getOutputsWidth());
                    assert(concatCell->getOutputsHeight()
                        == cell->getOutputsHeight());
                }
            }
            else {
                isWrappable = (cell->getType() == ConvCell::Type
                    || cell->getType() == PoolCell::Type
                    || cell->getType() == ElemWiseCell::Type
                    || cell->getType() == ScalingCell::Type);
                allocableCells.push_back(cell);
            }

            const size_t fullSize = stride * length * count;

            // Check if wrap around buffer is possible for this cell
            // (re-using previous cell outputs memory for this cell outputs).
            // => only if this cell is the only child of its parent(s)
            size_t wrapAroundSize = 0;
            size_t wrapAroundExtra = 0;
            const MemoryManager::MemoryPlane* wrapAroundMemPlane = NULL;

            for (std::vector<std::shared_ptr<Cell> >::const_iterator
                itCell = allocableCells.begin(),
                itCellEnd = allocableCells.end();
                itCell != itCellEnd; ++itCell)
            {
                if (std::find(excludedAllocableCells.begin(),
                    excludedAllocableCells.end(), *itCell)
                        != excludedAllocableCells.end())
                {
                    continue;
                }

                // Select the best parent among all allocable cells for 
                // reallocation, which is the one with most memory (in order
                // to minimize the reallocation size).
                std::vector<std::shared_ptr<Cell> > parents
                    = deepNet.getParentCells((*itCell)->getName());

                for (std::vector<std::shared_ptr<Cell> >::const_iterator
                    itParent = parents.begin(), itParentEnd = parents.end();
                    itParent != itParentEnd; ++itParent)
                {
                    const std::vector<std::shared_ptr<Cell> > parentChilds
                        = deepNet.getChildCells(((*itParent))
                            ? (*itParent)->getName() : "env");

                    if (parentChilds.size() == 1) {
                        const std::map<std::shared_ptr<Cell>,
                            MemoryManager::MemoryPlane>::iterator itConcat
                                = noBranchConcats.find((*itParent));

                        // Reminder: there can be multiple allocable cells only
                        // for concatenation. In this case, we want all the
                        // allocable cells to be allocated on the same memory
                        // space with striding to avoid a concat operation.

                        // Nb planes may be 0 if the parent cell was not yet 
                        // processed.
                        // In this case, this allocable cell cannot be the
                        // current cell, and an other allocable cell will be
                        // allocated in this round. Therefore, for the next
                        // rounds, the memory of this allocable cell's parent
                        // cannot be used for wrapping as at least one other
                        // allocable cell was already allocated on a different
                        // memory space (using it would prevent concatenation
                        // with stride).
                        // TODO: depending on the processing order of the graph,
                        // this may lead to sub-optimal memory mapping!
                        if (itConcat == noBranchConcats.end()
                            && memManager.getNbPlanes((*itParent)) == 0)
                        {
                            excludedAllocableCells.push_back(*itCell);
                            continue;
                        }

                        const MemoryManager::MemoryPlane& memPlane
                            = (itConcat != noBranchConcats.end())
                                ? (*itConcat).second
                                : memManager.getPlanes((*itParent)).back();

                        if (isWrappable || !memManager.isWrapAround(
                                    memPlane.memSpace,
                                    memPlane.getFinalOffset()
                                        - memPlane.memSpace->offset,
                                    fullSize))
                        {
                            if (memPlane.getSize() > wrapAroundSize) {
                                wrapAroundSize = memPlane.getSize();
                                wrapAroundMemPlane = &memPlane;
                            }
                        }
                    }
                }
            }

            // Compute the extra memory needed for wrapping
            if (isWrappable && wrapAroundSize > 0) {
                for (std::vector<std::shared_ptr<Cell> >::const_iterator
                    itCell = allocableCells.begin(),
                    itCellEnd = allocableCells.end();
                    itCell != itCellEnd; ++itCell)
                {
                    // Compute the minimum number of lines that must be retained
                    // at the input before overwriting by the ouput.
                    std::size_t marginLines = 0;

                    if ((*itCell)->getType() == ConvCell::Type) {
                        const auto convCell
                            = std::dynamic_pointer_cast<ConvCell>((*itCell));
                        marginLines = convCell->getPaddingY()
                            / (double)convCell->getStrideY();
                    }
                    else if ((*itCell)->getType() == PoolCell::Type) {
                        const auto poolCell
                            = std::dynamic_pointer_cast<PoolCell>((*itCell));
                        marginLines = poolCell->getPaddingY()
                            / (double)poolCell->getStrideY();
                    }
                    // No margin necessary for ElemWiseCell
                    // No margin necessary for ScalingCell

                    // Take into account memory alignment of the input
                    const size_t nbChannels
                        = ((*itCell)->getType() == ElemWiseCell::Type)
                            // Specific case for ElemWise: each input has the
                            // same number of channels and we are overwriting 
                            // only one of them
                            ? (*itCell)->getNbOutputs()
                            : (*itCell)->getNbChannels();
                    const size_t inputSize = (nbChannels > 1)
                            ? memoryAlignment * (unsigned int)std::ceil(
                                nbChannels / (double)memoryAlignment) : 1;
                    const size_t inputFullSize = inputSize
                        * (*itCell)->getChannelsWidth()
                        * (*itCell)->getChannelsHeight();

                    // The extra space must accomodate for (marginLines + 1)
                    // ouput lines (+1 because the full input line has not 
                    // necessarily been read before the writing starts).
                    const size_t outputLineSize = stride * length;
                    const size_t wrapAroundCellExtra
                        = (std::max(inputFullSize, fullSize)
                            + outputLineSize * (marginLines + 1)) - fullSize;

                    if (wrapAroundCellExtra > wrapAroundExtra)
                        wrapAroundExtra = wrapAroundCellExtra;
                }
            }

            // Dependencies should be concat cell childs, not concat cell
            if (concatCell)
                childs = deepNet.getChildCells(concatCell->getName());

            std::map<std::shared_ptr<Cell>, MemoryManager::MemoryPlane>
                ::iterator itConcat = (concatCell)
                    ? noBranchConcats.find(concatCell)
                    : noBranchConcats.end();

            // MemoryPlane to (re)use
            const MemoryManager::MemoryPlane& memPlane
                = (itConcat != noBranchConcats.end())
                    ? (*itConcat).second :
                  (wrapAroundBuffer && wrapAroundSize > 0)
                    ? (*wrapAroundMemPlane) :
                       memManager.allocate(size, childs, stride, length, count);

            // Compute concatOffset
            unsigned int concatOffset = 0;

            if (concatCell) {
                std::vector<std::shared_ptr<Cell> > concatParents
                    = deepNet.getParentCells(concatCell->getName());

                for (std::vector<std::shared_ptr<Cell> >::const_iterator
                    itCell = concatParents.begin(),
                    itCellEnd = concatParents.end();
                    itCell != itCellEnd; ++itCell)
                {
                    if ((*itCell) == cell)
                        break;
                    else
                        concatOffset += (*itCell)->getNbOutputs();
                }
            }

            if (wrapAroundBuffer && wrapAroundSize > 0) {
                memManager.reallocate(memPlane,
                    cell, concatOffset,
                    size, true, wrapAroundExtra, childs,
                    stride, length, count);
            }
            else {
                memManager.reallocate(memPlane.memSpace,
                    cell, memPlane.offset + concatOffset,
                    size, false, 0, childs, stride, length, count);
            }

            if (concatCell && itConcat == noBranchConcats.end()) {
                std::tie(itConcat, std::ignore)
                    = noBranchConcats.emplace(concatCell, memPlane);
            }

            memManager.releaseDependencies(cell);
        }

        memManager.tick();
    }

    // Remove noBranchConcats cells
    for (std::map<std::shared_ptr<Cell>, MemoryManager::MemoryPlane>
        ::const_iterator itConcat = noBranchConcats.begin(),
        itConcatEnd = noBranchConcats.end();
        itConcat != itConcatEnd; ++itConcat)
    {
        deepNet.removeCell((*itConcat).first);
    }

    return memManager;
}

void N2D2::CPP_DeepNetExport::addBranchesCells(DeepNet& deepNet) {
    // Need a copy of layers as we will modify the deepNet during the iteration.
    const std::vector<std::vector<std::string>> layers = deepNet.getLayers();

    for(auto itLayer = layers.begin() + 1; itLayer != layers.end(); itLayer++) {
        for(auto itCell = itLayer->begin(); itCell != itLayer->end(); ++itCell) {
            std::shared_ptr<Cell> cell = deepNet.getCell(*itCell);
            if(!cell) {
                throw std::runtime_error("Invalid cell.");
            }

            auto parentsCells = cell->getParentsCells();
            if(parentsCells.size() > 1) {
                if(std::string(cell->getType()) != ElemWiseCell::Type) {
                    auto reg = Registrar<CPP_ConcatCell>::create(getCellModelType(*cell));
                    auto concatCell = reg(deepNet, 
                                          deepNet.generateNewCellName(cell->getName() + "_concat"), 
                                          cell->getNbChannels());

                    deepNet.addCellBefore(concatCell, cell);
                }
            }
        }
    }
}

void N2D2::CPP_DeepNetExport::generateParamsHeader(const std::string& fileName)
{
    // Export parameters
    std::ofstream paramsHeader(fileName.c_str());

    if (!paramsHeader.good())
        throw std::runtime_error("Could not create CPP header file: params.h");

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    paramsHeader << "// N2D2 auto-generated file.\n"
                    "// @ " << std::asctime(localNow)
                 << "\n"; // std::asctime() already appends end of line

    paramsHeader << "#ifndef N2D2_EXPORTC_PARAMS_H\n"
                    "#define N2D2_EXPORTC_PARAMS_H\n\n";

    // Constants
    paramsHeader << "#define NB_BITS " << (int)CellExport::mPrecision << "\n"
                 << "#define UNSIGNED_DATA " << DeepNetExport::mUnsignedData << "\n\n";

    paramsHeader << "#endif // N2D2_EXPORTC_PARAMS_H" << std::endl;
}

void N2D2::CPP_DeepNetExport::generateEnvironmentHeader(DeepNet& deepNet,
                                                      const std::string
                                                      & fileName)
{
    // Environment
    std::ofstream envHeader(fileName.c_str());

    if (!envHeader.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    envHeader << "// N2D2 auto-generated file.\n"
                 "// @ " << std::asctime(localNow)
              << "\n"; // std::asctime() already appends end of line

    envHeader << "#ifndef N2D2_EXPORTCPP_ENV_LAYER_H\n"
                 "#define N2D2_EXPORTCPP_ENV_LAYER_H\n\n";

    envHeader << "#include <stdint.h>\n\n";

    const std::shared_ptr<StimuliProvider> sp = deepNet.getStimuliProvider();

    // Constants
    envHeader << "#define ENV_SIZE_X " << sp->getSizeX() << "\n"
              << "#define ENV_SIZE_Y " << sp->getSizeY() << "\n"
              << "#define ENV_NB_OUTPUTS " << sp->getNbChannels() << "\n\n"
              << "#define ENV_DATA_UNSIGNED " << mEnvDataUnsigned << "\n\n"
              << "#define ENV_OUTPUTS_SIZE (ENV_NB_OUTPUTS*ENV_SIZE_X*ENV_SIZE_Y)\n\n";

    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();

    const unsigned int nbTarget = outputTargets.size();
    envHeader << "#define NETWORK_TARGETS " << nbTarget << "\n";
    envHeader << "//Output targets network dimension definition:\n";
    envHeader << "static unsigned int OUTPUTS_HEIGHT[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << cell->getOutputsHeight();
    }
    envHeader << "};\n";

    envHeader << "static unsigned int OUTPUTS_WIDTH[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << cell->getOutputsWidth();
    }
    envHeader << "};\n";

    envHeader << "static unsigned int NB_OUTPUTS[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        std::shared_ptr<Cell_Frame_Top> targetCellTop = std::dynamic_pointer_cast
            <Cell_Frame_Top>(cell);

        const BaseTensor& outputsShape = targetCellTop->getOutputs();

        envHeader << cell->getNbOutputs()*(outputsShape.dimB()/sp->getBatchSize());
    }
    envHeader << "};\n";

    envHeader << "static unsigned int NB_TARGET[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << ((cell->getNbOutputs() > 1) ? cell->getNbOutputs() : 2);
    }
    envHeader << "};\n";

    envHeader << "static unsigned int OUTPUTS_SIZE[NETWORK_TARGETS] = {";
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
        if(targetIdx > 0)
            envHeader << ",";

        envHeader << "(OUTPUTS_WIDTH[" << targetIdx << "]"
                  << "*OUTPUTS_HEIGHT["<< targetIdx << "]";

        if (outputTargets[targetIdx]->getParameter<bool>("DataAsTarget"))
            envHeader << "*NB_OUTPUTS["<< targetIdx << "]";

        envHeader << ")";
    }
    envHeader << "};\n";

    // Target type
    for(unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx)
    {
        const std::shared_ptr<Cell> cell = deepNet.getTargetCell(targetIdx);
    
        if (!outputTargets[targetIdx]->getParameter<bool>("DataAsTarget")) {
            envHeader << "typedef int32_t Target_" << targetIdx << "_T;\n";
        }
        else {
            std::string dataType = DeepNetExport::isCellOutputUnsigned(*cell)
                ? "UDATA_T" : "DATA_T";

            envHeader << "typedef " << dataType << " Target_"
                << targetIdx << "_T;\n";
        }
    }

    // Default target type
    if (nbTarget > 0)
        envHeader << "typedef Target_0_T Target_T;\n";
    else
        envHeader << "typedef int32_t Target_T;\n";

    envHeader << "#endif // N2D2_EXPORTCPP_ENV_LAYER_H" << std::endl;
}

void N2D2::CPP_DeepNetExport::generateHeaderBegin(DeepNet& /*deepNet*/,
                                                std::ofstream& header,
                                                const std::string& fileName)
{
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    header << "// N2D2 auto-generated file.\n"
              "// @ " << std::asctime(localNow)
           << "\n"; // std::asctime() already appends end of line

    const std::string guardName
        = Utils::upperCase(Utils::baseName(Utils::fileBaseName(fileName)));

    header << "#ifndef N2D2_EXPORTC_" << guardName << "_H\n"
                                                      "#define N2D2_EXPORTC_"
           << guardName << "_H\n\n";
}

void N2D2::CPP_DeepNetExport::generateHeaderIncludes(DeepNet& deepNet,
                                                     const std::string typeStr,
                                                     std::ofstream& header)
{
    header << "#include \"n2d2" + typeStr + ".hpp\"\n"
              "#include \"env.hpp\"\n"
              "#include \"../../include/Scaling.hpp\"\n";
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

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
            header << "#include \"" << Utils::CIdentifier(cell->getName())
                << ".hpp\"\n";
        }
    }
}


void N2D2::CPP_DeepNetExport::generateHeaderEnd(DeepNet& /*deepNet*/,
                                              std::ofstream& header)
{
    header << "\n"
              "#endif" << std::endl;
    header.close();
}

void N2D2::CPP_DeepNetExport::generateHeaderUtils(std::ofstream& header)
{

    header << "#include \"../../include/typedefs.h\"\n";
    header << "#include \"../../include/utils.h\"\n";
    header << "\n";
    header << "void setProfiling();\n";
    header << "void reportProfiling(unsigned int nbIter);\n";
    header << "unsigned int getOutputNbTargets();\n";
    header << "unsigned int getOutputTarget(unsigned int target);\n";
    header << "unsigned int getOutputDimZ(unsigned int target);\n";
    header << "unsigned int getOutputDimY(unsigned int target);\n";
    header << "unsigned int getOutputDimX(unsigned int target);\n";
    header << "unsigned int getInputDimZ();\n";
    header << "unsigned int getInputDimY();\n";
    header << "unsigned int getInputDimX();\n";
    header << "\n";

}

void N2D2::CPP_DeepNetExport::generateProgramUtils(std::ofstream& prog)
{
    prog << "void setProfiling() { set_profiling(); }\n";
    prog << "void reportProfiling(unsigned int nbIter) { report_per_layer_profiling(nbIter); }\n";
    prog << "unsigned int getOutputNbTargets(){ return NETWORK_TARGETS; }\n";
    prog << "unsigned int getOutputTarget(unsigned int target){ return NB_TARGET[target]; }\n";
    prog << "unsigned int getOutputDimZ(unsigned int target){ return NB_OUTPUTS[target]; }\n";
    prog << "unsigned int getOutputDimY(unsigned int target){ return OUTPUTS_HEIGHT[target]; }\n";
    prog << "unsigned int getOutputDimX(unsigned int target){ return OUTPUTS_WIDTH[target]; }\n";
    prog << "unsigned int getInputDimZ(){ return ENV_NB_OUTPUTS; }\n";
    prog << "unsigned int getInputDimY(){ return ENV_SIZE_Y; }\n";
    prog << "unsigned int getInputDimX(){ return ENV_SIZE_X; }\n";

}

void N2D2::CPP_DeepNetExport::generateMemoryInfoHeader(
    const DeepNet& deepNet,
    const std::string& fileName, 
    const MemoryManager& memManager,
    int memoryAlignment)
{
    std::ofstream memInfo(fileName.c_str());

    if (!memInfo.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    memInfo << "// N2D2 auto-generated file.\n"
                 "// @ " << std::asctime(localNow)
              << "\n"; // std::asctime() already appends end of line

    memInfo << "#ifndef N2D2_EXPORTCPP_MEM_INFO_H\n"
               "#define N2D2_EXPORTCPP_MEM_INFO_H\n\n";

    memInfo << "#define MEMORY_SIZE " << memManager.getPeakUsage() << "\n"
        "#define MEMORY_ALIGNMENT " << memoryAlignment << "\n";

    // env
    const std::vector<N2D2::MemoryManager::MemoryPlane>& envMemPlanes
        = memManager.getPlanes(std::shared_ptr<Cell>());

    if (!envMemPlanes.empty()) {
        assert(envMemPlanes.size() == 1);
        const N2D2::MemoryManager::MemoryPlane& memPlane = envMemPlanes.back();

        memInfo << "#define ENV_MEM_SIZE " << memPlane.size <<"\n";
        memInfo << "#define ENV_MEM_STRIDE " << memPlane.stride <<"\n";
        memInfo << "#define ENV_MEM_LENGTH " << memPlane.length <<"\n";
        memInfo << "#define ENV_MEM_COUNT " << memPlane.count <<"\n";

        memInfo << "#define ENV_MEM_CONT_OFFSET "
            << memPlane.getContiguousOffset() <<"\n";
        memInfo << "#define ENV_MEM_CONT_SIZE "
            << memPlane.getContiguousSize() <<"\n";
        memInfo << "#define ENV_MEM_WRAP_OFFSET "
            << memPlane.getWrappedOffset() <<"\n";
        memInfo << "#define ENV_MEM_WRAP_SIZE "
            << memPlane.getWrappedSize() <<"\n";
    }
    else {
        memInfo << "#define ENV_MEM_SIZE ENV_NB_OUTPUTS\n";
        memInfo << "#define ENV_MEM_STRIDE ENV_NB_OUTPUTS\n";
        memInfo << "#define ENV_MEM_LENGTH ENV_SIZE_X\n";
        memInfo << "#define ENV_MEM_COUNT ENV_SIZE_Y\n";

        memInfo << "#define ENV_MEM_CONT_OFFSET 0\n";
        memInfo << "#define ENV_MEM_CONT_SIZE ENV_MEM_SIZE\n";
        memInfo << "#define ENV_MEM_WRAP_OFFSET 0\n";
        memInfo << "#define ENV_MEM_WRAP_SIZE 0\n";
    }

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end(); itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);
            const std::vector<N2D2::MemoryManager::MemoryPlane>& memPlanes
                = memManager.getPlanes(cell);

            assert(memPlanes.size() == 1);
            const N2D2::MemoryManager::MemoryPlane& memPlane = memPlanes.back();

            const std::string identifier
                = N2D2::Utils::CIdentifier(cell->getName());
            const std::string prefix = Utils::upperCase(identifier);

            memInfo << "#define " << prefix << "_MEM_SIZE "
                << memPlane.size <<"\n";
            memInfo << "#define " << prefix << "_MEM_STRIDE "
                << memPlane.stride <<"\n";
            memInfo << "#define " << prefix << "_MEM_LENGTH "
                << memPlane.length <<"\n";
            memInfo << "#define " << prefix << "_MEM_COUNT "
                << memPlane.count <<"\n";

            memInfo << "#define " << prefix << "_MEM_CONT_OFFSET "
                << memPlane.getContiguousOffset() <<"\n";
            memInfo << "#define " << prefix << "_MEM_CONT_SIZE "
                << memPlane.getContiguousSize() <<"\n";
            memInfo << "#define " << prefix << "_MEM_WRAP_OFFSET "
                << memPlane.getWrappedOffset() <<"\n";
            memInfo << "#define " << prefix << "_MEM_WRAP_SIZE "
                << memPlane.getWrappedSize() <<"\n";
        }
    }

    memInfo << "\n"
              "#endif" << std::endl;
    memInfo.close();
}

void N2D2::CPP_DeepNetExport::generateNetworkPropagateFile(
    const DeepNet& deepNet, 
    const std::string& filePath) 
{
    std::stringstream includes;
    std::stringstream buffers;
    std::stringstream functionCalls;

    // Fill in includes, buffers and functionCalls for each layer
    /*
    buffers << "static DATA_T mem[MEMORY_SIZE]"
        " N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_MEMORY);\n";
    */

    //find the max type for mem (excluding the very last output, which can be int32)
    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();
    int maxActPrecision = 0;
    //the output of the last layer is in int32, do not take into account
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end() -1; itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(*cell);

            if (cellFrame.getActivation() && (cellFrame.getActivation()->getType() == "Rectifier" ||
                                              cellFrame.getActivation()->getType() == "Linear"))
            {
                const Activation& activation = *cellFrame.getActivation();
                int actPrecision = (int) activation.getQuantizedNbBits();
                maxActPrecision = std::max(actPrecision,maxActPrecision);
            }
        }
    }
    //this is a default case
    std::string dataType = "DATA_T";
    //change the type according to the max
    if(maxActPrecision > 0 && maxActPrecision <= 8){
            dataType = "int8_t";
    }
    else if(maxActPrecision > 8 && maxActPrecision <= 16){
            dataType = "int16_t";
    }
    else if(maxActPrecision > 16){
            dataType = "int32_t";
    }

    // Fill in includes, buffers and functionCalls for each layer
    buffers << "static "<< dataType << " mem[MEMORY_SIZE]"
        " N2D2_SECTION_ATTRIBUTE(N2D2_SECTION_NN_MEMORY);\n";

    functionCalls << "#ifdef SAVE_OUTPUTS\n"
                << "    FILE* env_stream = fopen(\"env_output.txt\", \"w\");\n"
                << "    saveOutputs("
                << "ENV_NB_OUTPUTS, "
                << "ENV_SIZE_Y, " 
                << "ENV_SIZE_X, "
                << "ENV_MEM_CONT_OFFSET, "
                << "ENV_MEM_CONT_SIZE, "
                << "ENV_MEM_WRAP_OFFSET, "
                << "ENV_MEM_WRAP_SIZE, "
                << "ENV_MEM_STRIDE, "
                << "inputs, "
                << "env_stream, "
                << "Network::Format::CHW"
                << ");\n"
                << "    fclose(env_stream);\n"
                << "#endif\n";

    //const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
        = layers.begin() + 1,
        itLayerEnd = layers.end(); itLayer != itLayerEnd; ++itLayer)
    {
        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
            itEnd = (*itLayer).end();
            it != itEnd; ++it)
        {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);
            const std::string identifier
                = N2D2::Utils::CIdentifier(cell->getName());
            const std::string prefix = Utils::upperCase(identifier);

            std::string dataType = DeepNetExport::isCellOutputUnsigned(*cell)
                ? "UDATA_T" : "DATA_T";

            if (cell->getParentsCells().size() > 1) {
                dataType = DeepNetExport::isCellInputsUnsigned(*cell)
                    ? "UDATA_T" : "DATA_T";
            }

            // functionCalls
            //this is replaced by CPP_CellExport::generateOutputType
            //called by each cell
            /*
            functionCalls << "    // " << cell->getName() << "\n";
            functionCalls << "    " << dataType << "* " << identifier
                << "_output = " << "(" << dataType << "*) mem + " 
                << prefix << "_MEM_CONT_OFFSET" <<";\n\n";
            */

            CPP_CellExport::getInstance(*cell)->generateCallCode(deepNet, *cell, 
                includes, buffers, functionCalls);

            functionCalls << "\n\n\n\n";
        }
    }

    // Handle network output in functionCalls
    const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet.getTargets();
    const unsigned int nbTarget = outputTargets.size();

    for (unsigned int targetIdx = 0; targetIdx < nbTarget; ++targetIdx) {
        const std::shared_ptr<Cell> targetCell = deepNet.getTargetCell(targetIdx);
        const std::string targetCellIdentifier = N2D2::Utils::CIdentifier(targetCell->getName());
        const std::string targetCellPrefix = N2D2::Utils::upperCase(targetCellIdentifier);

        if (!outputTargets[targetIdx]->getParameter<bool>("DataAsTarget")) {

            //if this is QAT, maxActPrecision >0
            //replace maxpropagate to take the output of the last layer into account
            //the output of the last FC is int32
            if(maxActPrecision > 0){
                functionCalls << "    maxPropagate<"
                            << targetCellPrefix << "_NB_OUTPUTS, "
                            << targetCellPrefix << "_OUTPUTS_HEIGHT, "
                            << targetCellPrefix << "_OUTPUTS_WIDTH, "
                            << "0, "
                            << targetCellPrefix << "_MEM_CONT_SIZE, "
                            << targetCellPrefix << "_MEM_WRAP_OFFSET, "
                            << targetCellPrefix << "_MEM_WRAP_SIZE, "
                            << targetCellPrefix << "_MEM_STRIDE"
                        << ">("
                            << targetCellIdentifier << "_output, "
                            << "outputs"
                        << ");\n\n";
            }
            else{
                functionCalls << "    maxPropagate<"
                            << targetCellPrefix << "_NB_OUTPUTS, "
                            << targetCellPrefix << "_OUTPUTS_HEIGHT, "
                            << targetCellPrefix << "_OUTPUTS_WIDTH, "
                            << targetCellPrefix << "_MEM_CONT_OFFSET, "
                            << targetCellPrefix << "_MEM_CONT_SIZE, "
                            << targetCellPrefix << "_MEM_WRAP_OFFSET, "
                            << targetCellPrefix << "_MEM_WRAP_SIZE, "
                            << targetCellPrefix << "_MEM_STRIDE"
                        << ">("
                            << targetCellIdentifier << "_output, "
                            << "outputs"
                        << ");\n\n";
            }

            functionCalls << "#ifdef SAVE_OUTPUTS\n"
                        << "    FILE* max_stream = fopen(\"max_output.txt\", \"w\");\n"
                        << "    saveOutputs("
                        << targetCellPrefix << "_NB_OUTPUTS, "
                        << targetCellPrefix << "_OUTPUTS_HEIGHT, " 
                        << targetCellPrefix << "_OUTPUTS_WIDTH, "
                        << targetCellPrefix << "_MEM_CONT_OFFSET, "
                        << targetCellPrefix << "_MEM_CONT_SIZE, "
                        << targetCellPrefix << "_MEM_WRAP_OFFSET, "
                        << targetCellPrefix << "_MEM_WRAP_SIZE, "
                        << targetCellPrefix << "_MEM_STRIDE, "
                        << "outputs, "
                        << "max_stream, "
                        << "Network::Format::CHW"
                        << ");\n"
                        << "    fclose(max_stream);\n"
                        << "#endif\n";
        }
        else {
            std::string dataType = DeepNetExport::isCellOutputUnsigned(*targetCell)
                ? "UDATA_T" : "DATA_T";

            functionCalls << "    memcpy(outputs, "
                        << targetCellIdentifier << "_output, "
                        << targetCellPrefix << "_NB_OUTPUTS * "
                        << targetCellPrefix << "_OUTPUTS_HEIGHT * " 
                        << targetCellPrefix << "_OUTPUTS_WIDTH * "
                        "sizeof(" << dataType << "));\n";
        }
    }

    // Write source file with includes, buffers and functionCalls
    std::ofstream networkPropagateFile(filePath);

    networkPropagateFile << "#include \"Network.hpp\"\n"
                         << "#include \"Scaling.hpp\"\n"
                         << "#include \"env.hpp\"\n"
                         << "#include \"mem_info.hpp\"\n"
                         << "\n"
                         << includes.str()
                         << "\n\n";




    networkPropagateFile << buffers.str()
                         << "\n\n";

    const std::string inputType = DeepNetExport::mEnvDataUnsigned?"UDATA_T":"DATA_T";

    networkPropagateFile << "namespace N2D2 {\n"
                            << "\n"
                            << "template<>\n"
                            << "void Network::propagate(const " << inputType << "* inputs, "
                                                    << "Target_T* outputs) const \n"
                            << "{\n"
                            << functionCalls.str()
                            << "\n"
                            << "}\n"
                            << "\n";


    networkPropagateFile << "/*template<>\n"
                         << "float Network::backpropagate(const DATA_T* input, const std::int32_t* labels){\n"
                         << "   const float loss = 0.0f;\n"
                         << "   return loss;\n "
                         << "}\n"
                         << "\n";

    networkPropagateFile << "int Network::gradientCheck(){\n"
                         << "   return(0);\n"
                         << "}*/\n"
                         << "\n"
                         << "}\n";

    networkPropagateFile.close();
    if(!networkPropagateFile) {
        throw std::runtime_error("Coudln't write file '" + filePath + "'.");
    }
}

void N2D2::CPP_DeepNetExport::printStats(const DeepNet& deepNet, 
                                         const MemoryManager& memManager) 
{
    Cell::Stats globalStats;

    const std::vector<std::vector<std::string>>& layers = deepNet.getLayers();
    for(std::size_t iLayer = 1; iLayer < layers.size(); iLayer++) {
        for(std::size_t iCell = 0; iCell < layers[iLayer].size(); iCell++) {
            const auto& cell = deepNet.getCell(layers[iLayer][iCell]);
            cell->getStats(globalStats);
        }
    }

    std::cout << "\nEstimated intermediate buffer usage: " << 
        memManager.getPeakUsage()*(std::abs(CellExport::mPrecision)/8)/1024.0 
        << " KiB." << std::endl;
    
    std::cout << "Estimated weights and constants usage: " << 
        globalStats.nbSynapses*(std::abs(CellExport::mPrecision)/8)/1024.0 
        << " KiB.\n" << std::endl;
}

std::string N2D2::CPP_DeepNetExport::getCellModelType(const Cell& cell) {
    const Cell_Frame_Top& cellFrameTop
        = dynamic_cast<const Cell_Frame_Top&>(cell);

    return (cellFrameTop.isCuda())
        ? Cell_Frame_Top::FRAME_CUDA_TYPE
        : Cell_Frame_Top::FRAME_TYPE;
}
