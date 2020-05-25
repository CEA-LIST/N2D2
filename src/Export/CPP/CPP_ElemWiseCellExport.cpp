
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

#include "Export/CPP/CPP_ElemWiseCellExport.hpp"

N2D2::Registrar<N2D2::ElemWiseCellExport>
N2D2::CPP_ElemWiseCellExport::mRegistrar(
    "CPP", N2D2::CPP_ElemWiseCellExport::generate);

void N2D2::CPP_ElemWiseCellExport::generate(ElemWiseCell& cell,
                                             const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create C header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_ElemWiseCellExport::generateHeaderConstants(ElemWiseCell& cell,
                                                            std::ofstream
                                                            & header)
{

    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                    cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs()
           << "\n"
              "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels()
           << "\n"
              "#define " << prefix << "_OUTPUTS_WIDTH "
           << cell.getOutputsWidth() << "\n"
              "#define " << prefix
           << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
                                                               "#define "
           << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth()
           << "\n"
              "#define " << prefix << "_CHANNELS_HEIGHT "
           << cell.getChannelsHeight() << "\n\n";

    CPP_CellExport::generateActivation(cell, header);
    CPP_CellExport::generateActivationScaling(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix
           << "_NB_OUTPUTS*" << prefix << "_OUTPUTS_WIDTH*" << prefix
           << "_OUTPUTS_HEIGHT)\n"
              "#define " << prefix << "_CHANNELS_SIZE (" << prefix
           << "_NB_CHANNELS*" << prefix << "_CHANNELS_WIDTH*" << prefix
           << "_CHANNELS_HEIGHT)\n"
              "#define " << prefix << "_BUFFER_SIZE (MAX(" << prefix
           << "_OUTPUTS_SIZE, " << prefix << "_CHANNELS_SIZE))\n\n";

    const std::vector<Float_T> weights = cell.getWeights();
    header << "static WDATA_T  " << prefix << "_WEIGHTS[";

    header << weights.size();

    header << "] = {";
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        //header << weights[i];
        CellExport::generateFreeParameter(weights[i], header);
        if(i < weights.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";


    const std::vector<Float_T> shifts = cell.getShifts();
    header << "static WDATA_T  " << prefix << "_SHIFTS[";

    header << shifts.size();

    header << "] = {";
    for(unsigned int i = 0; i < shifts.size(); ++i)
    {
        CellExport::generateFreeParameter(shifts[i], header);
        if(i < shifts.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";
    header <<  "static WDATA_T  " << prefix << "_POWER[";

    header << shifts.size();

    header << "] = {";
    for(unsigned int i = 0; i < shifts.size(); ++i)
    {
        header << "1";

        if(i < shifts.size() - 1)
            header << ", ";
    }

    header << "};"
            << "\n";

    const ElemWiseCell::Operation elemOp = cell.getOperation();

    header << "#define " << prefix << "_ELEM_OP " << elemOp << "\n"
           << std::endl;


}
