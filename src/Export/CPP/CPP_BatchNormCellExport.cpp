
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Export/CPP/CPP_BatchNormCellExport.hpp"

N2D2::Registrar<N2D2::BatchNormCellExport>
N2D2::CPP_BatchNormCellExport::mRegistrar(
    "CPP", N2D2::CPP_BatchNormCellExport::generate);

void N2D2::CPP_BatchNormCellExport::generate(const BatchNormCell& cell,
                                             const std::string& dirName)
{
    Utils::createDirectories(dirName + "/dnn");
    Utils::createDirectories(dirName + "/dnn/include");

    const std::string fileName = dirName + "/dnn/include/"
        + Utils::CIdentifier(cell.getName()) + ".hpp";

    std::ofstream header(fileName.c_str());

    if (!header.good())
        throw std::runtime_error("Could not create CPP header file: " + fileName);

    CPP_CellExport::generateHeaderBegin(cell, header);
    CPP_CellExport::generateHeaderIncludes(cell, header);
    generateHeaderConstants(cell, header);
    generateHeaderFreeParameters(cell, header);
    CPP_CellExport::generateHeaderEnd(cell, header);
}

void N2D2::CPP_BatchNormCellExport::generateHeaderConstants(const BatchNormCell& cell,
                                                            std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    header << "#define " << prefix << "_NB_OUTPUTS " << cell.getNbOutputs() << "\n"
           << "#define " << prefix << "_NB_CHANNELS " << cell.getNbChannels() << "\n"
           << "#define " << prefix << "_OUTPUTS_WIDTH " << cell.getOutputsWidth() << "\n"
           << "#define " << prefix << "_OUTPUTS_HEIGHT " << cell.getOutputsHeight() << "\n"
           << "#define " << prefix << "_CHANNELS_WIDTH " << cell.getChannelsWidth() << "\n"
           << "#define " << prefix << "_CHANNELS_HEIGHT " << cell.getChannelsHeight() << "\n\n";

    CPP_CellExport::generateActivation(cell, header);

    header << "#define " << prefix << "_OUTPUTS_SIZE (" << prefix << "_NB_OUTPUTS*" 
                                                        << prefix << "_OUTPUTS_WIDTH*" 
                                                        << prefix << "_OUTPUTS_HEIGHT)\n"
           << "#define " << prefix << "_CHANNELS_SIZE (" << prefix << "_NB_CHANNELS*" 
                                                         << prefix << "_CHANNELS_WIDTH*" 
                                                         << prefix << "_CHANNELS_HEIGHT)\n\n";
}

void N2D2::CPP_BatchNormCellExport::generateHeaderFreeParameters(const BatchNormCell& cell,
                                                                 std::ofstream& header)
{
    generateHeaderEpsilon(cell, header);
    generateHeaderBias(cell, header);
    generateHeaderVariance(cell, header);
    generateHeaderMean(cell, header);
    generateHeaderScale(cell, header);
}

void N2D2::CPP_BatchNormCellExport::generateHeaderEpsilon(const BatchNormCell& cell,
                                                          std::ofstream& header)
{
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                        cell.getName()));
    header << "static double " << prefix
           << "_EPSILON = " << cell.getParameter<double>("Epsilon") << ";\n";
}

void N2D2::CPP_BatchNormCellExport::generateHeaderBias(const BatchNormCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());

    header << "static WDATA_T " << identifier
        << "_biases[" << Utils::upperCase(identifier)
        << "_NB_OUTPUTS] = ";

    header << "{";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> bias;
        cell.getBias(output, bias);

        CellExport::generateFreeParameter(bias(0), header);
    }
    header << "};\n";
}

void N2D2::CPP_BatchNormCellExport::generateHeaderVariance(const BatchNormCell& cell,
                                                           std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier
        << "_variances[" << prefix << "_NB_OUTPUTS] = {\n";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> variance;
        cell.getVariance(output, variance);

        CellExport::generateFreeParameter(variance(0), header);
    }
    header << "};\n\n";
}

void N2D2::CPP_BatchNormCellExport::generateHeaderMean(const BatchNormCell& cell,
                                                       std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier
        << "_means[" << prefix << "_NB_OUTPUTS] = {\n";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> mean;
        cell.getMean(output, mean);

        CellExport::generateFreeParameter(mean(0), header);
    }
    header << "};\n\n";
}

void N2D2::CPP_BatchNormCellExport::generateHeaderScale(const BatchNormCell& cell,
                                                        std::ofstream& header)
{
    const std::string identifier = Utils::CIdentifier(cell.getName());
    const std::string prefix = Utils::upperCase(identifier);

    header << "static WDATA_T " << identifier
        << "_scales[" << prefix << "_NB_OUTPUTS] = {\n";

    for (std::size_t output = 0; output < cell.getNbOutputs(); ++output) {
        if (output > 0)
            header << ", ";

        Tensor<double> scale;
        cell.getScale(output, scale);

        CellExport::generateFreeParameter(scale(0), header);
    }

    header << "};\n\n";
}

