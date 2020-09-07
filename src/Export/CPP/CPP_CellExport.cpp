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

#include "Cell/Cell_Frame_Top.hpp"
#include "Export/C/C_CellExport.hpp"
#include "Export/CPP/CPP_CellExport.hpp"
#include "Export/DeepNetExport.hpp"
#include "utils/Utils.hpp"

void N2D2::CPP_CellExport::generateHeaderBegin(const Cell& cell, std::ofstream& header) {
    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    header << "// N2D2 auto-generated file.\n"
              "// @ " << std::asctime(localNow)
           << "\n"; // std::asctime() already appends end of line

    const std::string prefix = Utils::upperCase(Utils::CIdentifier(
                                                            cell.getName()));

    header << "#ifndef N2D2_EXPORTC_" << prefix << "_LAYER_H\n"
                                                   "#define N2D2_EXPORTC_"
           << prefix << "_LAYER_H\n\n";
}

void N2D2::CPP_CellExport::generateHeaderIncludes(const Cell& /*cell*/, std::ofstream& header) {
    header << "#include \"../../include/typedefs.h\"\n";
    header << "#include \"../../include/utils.h\"\n";
}

void N2D2::CPP_CellExport::generateHeaderEnd(const Cell& cell, std::ofstream& header) {
    header << "#endif " 
           << "// N2D2_EXPORTC_" << Utils::upperCase(Utils::CIdentifier(cell.getName())) << "_LAYER_H\n";
}

void N2D2::CPP_CellExport::generateActivation(const Cell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    header << "#define " << prefix << "_ACTIVATION "  
                         << (cellFrame.getActivation()?cellFrame.getActivation()->getType():
                                                       "Linear") 
                         << "\n";
}

void N2D2::CPP_CellExport::generateActivationScaling(const Cell& cell, std::ofstream& header) {
    // TODO: needed for legacy code in CPP_Cuda and CPP_OpenCL
    // To be removed in the future --->
    C_CellExport::generateActivationScaling(cell, header);
    // <---

    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    if (cellFrame.getActivation() == nullptr) {
        header << "static const N2D2::NoScaling " << prefix << "_SCALING;\n";
        return;
    }
    
    const Activation& activation = *cellFrame.getActivation();
    const Scaling& activationScaling = activation.getActivationScaling();

    generateScaling(prefix, activationScaling,
        DeepNetExport::isCellOutputUnsigned(cell), header);
}

void N2D2::CPP_CellExport::generateScaling(
    const std::string& prefix,
    const Scaling& scaling,
    bool outputUnsigned,
    std::ofstream& header)
{
    if(scaling.getMode() == ScalingMode::NONE) {
        header << "static const N2D2::NoScaling " << prefix << "_SCALING;\n";
    }
    else if(scaling.getMode() == ScalingMode::FLOAT_MULT) {
        const auto& scalingPerOutput = scaling.getFloatingPointScaling().getScalingPerOutput();
        if(Utils::all_same(scalingPerOutput.begin(), scalingPerOutput.end())) {
            header << "static const N2D2::FloatingPointScaling " << prefix << "_SCALING = {" 
                                                                    << scalingPerOutput.front() << "};\n";
        }
        else {
            header << "static const N2D2::FloatingPointScalingPerChannel<" << scalingPerOutput.size() << "> " 
                                                                           << prefix << "_SCALING = {";
            header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
            header << "};\n";
        }
    }
    else if(scaling.getMode() == ScalingMode::FIXED_MULT) {
        const auto& fpScaling = scaling.getFixedPointScaling();
        const auto& scalingPerOutput = fpScaling.getScalingPerOutput();

        if(Utils::all_same(scalingPerOutput.begin(), scalingPerOutput.end())) {
            header << "static const N2D2::FixedPointScaling<" << scalingPerOutput.front() << ", "
                                                              << fpScaling.getFractionalBits() 
                                                        << "> " << prefix << "_SCALING;\n";
        }
        else {
            header << "static const N2D2::FixedPointScalingScalingPerChannel<" << scalingPerOutput.size() << ", " 
                                                                               << fpScaling.getFractionalBits() 
                                                                        << "> " << prefix << "_SCALING = {";
            header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
            header << "};\n";
        }
    }
    else if(scaling.getMode() == ScalingMode::SINGLE_SHIFT) {
        const auto& scalingPerOutput = scaling.getSingleShiftScaling().getScalingPerOutput();
        if(Utils::all_same(scalingPerOutput.begin(), scalingPerOutput.end())) {
            header << "static const N2D2::SingleShiftScaling<" << +scalingPerOutput.front() << "> " 
                                                                   << prefix << "_SCALING;\n";
        }
        else {
            header << "static const N2D2::SingleShiftScalingPerChannel<" << scalingPerOutput.size() << "> " 
                                                                             << prefix << "_SCALING = {";
            for(const auto& sc: scalingPerOutput) {
                header << +sc << ", ";
            }
            header << "};\n";
        }
    }
    else if(scaling.getMode() == ScalingMode::DOUBLE_SHIFT) {
        const auto& scalingPerOutput = scaling.getDoubleShiftScaling().getScalingPerOutput();
        if(Utils::all_same(scalingPerOutput.begin(), scalingPerOutput.end())) {
            header << "static const N2D2::DoubleShiftScaling<" << +scalingPerOutput.front().first << ", "
                                                               << +scalingPerOutput.front().second << "> " 
                                                                   << prefix << "_SCALING;\n";
        }
        else {
            header << "static const N2D2::DoubleShiftScalingPerChannel" 
                                                << "<" 
                                                    << scalingPerOutput.size() << ", " 
                                                    << outputUnsigned
                                                << "> " << prefix << "_SCALING = {";
            for(const auto& sc: scalingPerOutput) {
                header << "std::array<SUM_T, 3>{" << +sc.first << ", " << +sc.second << ", " 
                                                  << +(1ul << (sc.second - 1)) << "}, ";
            }
            header << "};\n";
        }
    }
    else {
        throw std::runtime_error("Unsupported rescaling mode.");
    }

    header << "\n";
}

void N2D2::CPP_CellExport::generateBenchmarkStart(const DeepNet& /*deepNet*/,
                                                  const Cell& cell, 
                                                  std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());

    // functionCalls: start benchmark
    functionCalls << "#ifdef BENCHMARK\n"
        "    const Tick_T start_" << identifier << " = tick();\n"
        "#endif\n\n";
}

void N2D2::CPP_CellExport::generateBenchmarkEnd(const DeepNet& /*deepNet*/,
                                                const Cell& cell, 
                                                std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());

    // functionCalls: stop benchmark
    functionCalls << "#ifdef BENCHMARK\n"
        "    const Tick_T end_" << identifier << " = tick();\n"
        "    static RunningMean_T " << identifier << "_timing = {0.0, 0};\n"
        "    benchmark(\"" << identifier << "\", start_" << identifier
        << ", end_" << identifier << ", " << identifier << "_timing);\n"
        "#endif\n\n";
}

void N2D2::CPP_CellExport::generateSaveOutputs(const DeepNet& /*deepNet*/,
                                               const Cell& cell, 
                                               std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    const std::string outputBuffer = Utils::CIdentifier(cell.getName() + "_output");

    // functionCalls: save outputs
    functionCalls << "#ifdef SAVE_OUTPUTS\n";
    functionCalls << "    std::ofstream " << identifier << "_stream(\"" 
                                              << identifier << "_output.txt\");\n";
    functionCalls << "    saveOutputs("
                << prefix << "_NB_OUTPUTS, "
                << prefix << "_OUTPUTS_HEIGHT, " 
                << prefix << "_OUTPUTS_WIDTH, "
                << prefix << "_MEM_CONT_OFFSET, "
                << prefix << "_MEM_CONT_SIZE, "
                << prefix << "_MEM_WRAP_OFFSET, "
                << prefix << "_MEM_WRAP_SIZE, "
                << prefix << "_MEM_STRIDE, "
                << outputBuffer << " , "
                << identifier << "_stream, "
                << "Network::Format::CHW"
            << ");\n";
    functionCalls << "    " << identifier << "_stream.close();\n";
    functionCalls << "#endif\n";
}
