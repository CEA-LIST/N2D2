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
    const std::string type = (cellFrame.getActivation())
        ? cellFrame.getActivation()->getType() : "Linear";

    header << "#define " << prefix << "_ACTIVATION " << type << "\n";

    if (cellFrame.getActivation()
        && (type == "Rectifier" || type == "Linear"))
    {
        const double clipping = cellFrame.getActivation()
            ->getParameter<double>("Clipping");

        if (clipping > 0) {
            std::cout << Utils::cwarning << "Clipping (" << clipping << ") for "
                << type << " in cell " << cell.getName() << " is not supported!"
                << Utils::cdef << std::endl;
        }
        const Activation& activation = *cellFrame.getActivation();
        if(activation.getQuantizedNbBits() > 0) {
            std::cout << Utils::cwarning << "Mixed-precision from QAT have been detected"
            <<" in cell " << cell.getName() << ": An additional clipping value per channel "
            << " is required"
            << Utils::cdef << std::endl;
            header << "#define " << prefix 
                    << "_NB_BITS_ACT " << (int) activation.getQuantizedNbBits() 
                    << "\n";
        }
    }
}

void N2D2::CPP_CellExport::generateWeightPrecision(const Cell& cell, std::ofstream& header) {
    const std::string prefix = Utils::upperCase(Utils::CIdentifier(cell.getName()));
    if(cell.getQuantizedNbBits() > 0) {
        header << "#define " << prefix
                << "_NB_BITS_W " << (int) cell.getQuantizedNbBits()
                << "\n";
    }
}

void N2D2::CPP_CellExport::generateActivationScaling(const Cell& cell, std::ofstream& header) {
    // TODO: needed for legacy code in CPP_Cuda and CPP_OpenCL
    // To be removed in the future --->
    try {
        C_CellExport::generateActivationScaling(cell, header);
    }
    catch (const std::exception& e) {
        // Pass for scaling modes not supported by the C export
    }
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
            if(!scaling.getFloatingPointScaling().getIsClipped()) {
            header << "static const N2D2::FloatingPointScalingPerChannel<" << scalingPerOutput.size() << "> " 
                                                                           << prefix << "_SCALING = {";
            header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
            header << "};\n";
            }
            else {
                const std::vector<Float_T>& clippingPerOutput 
                        = scaling.getFloatingPointScaling().getClippingPerOutput(); 
                //Implicit cast from float to int32_t... To improve
                std::vector<int32_t> clippingPerOutput_INT32(  clippingPerOutput.begin(), 
                                                                clippingPerOutput.end());
                assert(clippingPerOutput_INT32.size() == scalingPerOutput.size());
                header << "static const N2D2::FloatingPointClippingAndScalingPerChannel<" 
                                                                            << clippingPerOutput_INT32.size() << "> " 
                                                                            << prefix << "_CLIPPED_SCALING = {{";
                header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
                header << "}, {";
                header << Utils::join(clippingPerOutput_INT32.begin(), clippingPerOutput_INT32.end(), ',');
                header << "}};\n";
            }
        }
    }
    else if(scaling.getMode() == ScalingMode::FIXED_MULT16
        || scaling.getMode() == ScalingMode::FIXED_MULT32)
    {
        const auto& fpScaling = scaling.getFixedPointScaling();
        const auto& scalingPerOutput = fpScaling.getScalingPerOutput();

        if(Utils::all_same(scalingPerOutput.begin(), scalingPerOutput.end())) {
            header << "static const N2D2::FixedPointScaling<" << scalingPerOutput.front() << ", "
                                                              << fpScaling.getFractionalBits() 
                                                        << "> " << prefix << "_SCALING;\n";
        }
        else {
            if(!scaling.getFixedPointScaling().getIsClipped()) {
                header << "static const N2D2::FixedPointScalingPerChannel<" << scalingPerOutput.size() << ", " 
                                                                                << fpScaling.getFractionalBits() 
                                                                            << "> " << prefix << "_SCALING = {";
                header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
                header << "};\n";
            }            
            else {
                const std::vector<Float_T>& clippingPerOutput 
                        = scaling.getFixedPointScaling().getClippingPerOutput(); 
                //Implicit cast from float to int32_t... To improve
                std::vector<int32_t> clippingPerOutput_INT32(  clippingPerOutput.begin(), 
                                                                clippingPerOutput.end());
                assert(clippingPerOutput_INT32.size() == scalingPerOutput.size());
                header << "static const N2D2::FixedPointClippingAndScalingPerChannel<" 
                                                                            << clippingPerOutput_INT32.size()  << ", " 
                                                                            << fpScaling.getFractionalBits() 
                                                                            << "> " 
                                                                            << prefix << "_CLIPPED_SCALING = {{";
                header << Utils::join(scalingPerOutput.begin(), scalingPerOutput.end(), ',');
                header << "}, {";
                header << Utils::join(clippingPerOutput_INT32.begin(), clippingPerOutput_INT32.end(), ',');
                header << "}};\n";
            }

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

void N2D2::CPP_CellExport::generateOutputType(const DeepNet& /*deepNet*/,
                                                const Cell& cell,
                                                std::stringstream& functionCalls)
{
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);

    std::string dataType = DeepNetExport::isCellOutputUnsigned(cell)
                ? "UDATA_T" : "DATA_T";

    const std::string prefix = Utils::upperCase(identifier);

    //adapt cell output type
    if (cellFrame.getActivation()
        && (cellFrame.getActivation()->getType() == "Rectifier" || cellFrame.getActivation()->getType() == "Linear"))
    {
        const Activation& activation = *cellFrame.getActivation();
        int actPrecision = (int) activation.getQuantizedNbBits();

        if(actPrecision > 0 && actPrecision <= 8){
            dataType = DeepNetExport::isCellOutputUnsigned(cell) ? "uint8_t" : "int8_t";
        }
        else if(actPrecision > 8 && actPrecision <= 16){
            dataType = DeepNetExport::isCellOutputUnsigned(cell) ? "uint16_t" : "int16_t";
        }
        else if(actPrecision > 16){
            dataType = DeepNetExport::isCellOutputUnsigned(cell) ? "uint32_t" : "int32_t";
        }

        //functionCalls : cell output type
        //for the last FC
        if(actPrecision>8){
            functionCalls << "    // " << cell.getName() << "\n";
            functionCalls << "    " << dataType << " " << identifier
                            << "_output["  << prefix << "_MEM_CONT_SIZE" << "] = " << "{0};\n\n";
        }
        //for all other preceding layers
        else{
            functionCalls << "    // " << cell.getName() << "\n";
            functionCalls << "    " << dataType << "* " << identifier
                << "_output = " << "(" << dataType << "*) mem + "
                << prefix << "_MEM_CONT_OFFSET" <<";\n\n";
        }

    }
    else{
        //functionalCalls : cell output type by default
        functionCalls << "    // " << cell.getName() << "\n";
        functionCalls << "    " << dataType << "* " << identifier
                << "_output = " << "(" << dataType << "*) mem + "
                << prefix << "_MEM_CONT_OFFSET" <<";\n\n";
    }

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
    functionCalls << "    FILE* " << identifier << "_stream = fopen(\"" 
                                << identifier << "_output.txt\", \"w\");\n";
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
    functionCalls << "    fclose(" << identifier << "_stream);\n";
    functionCalls << "#endif\n";
}

std::string N2D2::CPP_CellExport::getLabelActivationRange(const Cell& cell) const {
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);
    const Activation& activation = *cellFrame.getActivation();
    /* If activation have been quantized through QAT method
       the dynamic can be layer specific, use the specific flag then.
       Else use the global DNN dynamic NBBITS
    */
    if(activation.getQuantizedNbBits() > 0) {
        std::string labelName = prefix + "_NB_BITS_ACT";
        return labelName;
    } 
    else {
        std::string labelName = "NB_BITS";
        return labelName;
    }
}

std::string N2D2::CPP_CellExport::getLabelScaling(const Cell& cell) const {
    const std::string identifier = N2D2::Utils::CIdentifier(cell.getName());
    const std::string prefix = N2D2::Utils::upperCase(identifier);
    std::string label = prefix + "_SCALING" ;

    const Cell_Frame_Top& cellFrame = dynamic_cast<const Cell_Frame_Top&>(cell);


    if (cellFrame.getActivation() == nullptr) {
        return label;
    }

    const Activation& activation = *cellFrame.getActivation();
    const Scaling& activationScaling = activation.getActivationScaling();

    /* 
        Check if the scaling is clipped then adapt the label
    */
    if(activationScaling.getMode() == ScalingMode::FLOAT_MULT) {
        if(activationScaling.getFloatingPointScaling().getIsClipped()) {
            label = prefix + "_CLIPPED_SCALING";
        }
    } 
    else if(activationScaling.getMode() == ScalingMode::FIXED_MULT16 ||
            activationScaling.getMode() == ScalingMode::FIXED_MULT32){
        if(activationScaling.getFixedPointScaling().getIsClipped()) {
            label = prefix + "_CLIPPED_SCALING";
        }
    }

    return label;
}