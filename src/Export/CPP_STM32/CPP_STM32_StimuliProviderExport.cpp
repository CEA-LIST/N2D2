/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <string>

#include "DeepNet.hpp"
#include "StimuliProvider.hpp"
#include "Database/Database.hpp"
#include "Export/CPP_STM32/CPP_STM32_StimuliProviderExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "utils/Registrar.hpp"

static const N2D2::Registrar<N2D2::StimuliProviderExport> registrar(
                    "CPP_STM32", N2D2::CPP_STM32_StimuliProviderExport::generate);

void N2D2::CPP_STM32_StimuliProviderExport::generate(const DeepNet& deepNet, StimuliProvider& sp,
                                                     const std::string& dirName,
                                                     Database::StimuliSet /*set*/,
                                                     bool unsignedData,
                                                     CellExport::Precision precision,
                                                     int nbStimuliMax)
{
    Utils::createDirectories(dirName);

    Database::StimuliSet setLearn = Database::StimuliSet::Learn;
    StimuliProviderExport::generate(deepNet, sp, dirName, setLearn, unsignedData, precision,
                                    nbStimuliMax, StimuliProviderExport::HWC);
                                    
    const std::size_t size = nbStimuliMax >= 0?std::min(sp.getDatabase().getNbStimuli(setLearn),
                                                        static_cast<unsigned int>(nbStimuliMax)):
                                               sp.getDatabase().getNbStimuli(setLearn);

    const std::vector<std::shared_ptr<Target>> outputTargets = deepNet.getTargets();
    const std::size_t netNbTarget = outputTargets.size();

    
    Utils::createDirectories(dirName + "/../include");

    const std::string stimulusFile = dirName + "/../include/test_stimulus.h";
    std::ofstream stimulusStream(stimulusFile);
    const std::string labelsFile = dirName + "/../include/test_labels.h";
    std::ofstream labelsStream(labelsFile);


    //stimulusStream << "static const " << (unsignedData?"UDATA_T":"DATA_T") 
    //               << " input[] __attribute__((section(\".nn_input\"))) = {\n";
    size_t sizeIm = sp.getSizeY()*sp.getSizeX()*sp.getNbChannels();
    size_t sizeLabel = 1;

    stimulusStream << "static const " << (unsignedData?"UDATA_T":"DATA_T") 
                   << " input[" << size << "][" << sizeIm << "]"
                   " __attribute__((section(\".nn_input\"))) = {";
    labelsStream << "static const int32_t "  
                   << " label[" << size << "][" << sizeLabel << "]"
                   " __attribute__((section(\".nn_input\"))) = { ";
    srand (0);
       
    for (std::size_t stimuliIdx = 0; stimuliIdx < size; ++stimuliIdx) {
        const int randStimuliIdx = rand() % sp.getDatabase().getNbStimuli(setLearn);
        sp.readStimulus(setLearn, randStimuliIdx);
        const Tensor<int> label = sp.getLabelsDataChannel(0);


        stimulusStream << "\n {"; 
        labelsStream << " {"; 
        std::size_t i = 0;
        for (std::size_t y = 0; y < sp.getSizeY(); y++) {
            for (std::size_t x = 0; x < sp.getSizeX(); x++) {
                for (std::size_t ch = 0; ch < sp.getNbChannels(); ch++) {
                    writeStimulusValue(sp.getData()(x, y, ch, 0), unsignedData, 
                                    precision, stimulusStream, false);
                    
                    i++;
                    if(i % 30 == 0)
                        stimulusStream << "\n";
                
                }
            }
        }
        for(std::size_t t = 0; t < netNbTarget; ++t) {
            for (std::size_t y = 0; y < label.dimY(); ++y) {
                for (std::size_t x = 0; x < label.dimX(); ++x) {
                    const uint32_t outputTarget = deepNet.getTarget(t)->getLabelTarget(label(x, y));
                    labelsStream << outputTarget;
                    i++;
                }
            }
        }
        
        stimulusStream << "},"; 
        labelsStream << "},\n"; 
    }
     
    stimulusStream << "};\n";
    stimulusStream.close();

    if(!stimulusStream) {
        throw std::runtime_error("Couldn't create '" + stimulusFile + "'.");
    }
    labelsStream << "};\n";
    labelsStream.close();
    if(!labelsStream) {
        throw std::runtime_error("Couldn't create '" + labelsFile + "'.");
    }

}
