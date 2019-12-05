/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
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

#include "Database/N_MNIST_Database.hpp"
#include <bitset>

N2D2::N_MNIST_Database::N_MNIST_Database(double validation)
    : AER_Database(), mValidation(validation)

{
    // ctor
}


/// This loads the database and partitions it into learning and testing samples
void N2D2::N_MNIST_Database::load(const std::string& dataPath,
                                    const std::string& /*labelPath*/,
                                    bool /*extractROIs*/)
{

    unsigned int nbClasses = 10;
    unsigned int nbTrainImages = 60000;

    for (unsigned int cls = 0; cls < nbClasses; ++cls) {
        for (unsigned int i = 1; i <= nbTrainImages; ++i) {

            std::ostringstream nameStr;
            nameStr << dataPath << "/Train/" << cls << "/";
            nameStr << std::setfill('0') << std::setw(5) << i << ".bin";

            std::ifstream fileExists(nameStr.str());
            if (fileExists) {

                mStimuli.push_back(Stimulus(nameStr.str(), cls));
                mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
            }
        }
    }

    // Assign loaded stimuli to learning and validation set
    partitionStimuli(1.0 - mValidation, mValidation, 0.0);

    unsigned int nbTestImages = 10000;

    for (unsigned int cls = 0; cls < nbClasses; ++cls) {
        std::stringstream clsStr;
        clsStr << cls;
        mLabelsName.push_back(clsStr.str());

        for (unsigned int i = 1; i <= nbTestImages; ++i) {

            std::ostringstream nameStr;
            nameStr << dataPath << "/Test/" << cls << "/";
            nameStr << std::setfill('0') << std::setw(5) << i << ".bin";


            std::ifstream fileExists(nameStr.str());
            if (fileExists) {

                mStimuli.push_back(Stimulus(nameStr.str(), cls));
                mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
            }
        }
    }

    // Assign loaded stimuli to test set
    partitionStimuli(0.0, 0.0, 1.0);

}



void N2D2::N_MNIST_Database::loadAerStimulusData(
                                            std::vector<AerReadEvent>& aerData,
                                            StimuliSet set,
                                            StimulusID id,
                                            unsigned int batch)
{


    std::string filename = mStimuli[mStimuliSets(set)[id]].name;
    std::cout << "ID: " << id << std::endl;

    std::ifstream data(filename, std::ios::in|std::ios::binary|std::ios::ate);

    if (data.good()) {

            std::streampos size;
            char * memblock;
            size = data.tellg(); // Get pointer to end of file
            memblock = new char [size]; // Allocate memory for data
            data.seekg(0, std::ios::beg); // Set pointer to beginning
            data.read(memblock, size);
            data.close();

            unsigned int nbEvents = (unsigned int)((int)size/5);

            for (unsigned int ev = 0; ev < nbEvents; ++ev) {

                unsigned int offset = 5*ev;

                unsigned int xCoor =
                            (unsigned int)((unsigned char)memblock[offset]);
                unsigned int yCoor =
                            (unsigned int)((unsigned char)memblock[offset+1]);

                 // Use unsigned int because only 24 bit
                unsigned int bitstring = ((unsigned char)memblock[offset+2]);
                bitstring =
                    (bitstring << 8) | (unsigned char)memblock[offset+3];
                bitstring =
                    (bitstring << 8) | (unsigned char)memblock[offset+4];
                bitstring = bitstring << 8;

                // Bit 24 represents sign
                unsigned int sign = static_cast<unsigned int>(bitstring >> 31);
                // Bits 23-0 represent time
                unsigned int time = static_cast<unsigned int>((bitstring << 1)
                                                                >> (32-23));
                
                // We use here the polarity/sign for the channel, and not for
                // the event value
                aerData.push_back(AerReadEvent(xCoor, yCoor, sign, batch, 1, time));

            }

            delete[] memblock;
    }
    else {
        throw std::runtime_error("N_MNIST_Database::loadAerStimulusData: "
                                    "Could not open AER file: " + filename);
    }
}


/// This is only used in clock-simulation
void N2D2::N_MNIST_Database::loadAerStimulusData(
                                                std::vector<AerReadEvent>& aerData,
                                                StimuliSet set,
                                                StimulusID id,
                                                unsigned int batch,
                                                Time_T start,
                                                Time_T stop,
                                                unsigned int repetitions,
                                                unsigned int partialStimulus)
{


  

    std::vector<AerReadEvent> stimu;
    loadAerStimulusData(stimu, set, id, batch);

    Time_T intervalSize = (stop-start)/repetitions;
    if ((stop-start)%repetitions != 0) {
        std::cout << "start: " << start << std::endl;
        std::cout << "stop: " << stop << std::endl;
        std::cout << "repetitions: " << repetitions << std::endl;
        throw std::runtime_error("N_MNIST_Database::loadAerStimulusData: "
                                  " repetitions not multiple of stop-start");
    }

    unsigned int startCounter = 0;
    unsigned int stopCounter = 0;
    Time_T lastTime = stimu[stimu.size()-1].time;
    Time_T startTime = 0;

    if (partialStimulus >= 1 && partialStimulus <= 3) {
        if (partialStimulus == 1){
            lastTime = std::floor(0.33*stimu[stimu.size()-1].time);
            startTime = 0;
        }
        else if (partialStimulus == 2){
            lastTime = std::floor(0.66*stimu[stimu.size()-1].time);
            startTime = std::floor(0.33*stimu[stimu.size()-1].time);
        }
        else if (partialStimulus == 3){
            lastTime = std::floor(stimu[stimu.size()-1].time);
            startTime = std::floor(0.66*stimu[stimu.size()-1].time);
        }
        else {
            std::cout << partialStimulus << std::endl;
            throw std::runtime_error("N_MNIST_Database::loadAerStimulusData: "
                                  " partialStimulus value invalid");
        }
        for(std::vector<AerReadEvent>::iterator it=stimu.begin();
        it!=stimu.end(); ++it) {
            if ((*it).time <= lastTime) {
                stopCounter++;
                if ((*it).time <= startTime) {
                    startCounter++;
                }
            }
            else {
                break;
            }
        }
    }


    double scalingFactor = ((double)intervalSize)/(lastTime-startTime);

    for (unsigned int i = 0; i < repetitions; ++i) {
        for(std::vector<AerReadEvent>::iterator it=stimu.begin()+startCounter;
        it<=stimu.begin()+stopCounter; ++it) {
            Time_T scaledTime = std::floor((((*it).time-startTime)
                                            *scalingFactor) + i*intervalSize);
            aerData.push_back(AerReadEvent((*it).x, (*it).y,
                                    (*it).channel, (*it).batch, 1, scaledTime));
        }
    }
}



