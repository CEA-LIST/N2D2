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

#include "N2D2.hpp"

#include "Database/Caltech101_DIR_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

const char* labelNames[] = {
    "BACKGROUND_Google", "Faces",          "Leopards",      "Motorbikes",
    "accordion",         "airplanes",      "anchor",        "ant",
    "barrel",            "bass",           "beaver",        "binocular",
    "bonsai",            "brain",          "brontosaurus",  "buddha",
    "butterfly",         "camera",         "cannon",        "car_side",
    "ceiling_fan",       "cellphone",      "chair",         "chandelier",
    "cougar_body",       "cougar_face",    "crab",          "crayfish",
    "crocodile",         "crocodile_head", "cup",           "dalmatian",
    "dollar_bill",       "dolphin",        "dragonfly",     "electric_guitar",
    "elephant",          "emu",            "euphonium",     "ewer",
    "ferry",             "flamingo",       "flamingo_head", "garfield",
    "gerenuk",           "gramophone",     "grand_piano",   "hawksbill",
    "headphone",         "hedgehog",       "helicopter",    "ibis",
    "inline_skate",      "joshua_tree",    "kangaroo",      "ketch",
    "lamp",              "laptop",         "llama",         "lobster",
    "lotus",             "mandolin",       "mayfly",        "menorah",
    "metronome",         "minaret",        "nautilus",      "octopus",
    "okapi",             "pagoda",         "panda",         "pigeon",
    "pizza",             "platypus",       "pyramid",       "revolver",
    "rhino",             "rooster",        "saxophone",     "schooner",
    "scissors",          "scorpion",       "sea_horse",     "snoopy",
    "soccer_ball",       "stapler",        "starfish",      "stegosaurus",
    "stop_sign",         "strawberry",     "sunflower",     "tick",
    "trilobite",         "umbrella",       "watch",         "water_lilly",
    "wheelchair",        "wild_cat",       "windsor_chair", "wrench",
    "yin_yang"};

TEST_DATASET(Caltech101_DIR_Database,
             load,
             (double learn, double validation, bool incClutter),
             std::make_tuple(0.0, 0.0, true),
             std::make_tuple(1.0, 0.0, true),
             std::make_tuple(0.0, 1.0, true),
             std::make_tuple(0.6, 0.1, true),
             std::make_tuple(0.1, 0.4, true),
             std::make_tuple(0.0, 0.0, false),
             std::make_tuple(1.0, 0.0, false),
             std::make_tuple(0.0, 1.0, false),
             std::make_tuple(0.6, 0.1, false),
             std::make_tuple(0.1, 0.4, false))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("101_ObjectCategories")));

    Caltech101_DIR_Database db(learn, validation, incClutter);
    db.load(N2D2_DATA("101_ObjectCategories"));

    unsigned int nbStimuli
        = 8709; // Exclude Faces_easy and "tmp" file in BACKGROUND_Google
    unsigned int nbLabels
        = 101; // Exclude Faces_easy and "tmp" file in BACKGROUND_Google

    if (!incClutter) {
        nbStimuli -= 467;
        --nbLabels;
    }

    if (incClutter) {
        ASSERT_EQUALS(db.getNbStimuliWithLabel(0), 467U);
    } else {
        ASSERT_EQUALS(db.getNbStimuliWithLabel(0), 435U);
    }

    for (unsigned int label = 0; label < nbLabels; ++label) {
        ASSERT_EQUALS(db.getLabelName(label),
                      std::string("/") + labelNames[label + (!incClutter)]);
    }

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);

    if (learn == 0.0 && validation == 0.0) {
        ASSERT_EQUALS(db.getNbStimuli(Database::Test), nbStimuli);
    }

    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn)
                  + db.getNbStimuli(Database::Validation)
                  + db.getNbStimuli(Database::Test),
                  nbStimuli);
    ASSERT_EQUALS(db.getNbLabels(), nbLabels);
}

RUN_TESTS()
