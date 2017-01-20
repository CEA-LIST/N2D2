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

#include "Database/Caltech256_DIR_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

const char* labelNames[]
    = {"001.ak47",                "002.american-flag",
       "003.backpack",            "004.baseball-bat",
       "005.baseball-glove",      "006.basketball-hoop",
       "007.bat",                 "008.bathtub",
       "009.bear",                "010.beer-mug",
       "011.billiards",           "012.binoculars",
       "013.birdbath",            "014.blimp",
       "015.bonsai-101",          "016.boom-box",
       "017.bowling-ball",        "018.bowling-pin",
       "019.boxing-glove",        "020.brain-101",
       "021.breadmaker",          "022.buddha-101",
       "023.bulldozer",           "024.butterfly",
       "025.cactus",              "026.cake",
       "027.calculator",          "028.camel",
       "029.cannon",              "030.canoe",
       "031.car-tire",            "032.cartman",
       "033.cd",                  "034.centipede",
       "035.cereal-box",          "036.chandelier-101",
       "037.chess-board",         "038.chimp",
       "039.chopsticks",          "040.cockroach",
       "041.coffee-mug",          "042.coffin",
       "043.coin",                "044.comet",
       "045.computer-keyboard",   "046.computer-monitor",
       "047.computer-mouse",      "048.conch",
       "049.cormorant",           "050.covered-wagon",
       "051.cowboy-hat",          "052.crab-101",
       "053.desk-globe",          "054.diamond-ring",
       "055.dice",                "056.dog",
       "057.dolphin-101",         "058.doorknob",
       "059.drinking-straw",      "060.duck",
       "061.dumb-bell",           "062.eiffel-tower",
       "063.electric-guitar-101", "064.elephant-101",
       "065.elk",                 "066.ewer-101",
       "067.eyeglasses",          "068.fern",
       "069.fighter-jet",         "070.fire-extinguisher",
       "071.fire-hydrant",        "072.fire-truck",
       "073.fireworks",           "074.flashlight",
       "075.floppy-disk",         "076.football-helmet",
       "077.french-horn",         "078.fried-egg",
       "079.frisbee",             "080.frog",
       "081.frying-pan",          "082.galaxy",
       "083.gas-pump",            "084.giraffe",
       "085.goat",                "086.golden-gate-bridge",
       "087.goldfish",            "088.golf-ball",
       "089.goose",               "090.gorilla",
       "091.grand-piano-101",     "092.grapes",
       "093.grasshopper",         "094.guitar-pick",
       "095.hamburger",           "096.hammock",
       "097.harmonica",           "098.harp",
       "099.harpsichord",         "100.hawksbill-101",
       "101.head-phones",         "102.helicopter-101",
       "103.hibiscus",            "104.homer-simpson",
       "105.horse",               "106.horseshoe-crab",
       "107.hot-air-balloon",     "108.hot-dog",
       "109.hot-tub",             "110.hourglass",
       "111.house-fly",           "112.human-skeleton",
       "113.hummingbird",         "114.ibis-101",
       "115.ice-cream-cone",      "116.iguana",
       "117.ipod",                "118.iris",
       "119.jesus-christ",        "120.joy-stick",
       "121.kangaroo-101",        "122.kayak",
       "123.ketch-101",           "124.killer-whale",
       "125.knife",               "126.ladder",
       "127.laptop-101",          "128.lathe",
       "129.leopards-101",        "130.license-plate",
       "131.lightbulb",           "132.light-house",
       "133.lightning",           "134.llama-101",
       "135.mailbox",             "136.mandolin",
       "137.mars",                "138.mattress",
       "139.megaphone",           "140.menorah-101",
       "141.microscope",          "142.microwave",
       "143.minaret",             "144.minotaur",
       "145.motorbikes-101",      "146.mountain-bike",
       "147.mushroom",            "148.mussels",
       "149.necktie",             "150.octopus",
       "151.ostrich",             "152.owl",
       "153.palm-pilot",          "154.palm-tree",
       "155.paperclip",           "156.paper-shredder",
       "157.pci-card",            "158.penguin",
       "159.people",              "160.pez-dispenser",
       "161.photocopier",         "162.picnic-table",
       "163.playing-card",        "164.porcupine",
       "165.pram",                "166.praying-mantis",
       "167.pyramid",             "168.raccoon",
       "169.radio-telescope",     "170.rainbow",
       "171.refrigerator",        "172.revolver-101",
       "173.rifle",               "174.rotary-phone",
       "175.roulette-wheel",      "176.saddle",
       "177.saturn",              "178.school-bus",
       "179.scorpion-101",        "180.screwdriver",
       "181.segway",              "182.self-propelled-lawn-mower",
       "183.sextant",             "184.sheet-music",
       "185.skateboard",          "186.skunk",
       "187.skyscraper",          "188.smokestack",
       "189.snail",               "190.snake",
       "191.sneaker",             "192.snowmobile",
       "193.soccer-ball",         "194.socks",
       "195.soda-can",            "196.spaghetti",
       "197.speed-boat",          "198.spider",
       "199.spoon",               "200.stained-glass",
       "201.starfish-101",        "202.steering-wheel",
       "203.stirrups",            "204.sunflower-101",
       "205.superman",            "206.sushi",
       "207.swan",                "208.swiss-army-knife",
       "209.sword",               "210.syringe",
       "211.tambourine",          "212.teapot",
       "213.teddy-bear",          "214.teepee",
       "215.telephone-box",       "216.tennis-ball",
       "217.tennis-court",        "218.tennis-racket",
       "219.theodolite",          "220.toaster",
       "221.tomato",              "222.tombstone",
       "223.top-hat",             "224.touring-bike",
       "225.tower-pisa",          "226.traffic-light",
       "227.treadmill",           "228.triceratops",
       "229.tricycle",            "230.trilobite-101",
       "231.tripod",              "232.t-shirt",
       "233.tuning-fork",         "234.tweezer",
       "235.umbrella-101",        "236.unicorn",
       "237.vcr",                 "238.video-projector",
       "239.washing-machine",     "240.watch-101",
       "241.waterfall",           "242.watermelon",
       "243.welding-mask",        "244.wheelbarrow",
       "245.windmill",            "246.wine-bottle",
       "247.xylophone",           "248.yarmulke",
       "249.yo-yo",               "250.zebra",
       "251.airplanes-101",       "252.car-side-101",
       "253.faces-easy-101",      "254.greyhound",
       "255.tennis-shoes",        "256.toad",
       "257.clutter"};

TEST_DATASET(Caltech256_DIR_Database,
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
    REQUIRED(UnitTest::DirExists(N2D2_DATA("256_ObjectCategories")));

    Caltech256_DIR_Database db(learn, validation, incClutter);
    db.load(N2D2_DATA("256_ObjectCategories"));

    unsigned int nbStimuli = 30607; // Exclude "RENAME2" file in 198.spider and
    // "greg" and "greg/vision309" subdirs in
    // 056.dog
    unsigned int nbLabels = 257;

    if (!incClutter) {
        nbStimuli -= 827;
        --nbLabels;
    }

    ASSERT_EQUALS(db.getNbStimuliWithLabel(0), 98U);

    for (unsigned int label = 0; label < nbLabels; ++label) {
        ASSERT_EQUALS(db.getLabelName(label),
                      std::string("/") + labelNames[label]);
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
