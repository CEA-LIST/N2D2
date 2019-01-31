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

#ifdef JSONCPP

#include "Database/Cityscapes_Database.hpp"
#include "ROI/PolygonalROI.hpp"

N2D2::Cityscapes_Database::Cityscapes_Database(bool incTrainExtra,
                                               bool useCoarse,
                                               bool singleInstanceLabels)
    : DIR_Database(),
      mIncTrainExtra(incTrainExtra),
      mUseCoarse(useCoarse),
      mSingleInstanceLabels(singleInstanceLabels)
{
    // ctor
    mLabels.push_back(Label(  "unlabeled"            ,  0 ,
        "void"            , 0       , false , true  , Color(  0,  0,  0) ));
    mLabels.push_back(Label(  "ego vehicle"          ,  1 ,
        "void"            , 0       , false , true  , Color(  0,  0,  0) ));
    mLabels.push_back(Label(  "rectification border" ,  2 ,
        "void"            , 0       , false , true  , Color(  0,  0,  0) ));
    mLabels.push_back(Label(  "out of roi"           ,  3 ,
        "void"            , 0       , false , true  , Color(  0,  0,  0) ));
    mLabels.push_back(Label(  "static"               ,  4 ,
        "void"            , 0       , false , true  , Color(  0,  0,  0) ));
    mLabels.push_back(Label(  "dynamic"              ,  5 ,
        "void"            , 0       , false , true  , Color(111, 74,  0) ));
    mLabels.push_back(Label(  "ground"               ,  6 ,
        "void"            , 0       , false , true  , Color( 81,  0, 81) ));
    mLabels.push_back(Label(  "road"                 ,  7 ,
        "flat"            , 1       , false , false , Color(128, 64,128) ));
    mLabels.push_back(Label(  "sidewalk"             ,  8 ,
        "flat"            , 1       , false , false , Color(244, 35,232) ));
    mLabels.push_back(Label(  "parking"              ,  9 ,
        "flat"            , 1       , false , true  , Color(250,170,160) ));
    mLabels.push_back(Label(  "rail track"           , 10 ,
        "flat"            , 1       , false , true  , Color(230,150,140) ));
    mLabels.push_back(Label(  "building"             , 11 ,
        "construction"    , 2       , false , false , Color( 70, 70, 70) ));
    mLabels.push_back(Label(  "wall"                 , 12 ,
        "construction"    , 2       , false , false , Color(102,102,156) ));
    mLabels.push_back(Label(  "fence"                , 13 ,
        "construction"    , 2       , false , false , Color(190,153,153) ));
    mLabels.push_back(Label(  "guard rail"           , 14 ,
        "construction"    , 2       , false , true  , Color(180,165,180) ));
    mLabels.push_back(Label(  "bridge"               , 15 ,
        "construction"    , 2       , false , true  , Color(150,100,100) ));
    mLabels.push_back(Label(  "tunnel"               , 16 ,
        "construction"    , 2       , false , true  , Color(150,120, 90) ));
    mLabels.push_back(Label(  "pole"                 , 17 ,
        "object"          , 3       , false , false , Color(153,153,153) ));
    mLabels.push_back(Label(  "polegroup"            , 18 ,
        "object"          , 3       , false , true  , Color(153,153,153) ));
    mLabels.push_back(Label(  "traffic light"        , 19 ,
        "object"          , 3       , false , false , Color(250,170, 30) ));
    mLabels.push_back(Label(  "traffic sign"         , 20 ,
        "object"          , 3       , false , false , Color(220,220,  0) ));
    mLabels.push_back(Label(  "vegetation"           , 21 ,
        "nature"          , 4       , false , false , Color(107,142, 35) ));
    mLabels.push_back(Label(  "terrain"              , 22 ,
        "nature"          , 4       , false , false , Color(152,251,152) ));
    mLabels.push_back(Label(  "sky"                  , 23 ,
        "sky"             , 5       , false , false , Color( 70,130,180) ));
    mLabels.push_back(Label(  "person"               , 24 ,
        "human"           , 6       , true  , false , Color(220, 20, 60) ));
    mLabels.push_back(Label(  "rider"                , 25 ,
        "human"           , 6       , true  , false , Color(255,  0,  0) ));
    mLabels.push_back(Label(  "car"                  , 26 ,
        "vehicle"         , 7       , true  , false , Color(  0,  0,142) ));
    mLabels.push_back(Label(  "truck"                , 27 ,
        "vehicle"         , 7       , true  , false , Color(  0,  0, 70) ));
    mLabels.push_back(Label(  "bus"                  , 28 ,
        "vehicle"         , 7       , true  , false , Color(  0, 60,100) ));
    mLabels.push_back(Label(  "caravan"              , 29 ,
        "vehicle"         , 7       , true  , true  , Color(  0,  0, 90) ));
    mLabels.push_back(Label(  "trailer"              , 30 ,
        "vehicle"         , 7       , true  , true  , Color(  0,  0,110) ));
    mLabels.push_back(Label(  "train"                , 31 ,
        "vehicle"         , 7       , true  , false , Color(  0, 80,100) ));
    mLabels.push_back(Label(  "motorcycle"           , 32 ,
        "vehicle"         , 7       , true  , false , Color(  0,  0,230) ));
    mLabels.push_back(Label(  "bicycle"              , 33 ,
        "vehicle"         , 7       , true  , false , Color(119, 11, 32) ));
    mLabels.push_back(Label(  "license plate"        , -1 ,
        "vehicle"         , 7       , false , true  , Color(  0,  0,142) ));

    for (std::vector<Label>::const_iterator it = mLabels.begin(),
        itEnd = mLabels.end(); it != itEnd; ++it)
    {
        labelID((*it).name);
    }
}

void N2D2::Cityscapes_Database::load(const std::string& dataPath,
                                     const std::string& labelPath,
                                     bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty())
        ? Utils::dirName(dataPath) + ((mUseCoarse) ? "gtCoarse" : "gtFine")
        : labelPath;

    // Train
    loadDir(dataPath + "/train", 1, "", -1);
    loadLabels(labelPathDef + "/train");

    if (mIncTrainExtra) {
        const std::string labelPathExtraDef = (labelPath.empty())
            ? Utils::dirName(dataPath) + "/gtCoarse"
            : labelPath;

        loadDir(dataPath + "/train_extra", 1, "", -1);
        loadLabels(labelPathExtraDef + "/train");
    }

    partitionStimuli(1.0, 0.0, 0.0);

    // Val
    loadDir(dataPath + "/val", 1, "", -1);
    loadLabels(labelPathDef + "/val");
    partitionStimuli(0.0, 1.0, 0.0);

    // Test
    loadDir(dataPath + "/test", 1, "", -1);
    loadLabels(labelPathDef + "/test");
    partitionStimuli(0.0, 0.0, 1.0);
}

void N2D2::Cityscapes_Database::loadLabels(const std::string& labelPath) {
    for (std::vector<StimulusID>::const_iterator it
        = mStimuliSets(Unpartitioned).begin(),
         itEnd = mStimuliSets(Unpartitioned).end(); it != itEnd; ++it)
    {
        const std::string cityName
            = Utils::baseName(Utils::dirName(mStimuli[(*it)].name, true));
        const std::string typeName
            = Utils::baseName(Utils::dirName(labelPath, true));
        const std::string labelName = labelPath + "/" + cityName + "/"
            + Utils::baseName(Utils::fileBaseName(mStimuli[(*it)].name, "_"))
            + "_" + typeName + "_polygons.json";

        std::ifstream jsonData(labelName.c_str());

        if (!jsonData.good()) {
            throw std::runtime_error("Could not open JSON label file "
                                     "(missing?): " + labelName);
        }

        Json::Reader reader;
        Json::Value labels;

        if (!reader.parse(jsonData, labels)) {
            std::cerr << "Error parsing JSON file " << labelName << " at line "
                << reader.getFormattedErrorMessages() << std::endl;

            throw std::runtime_error("JSON file parsing failed");
        }

        const Json::Value& objects = labels["objects"];

        for (unsigned int i = 0; i < objects.size(); ++i) {
            const Json::Value& object = objects[i];
            const Json::Value& polygon = object["polygon"];
            std::vector<cv::Point> pts;

            for (unsigned int pt = 0; pt < polygon.size(); ++pt) {
                pts.push_back(cv::Point(polygon[pt][0].asInt(),
                                        polygon[pt][1].asInt()));
            }

            std::string label = object["label"].asString();
            const std::string group = "group";

            if (mSingleInstanceLabels
                && label.size() > group.size()
                && std::equal(group.rbegin(), group.rend(), label.rbegin()))
            {
                label = label.substr(0, label.size() - group.size());
            }

            mStimuli[(*it)].ROIs.push_back(new PolygonalROI<int>(
                labelID(label), pts));
        }
    }
}

#endif
