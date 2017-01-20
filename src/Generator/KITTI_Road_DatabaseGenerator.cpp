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

#include "Generator/KITTI_Road_DatabaseGenerator.hpp"

N2D2::Registrar<N2D2::DatabaseGenerator>
N2D2::KITTI_Road_DatabaseGenerator::mRegistrar(
    "KITTI_Road_Database", N2D2::KITTI_Road_DatabaseGenerator::generate);

std::shared_ptr<N2D2::KITTI_Road_Database>
N2D2::KITTI_Road_DatabaseGenerator::generate(IniParser& iniConfig,
                                             const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const double learn = iniConfig.getProperty<double>("Learn", 0.8);

    const std::string dataPath = iniConfig.getProperty<std::string>(
        "DataPath", N2D2_DATA("KITTI/data_road/training/image_2"));
    const std::string labelPath = iniConfig.getProperty<std::string>(
        "LabelPath", N2D2_DATA("KITTI/data_road/training/gt_image_2"));

    std::shared_ptr<KITTI_Road_Database> database = std::make_shared
        <KITTI_Road_Database>(learn);
    database->setParameters(iniConfig.getSection(section, true));
    database->load(dataPath, labelPath);
    return database;
}
