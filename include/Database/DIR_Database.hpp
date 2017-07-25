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

#ifndef N2D2_DIR_DATABASE_H
#define N2D2_DIR_DATABASE_H

#include "Database/Database.hpp"

namespace N2D2 {
class DIR_Database : public Database {
public:
    DIR_Database(bool loadDataInMemory = false);
    void setValidExtensions(const std::vector<std::string>& validExtensions);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool extractROIs = false);

    /**
     * Example:
     * @param depth
     *      depth = 0: load stimuli only from the current directory (dirPath)
     *      depth = 1: load stimuli from dirPath and stimuli contained in the
     * sub-directories of dirPath
     *      depth < 0: load stimuli recursively from dirPath and all its
     * sub-directories
     * @param labelDepth
     *      labelDepth = -1: no label for all stimuli (label ID = -1)
     *      labelDepth = 0: uses @p labelName string for all stimuli
     *      labelDepth = 1: uses @p labelName string for stimuli in the current
     * directory (dirPath) and @p labelName
     *       + sub-directory name for stimuli in the sub-directories
    */
    virtual void loadDir(const std::string& dirPath,
                         int depth = 0,
                         const std::string& labelName = "",
                         int labelDepth = 0);
    virtual void loadFile(const std::string& fileName);
    virtual void loadFile(const std::string& fileName,
                          const std::string& labelName);
    virtual ~DIR_Database() {};

protected:
    virtual void loadFile(const std::string& fileName, int label);

    std::vector<std::string> mValidExtensions;
};
}

#endif // N2D2_DIR_DATABASE_H
