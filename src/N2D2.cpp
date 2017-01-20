/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

const char* const N2D2::RCS_Product = "$Product: N2D2 $";
const char* const N2D2::RCS_Company = "$Company: CEA LIST $";
const char* const N2D2::RCS_BuildTime = "$BuildTime: " __DATE__ ", " __TIME__
                                        " $";

const char* N2D2::N2D2_DATA(const std::string& path)
{
    const char* base = std::getenv("N2D2_DATA");
    static std::string fullPath; // Must be static, otherwise the returned char*
    // pointer is not valid after the return!

    if (base == NULL) {
#if defined(WIN32)
        fullPath = std::string("C:\\n2d2_data\\") + path;
#else
        const char* user = std::getenv("USER");

        if (user != NULL)
            fullPath = std::string("/local/") + std::getenv("USER")
                       + "/n2d2_data/" + path;
        else
            fullPath = std::string("/local/n2d2_data/") + path;
#endif
    } else
        fullPath = std::string(base) + "/" + path;

    return fullPath.c_str();
}

const char* N2D2::N2D2_PATH(const std::string& path)
{
    const char* base = std::getenv("N2D2_PATH");
    static std::string fullPath; // Must be static, otherwise the returned char*
    // pointer is not valid after the return!

    if (base == NULL) {
#if defined(N2D2_COMPILE_PATH)
        fullPath = std::string(N2D2_COMPILE_PATH) + "/" + path;
#else
        fullPath = std::string("/local/n2d2/") + path;
#endif
    } else
        fullPath = std::string(base) + "/" + path;

    return fullPath.c_str();
}
