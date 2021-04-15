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

#ifndef N2D2_N2D2_H
#define N2D2_N2D2_H

#include <string>

namespace N2D2 {
extern const char* const RCS_Product;
extern const char* const RCS_Company;
extern const char* const RCS_BuildTime;

/**
 * Return the absolute path where the input stimuli for the simulations are
 *stored, using the N2D2_DATA environment
 * variable. The function provides a convenient way to eliminate absolute paths
 *in your code.
 *
 * This function appends the path contained in the N2D2_DATA environment
 *variable to the @p path parameter. If N2D2_DATA is not
 * set, the default location is used: /local/$USER/n2d2_data
 *
 * @param path Relative path of a stimuli.
 *
 * @note The N2D2_DATA variable does not need to have a trailing slash "/".
*/
const char* N2D2_DATA(const std::string& path);

const char* N2D2_PATH(const std::string& path);
}

#endif // N2D2_N2D2_H
