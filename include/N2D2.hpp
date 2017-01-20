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

/** @mainpage N2D2 Index Page
 *
 * @section intro_sec Introduction
 *
 * This is a simple yet customizable event-driven simulator for spiking neural
 *network. @n
 * To get started and build your network, go to N2D2::Network.
 *
 * @section prereq_sec Dependences
 *
 * gnuplot (for raster plot generation) @n
 * OpenCV 2.x (for image and video handling and display) @n
 * Doxygen [optional] (for documentation generation) @n
 *
 * @section intall_basic Installation basic steps
 *
 * @subsection step1 Step 1: Get a local copy of the Git repository with 'git
 *clone'.
 *
 * @subsection step2 Step 2: Compile the project with one of the following
 *commands:
 * @verbatim make @endverbatim
 * @verbatim make debug @endverbatim
 *
 * Please report any error/warning during the compilation with your version of
 *GCC.
 *
 * @subsection step3 Step 3: Generate the documentation with 'make doc'.
 *
 * Optional step to generate this documentation.
 *
 * @section install_detailed Detailed installation guide
 *
 * \verbinclude README
*/

#ifndef N2D2_N2D2_H
#define N2D2_N2D2_H

#include "Aer.hpp"
#include "Cochlea.hpp"
#include "Layer.hpp"
#include "Monitor.hpp"
#include "Network.hpp"
#include "Sound.hpp"
#include "Xcell.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/ProgramOptions.hpp"

#include <cstdlib>
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
