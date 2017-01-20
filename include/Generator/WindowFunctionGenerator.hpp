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

#ifndef N2D2_WINDOWFUNCTIONGENERATOR_H
#define N2D2_WINDOWFUNCTIONGENERATOR_H

#include <memory>

#include "utils/IniParser.hpp"
#include "utils/WindowFunction.hpp"

namespace N2D2 {
class WindowFunctionGenerator {
public:
    static std::string defaultWindow;
    static double defaultSigma;
    static double defaultAlpha;
    static double defaultBeta;
    static void setDefault(IniParser& iniConfig,
                           const std::string& section,
                           const std::string& name);

    template <class T>
    static std::shared_ptr<WindowFunction<T> >
    generate(IniParser& iniConfig,
             const std::string& section,
             const std::string& name);
};
}

template <class T>
std::shared_ptr<N2D2::WindowFunction<T> >
N2D2::WindowFunctionGenerator::generate(IniParser& iniConfig,
                                        const std::string& section,
                                        const std::string& name)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string windowName = iniConfig.getProperty
                                   <std::string>(name, defaultWindow);

    if (windowName == "Rectangular")
        return std::make_shared<Rectangular<T> >();
    else if (windowName == "Hann")
        return std::make_shared<Hann<T> >();
    else if (windowName == "Hamming")
        return std::make_shared<Hamming<T> >();
    else if (windowName == "Cosine")
        return std::make_shared<Cosine<T> >();
    else if (windowName == "Gaussian") {
        const double sigma = iniConfig.getProperty
                             <double>(name + ".Sigma", defaultSigma);
        return std::make_shared<Gaussian<T> >(sigma);
    } else if (windowName == "Blackman") {
        const double alpha = iniConfig.getProperty
                             <double>(name + ".Alpha", defaultAlpha);
        return std::make_shared<Blackman<T> >(alpha);
    } else if (windowName == "Kaiser") {
        const double beta = iniConfig.getProperty
                            <double>(name + ".Beta", defaultBeta);
        return std::make_shared<Kaiser<T> >(beta);
    } else
        throw std::runtime_error("Unknown window function \"" + windowName
                                 + "\" in section [" + section + "]");
}

#endif // N2D2_WINDOWFUNCTIONGENERATOR_H
