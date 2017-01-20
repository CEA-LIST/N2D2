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

#ifndef N2D2_KERNELGENERATOR_H
#define N2D2_KERNELGENERATOR_H

#include "utils/IniParser.hpp"
#include "utils/Kernel.hpp"

namespace N2D2 {
class KernelGenerator {
public:
    static std::string defaultKernel;
    static double defaultSigma;
    static double defaultSigma1;
    static double defaultSigma2;
    static double defaultLambda;
    static double defaultPsi;
    static double defaultGamma;
    static void setDefault(IniParser& iniConfig,
                           const std::string& section,
                           const std::string& name);

    template <class T>
    static Kernel<T> generate(IniParser& iniConfig,
                              const std::string& section,
                              const std::string& name);
};
}

template <class T>
N2D2::Kernel<T> N2D2::KernelGenerator::generate(IniParser& iniConfig,
                                                const std::string& section,
                                                const std::string& name)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string kernelName = iniConfig.getProperty
                                   <std::string>(name, defaultKernel);

    if (kernelName.empty())
        return Kernel<T>("");

    if (kernelName == "*") {
        return Kernel
            <T>(iniConfig.getProperty<std::string>(name + ".Mat"),
                iniConfig.getProperty<unsigned int>(name + ".SizeX", 0U),
                iniConfig.getProperty<unsigned int>(name + ".SizeY", 0U));
    } else {
        const unsigned int sizeX = iniConfig.getProperty
                                   <unsigned int>(name + ".SizeX");
        const unsigned int sizeY = iniConfig.getProperty
                                   <unsigned int>(name + ".SizeY");

        if (kernelName == "Gaussian") {
            const bool positive = iniConfig.getProperty
                                  <bool>(name + ".Positive", true);
            const double sigma = iniConfig.getProperty
                                 <double>(name + ".Sigma", defaultSigma);

            const Kernel<T> kernel = GaussianKernel<T>(sizeX, sizeY, sigma);
            return (positive) ? kernel : -kernel;
        } else if (kernelName == "LoG" || kernelName == "LaplacianOfGaussian") {
            const bool positive = iniConfig.getProperty
                                  <bool>(name + ".Positive", true);
            const double sigma = iniConfig.getProperty
                                 <double>(name + ".Sigma", defaultSigma);

            const Kernel<T> kernel = LaplacianOfGaussianKernel
                <T>(sizeX, sizeY, sigma);
            return (positive) ? kernel : -kernel;
        } else if (kernelName == "DoG" || kernelName
                                          == "DifferenceOfGaussian") {
            const bool positive = iniConfig.getProperty
                                  <bool>(name + ".Positive", true);
            const double sigma1 = iniConfig.getProperty
                                  <double>(name + ".Sigma1", defaultSigma1);
            const double sigma2 = iniConfig.getProperty
                                  <double>(name + ".Sigma2", defaultSigma2);

            const Kernel<T> kernel = DifferenceOfGaussianKernel
                <T>(sizeX, sizeY, sigma1, sigma2);
            return (positive) ? kernel : -kernel;
        } else if (kernelName == "Gabor") {
            const double theta = Utils::degToRad(iniConfig.getProperty
                                                 <double>(name + ".Theta"));
            const double sigma = iniConfig.getProperty
                                 <double>(name + ".Sigma", defaultSigma);
            const double lambda = iniConfig.getProperty
                                  <double>(name + ".Lambda", defaultLambda);
            const double psi = Utils::degToRad(
                iniConfig.getProperty<double>(name + ".Psi", defaultPsi));
            const double gamma = iniConfig.getProperty
                                 <double>(name + ".Gamma", defaultGamma);

            return GaborKernel
                <T>(sizeX, sizeY, theta, sigma, lambda, psi, gamma);
        } else
            throw std::runtime_error("Unknown kernel \"" + kernelName
                                     + "\" in section [" + section + "]");
    }
}

#endif // N2D2_KERNELGENERATOR_H
