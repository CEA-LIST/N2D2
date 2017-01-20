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

#include "utils/Parameterizable.hpp"

N2D2::Parameter_T& N2D2::Parameter_T::operator=(const N2D2::Parameter_T& value)
{
    if (value.mCopy != NULL)
        (this->*(this->mCopy))(value);

    return *this;
}

namespace N2D2 {
std::ostream& operator<<(std::ostream& os, const Parameter_T& value)
{
    return (value.mPrint != NULL) ? (value.*(value.mPrint))(os) : (os << "nil");
}
}

namespace N2D2 {
std::istream& operator>>(std::istream& is, const Parameter_T& value)
{
    return (value.mRead != NULL) ? (value.*(value.mRead))(is) : is;
}
}

bool N2D2::Parameterizable::isParameter(const std::string& name) const
{
    return (mParameters.find(name) != mParameters.end());
}

void N2D2::Parameterizable::setParameter(const std::string& name,
                                         const std::string& value)
{
    if (mParameters.find(name) != mParameters.end()) {
        std::stringstream valueStr(value);
        valueStr.imbue(Utils::locale);

        if (!(valueStr >> *mParameters[name]) || !valueStr.eof())
            throw std::runtime_error("Unreadable parameter: " + name);
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

unsigned int N2D2::Parameterizable::setParameters(
    const std::map<std::string, std::string>& params, bool ignoreUnknown)
{
    unsigned int nbLoaded = 0;

    for (std::map<std::string, std::string>::const_iterator it = params.begin(),
                                                            itEnd
                                                            = params.end();
         it != itEnd;
         ++it) {
        if (mParameters.find((*it).first) != mParameters.end()) {
            std::stringstream value((*it).second);
            value.imbue(Utils::locale);

            bool readOk = false;

            try
            {
                readOk = static_cast<bool>(value >> *mParameters[(*it).first]);
            }
            catch (const std::exception& /*e*/)
            {
                std::cout << Utils::cwarning
                          << "Unreadable value for parameter \"" << (*it).first
                          << "\":\n"
                             "\"" << (*it).second << "\"" << Utils::cdef
                          << std::endl;
                throw;
            }

            if (!readOk || !value.eof())
                throw std::runtime_error("Unreadable parameter: "
                                         + (*it).first);

            ++nbLoaded;
        } else {
            if (ignoreUnknown)
                std::cout << "Notice: Unknown parameter: " << (*it).first
                          << std::endl;
            else
                throw std::runtime_error("Unknown parameter: " + (*it).first);
        }
    }

    return nbLoaded;
}

unsigned int
N2D2::Parameterizable::setPrefixedParameters(std::map
                                             <std::string, std::string>& params,
                                             const std::string& prefix,
                                             bool greedy,
                                             bool ignoreUnknown)
{
    unsigned int nbLoaded = 0;
    std::map<std::string, std::string>::iterator it = params.begin();

    while (it != params.end()) {
        if ((*it).first.find(prefix) == 0) {
            const std::string name = (*it).first.substr(prefix.length());

            if (mParameters.find(name) != mParameters.end()) {
                std::stringstream value((*it).second);
                value.imbue(Utils::locale);

                bool readOk = false;

                try
                {
                    readOk = static_cast<bool>(value >> *mParameters[name]);
                }
                catch (const std::exception& /*e*/)
                {
                    std::cout << Utils::cwarning
                              << "Unreadable value for parameter \""
                              << (*it).first << "\":\n"
                                                "\"" << (*it).second << "\""
                              << Utils::cdef << std::endl;
                    throw;
                }

                if (!readOk || !value.eof())
                    throw std::runtime_error("Unreadable parameter: "
                                             + (*it).first);

                ++nbLoaded;

                if (greedy) {
                    params.erase(it++);
                    continue;
                }
            } else {
                if (ignoreUnknown)
                    std::cout << "Notice: Unknown parameter: " << (*it).first
                              << std::endl;
                else
                    throw std::runtime_error("Unknown parameter: "
                                             + (*it).first);
            }
        }

        ++it;
    }

    return nbLoaded;
}

unsigned int
N2D2::Parameterizable::setPrefixedParameters(const std::map
                                             <std::string, std::string>& params,
                                             const std::string& prefix,
                                             bool ignoreUnknown)
{
    std::map<std::string, std::string> paramsCopy(params);
    return setPrefixedParameters(paramsCopy, prefix, false, ignoreUnknown);
}

std::string N2D2::Parameterizable::getParameter(const std::string& name) const
{
    const std::map<std::string, Parameter_T*>::const_iterator it
        = mParameters.find(name);

    if (it != mParameters.end()) {
        std::ostringstream value;
        value << std::showpoint << (*((*it).second));
        return value.str();
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

unsigned int N2D2::Parameterizable::loadParameters(const std::string& fileName,
                                                   bool ignoreNotExists,
                                                   bool ignoreUnknown)
{
    std::ifstream cfg(fileName.c_str());

    if (!cfg.good()) {
        if (ignoreNotExists)
            std::cout << "Notice: Could not open configuration file: "
                      << fileName << std::endl;
        else
            throw std::runtime_error("Could not open configuration file: "
                                     + fileName);
    }

    unsigned int nbLoaded = 0;
    std::string line;

    while (std::getline(cfg, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '#'), line.end());
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        const size_t posEq = line.find_first_of('=');
        std::string name = line.substr(0, posEq); // if '=' is not found, name =
        // line and assume that the
        // value part is missing
        // Right trim
        name.erase(std::find_if(name.rbegin(),
                                name.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   name.end());

        if (posEq == std::string::npos)
            throw std::runtime_error("Missing value for parameter: " + name
                                     + " in config file " + fileName);

        if (mParameters.find(name) != mParameters.end()) {
            std::stringstream value(line.substr(posEq + 1)); // if posEq is
            // string::npos,
            // posEq + 1
            // overflows and
            // value = line
            // This is very problematic if the parameter is a string, that's why
            // an exception is thrown if posEq = string::npos
            value.imbue(Utils::locale);

            bool readOk = false;

            try
            {
                readOk = static_cast<bool>(value >> *mParameters[name]);
            }
            catch (const std::exception& /*e*/)
            {
                std::cout << Utils::cwarning
                          << "Unreadable value for parameter \"" << name
                          << "\" in config file " << fileName << ":\n"
                                                                 "\""
                          << value.str() << "\"" << Utils::cdef << std::endl;
                throw;
            }

            if (!readOk || !value.eof())
                throw std::runtime_error("Unreadable parameter: " + name
                                         + " in config file " + fileName);

            ++nbLoaded;
        } else {
            if (ignoreUnknown)
                std::cout << "Notice: Unknown parameter: " << name
                          << " in config file " << fileName << std::endl;
            else
                throw std::runtime_error("Unknown parameter: " + name
                                         + " in config file " + fileName);
        }
    }

    return nbLoaded;
}

void N2D2::Parameterizable::saveParameters(const std::string& fileName) const
{
    std::ofstream cfg(fileName.c_str());

    if (!cfg.good())
        throw std::runtime_error("Could not create configuration file: "
                                 + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    cfg << "# "
        << std::asctime(localNow); // std::asctime() already appends end of line
    cfg.imbue(Utils::locale);

    for (std::map<std::string, Parameter_T*>::const_iterator it
         = mParameters.begin(),
         itEnd = mParameters.end();
         it != itEnd;
         ++it) {
        // Dans le cas d'un nombre décimal, on ajoute systématiquement la
        // virgule, ce qui permet de toujours correctement déduire le
        // type du paramètre (entier ou réel) à la lecture du fichier.
        cfg << (*it).first << " = " << std::showpoint << (*((*it).second))
            << "\n";
    }
}

void N2D2::Parameterizable::copyParameters(const Parameterizable& from)
{
    for (std::map<std::string, Parameter_T*>::const_iterator it
         = from.mParameters.begin(),
         itEnd = from.mParameters.end();
         it != itEnd;
         ++it) {
        (*mParameters.at((*it).first)) = (*((*it).second));
    }
}
