/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "utils/IniParser.hpp"

N2D2::IniParser::IniParser() : mCheckForUnknown(false)
{
    currentSection("", false); // Create the global (default) section
}

void N2D2::IniParser::load(const std::string& fileName)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not open INI file: " + fileName);

    mFileName = fileName;
    load(data);
}

void N2D2::IniParser::load(std::istream& data)
{
    std::string line;
    std::string preLine;
    currentSection("", false); // Make sure the global (default) section exists
    std::string tplIni;

    while (std::getline(data, line)) {
        // Support for escaped new line
        if (!line.empty() && *(line.rbegin()) == '\\') {
            preLine.append(line.substr(0, line.size() - 1));
            continue;
        } else if (!preLine.empty()) {
            line.insert(0, preLine);
            preLine.clear();
        }

        // Remove optional comments
        std::size_t delim = 0;

        while ((delim = line.find_first_of("\";#", delim))
               != std::string::npos) {
            if (line[delim] == '"') {
                // Beginning of quote block, find the end
                delim = line.find_first_of("\"", delim + 1);

                if (delim == std::string::npos) {
                    throw std::runtime_error("Malformed property in section ["
                                             + mIniSections[mCurrentSection]
                                             + "] in INI file " + mFileName);
                }

                ++delim;
            } else
                line.erase(delim);
        }

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

        if (*(line.begin()) == '[' && *(line.rbegin()) == ']') {
            if (!tplIni.empty()) {
                // Process templated sub INI
                loadTplIni(tplIni);
                tplIni.clear();
            }

            std::string section = line.substr(1, line.size() - 2);

            // Check for templated sub INI
            const std::vector<std::string> sectionSplit
                = Utils::split(section, "@");

            if (sectionSplit.size() == 2) {
                section = sectionSplit[0];

                const std::string fileName
                    = Utils::expandEnvVars(sectionSplit[1]);

                tplIni = (Utils::isAbsolutePath(fileName))
                    ? fileName
                    : Utils::dirName(mFileName) + "/" + fileName;
            }

            std::vector<std::string>::const_iterator itSection
                = std::find(mIniSections.begin(), mIniSections.end(), section);

            if (itSection != mIniSections.end())
                throw std::runtime_error("Section [" + section
                                         + "] already exists in INI file "
                                         + mFileName);

            mCurrentSection = mIniSections.size();
            mIniSections.push_back(section);
            mIniData.push_back(std::map
                               <std::string, std::pair<std::string, bool> >());

            continue;
        }

        const size_t posEq = line.find_first_of('=');
        std::string property
            = line.substr(0, posEq); // if '=' is not found, property = line and
        // assume that the value part is
        // missing
        // Right trim
        property.erase(std::find_if(property.rbegin(),
                                    property.rend(),
                                    std::not1(std::ptr_fun
                                              <int, int>(std::isspace))).base(),
                       property.end());

        if (posEq == std::string::npos)
            throw std::runtime_error("Missing value for property: " + property
                                     + " in INI file " + mFileName);

        if (mIniData[mCurrentSection].find(property)
            != mIniData[mCurrentSection].end()) {
            throw std::runtime_error(
                "Property " + property + " already exists in section ["
                + mIniSections[mCurrentSection] + "] in INI file " + mFileName);
        }

        std::string value = line.substr(posEq + 1);
        // Left trim
        value.erase(
            value.begin(),
            std::find_if(value.begin(),
                         value.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));

        // Remove quotes from protected property value
        if (*(value.begin()) == '"' && *(value.rbegin()) == '"')
            value = value.substr(1, value.size() - 2);

        // Debug
        // std::cout << "[" << mCurrentSection << "] \"" << property << "\" =
        // \"" << value << "\"" << std::endl;

        mIniData[mCurrentSection][property] = std::make_pair(value, false);
        // if posEq is string::npos, posEq + 1 overflows and value = line. This
        // is very problematic if the parameter is a string,
        // that's why an exception is thrown if posEq = string::npos
    }

    if (!tplIni.empty()) {
        // Process templated sub INI
        loadTplIni(tplIni);
        tplIni.clear();
    }

    // Make sure the global (default) section is selected
    mCurrentSection = 0;
}

bool N2D2::IniParser::isSection(const std::string& name)
{
    const std::vector<std::string>::const_iterator itSection
        = std::find(mIniSections.begin(), mIniSections.end(), name);
    return (itSection != mIniSections.end());
}

unsigned int N2D2::IniParser::getNbSections() const
{
    return mIniData.size();
}

const std::vector<std::string>& N2D2::IniParser::getSections() const
{
    return mIniSections;
}

std::vector<std::string> N2D2::IniParser::getSections(const std::string
                                                      & section) const
{
    std::vector<std::string> sections;

    for (std::vector<std::string>::const_iterator it = mIniSections.begin(),
                                                  itEnd = mIniSections.end();
         it != itEnd;
         ++it) {
        if (Utils::match(section, *it))
            sections.push_back(*it);
    }

    return sections;
}

std::map<std::string, std::string>
N2D2::IniParser::getSection(const std::string& section, bool unreadOnly)
{
    std::vector<std::string>::const_iterator itSection
        = std::find(mIniSections.begin(), mIniSections.end(), section);

    if (itSection == mIniSections.end())
        throw std::runtime_error("Section [" + section + "] not found");

    const unsigned int sectionIndex = itSection - mIniSections.begin();

    std::map<std::string, std::string> properties;

    for (std::map<std::string, std::pair<std::string, bool> >::iterator it
         = mIniData[sectionIndex].begin(),
         itEnd = mIniData[sectionIndex].end();
         it != itEnd;
         ++it) {
        if ((unreadOnly && !(*it).second.second) || !unreadOnly) {
            properties[(*it).first] = getPropertyValue((*it).second.first);
            (*it).second.second = true;
        }
    }

    return properties;
}

bool N2D2::IniParser::currentSection(const std::string& section,
                                     bool checkForUnknown)
{
    // Check for unknown property in the previous section
    if (mCheckForUnknown) {
        for (std::map
             <std::string, std::pair<std::string, bool> >::const_iterator it
             = mIniData[mCurrentSection].begin(),
             itEnd = mIniData[mCurrentSection].end();
             it != itEnd;
             ++it) {
            if (!(*it).second.second)
                throw std::runtime_error("Unknown property " + (*it).first
                                         + " in section ["
                                         + mIniSections[mCurrentSection] + "]");
        }
    }

    mCheckForUnknown = checkForUnknown;

    // Select the new section
    std::vector<std::string>::const_iterator itSection
        = std::find(mIniSections.begin(), mIniSections.end(), section);

    if (itSection == mIniSections.end()) {
        mIniSections.push_back(section);
        mIniData.push_back(std::map
                           <std::string, std::pair<std::string, bool> >());
        mCurrentSection = mIniSections.size() - 1;
        return false;
    } else {
        mCurrentSection = itSection - mIniSections.begin();
        return true;
    }
}

std::string N2D2::IniParser::getCurrentSection() const
{
    return mIniSections[mCurrentSection];
}

bool N2D2::IniParser::isProperty(const std::string& name)
{
    const size_t posWildcard = name.find("*");

    if (posWildcard != std::string::npos) {
        const std::string prefixMask = name.substr(0, posWildcard);
        const std::string suffixMask = name.substr(posWildcard + 1);

        for (std::map<std::string, std::pair<std::string, bool> >::iterator it
             = mIniData[mCurrentSection].begin(),
             itEnd = mIniData[mCurrentSection].end();
             it != itEnd;
             ++it) {
            if ((prefixMask.empty() || (*it).first.find(prefixMask) == 0)
                && (suffixMask.empty() || (*it).first.find(suffixMask)
                                          == (*it).first.length()
                                             - suffixMask.length())) {
                return true;
            }
        }

        return false;
    } else {
        std::map<std::string, std::pair<std::string, bool> >::iterator it
            = mIniData[mCurrentSection].find(name);

        return (it != mIniData[mCurrentSection].end());
    }
}

void N2D2::IniParser::ignoreProperty(const std::string& name)
{
    const size_t posWildcard = name.find("*");

    if (posWildcard != std::string::npos) {
        const std::string prefixMask = name.substr(0, posWildcard);
        const std::string suffixMask = name.substr(posWildcard + 1);

        for (std::map<std::string, std::pair<std::string, bool> >::iterator it
             = mIniData[mCurrentSection].begin(),
             itEnd = mIniData[mCurrentSection].end();
             it != itEnd;
             ++it) {
            if ((prefixMask.empty() || (*it).first.find(prefixMask) == 0)
                && (suffixMask.empty() || (*it).first.find(suffixMask)
                                          == (*it).first.length()
                                             - suffixMask.length())) {
                (*it).second.second = true;
            }
        }
    } else {
        std::map<std::string, std::pair<std::string, bool> >::iterator it
            = mIniData[mCurrentSection].find(name);

        if (it != mIniData[mCurrentSection].end())
            (*it).second.second = true;
    }
}

std::vector<std::string>
N2D2::IniParser::getSectionsWithProperty(const std::string& name)
{
    std::vector<std::string> sections;

    const size_t posWildcard = name.find("*");

    if (posWildcard != std::string::npos) {
        const std::string prefixMask = name.substr(0, posWildcard);
        const std::string suffixMask = name.substr(posWildcard + 1);

        for (unsigned int section = 0, nbSections = mIniSections.size();
             section < nbSections;
             ++section) {
            for (std::map<std::string, std::pair<std::string, bool> >::iterator
                     it = mIniData[section].begin(),
                     itEnd = mIniData[section].end();
                 it != itEnd;
                 ++it) {
                if ((prefixMask.empty() || (*it).first.find(prefixMask) == 0)
                    && (suffixMask.empty() || (*it).first.find(suffixMask)
                                              == (*it).first.length()
                                                 - suffixMask.length())) {
                    (*it).second.second = true;
                }
            }
        }
    } else {
        for (unsigned int section = 0, nbSections = mIniSections.size();
             section < nbSections;
             ++section) {
            const std::map
                <std::string, std::pair<std::string, bool> >::iterator it
                = mIniData[section].find(name);

            if (it != mIniData[section].end()) {
                sections.push_back(mIniSections[section]);
                (*it).second.second = true;
            }
        }
    }

    return sections;
}

bool N2D2::IniParser::eraseSection(const std::string& section)
{
    if (section == mIniSections[mCurrentSection]) {
        // If the section to erase is the current section, just empty it
        mIniData[mCurrentSection].clear();
        return true;
    } else if (section.empty()) {
        // The global (default) section is always the first and cannot be
        // erased, just emptied
        assert(mIniSections[0].empty());
        mIniData[0].clear();
        return true;
    } else {
        std::vector<std::string>::iterator itSection
            = std::find(mIniSections.begin(), mIniSections.end(), section);

        if (itSection != mIniSections.end()) {
            const unsigned int sectionIndex = itSection - mIniSections.begin();
            mIniSections.erase(itSection);
            mIniData.erase(mIniData.begin() + sectionIndex);
            return true;
        }

        return false;
    }
}

unsigned int N2D2::IniParser::getNbProperties() const
{
    return mIniData[mCurrentSection].size();
}

bool N2D2::IniParser::eraseProperty(const std::string& name)
{
    return (mIniData[mCurrentSection].erase(name) > 0);
}

void N2D2::IniParser::save(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not open INI file for writing: "
                                 + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    data << "; " << std::asctime(
                        localNow); // std::asctime() already appends end of line
    data.imbue(Utils::locale);

    for (unsigned int section = 0, nbSections = mIniSections.size();
         section < nbSections;
         ++section) {
        data << "[" << mIniSections[section] << "]"
             << "\n";

        for (std::map
             <std::string, std::pair<std::string, bool> >::const_iterator it
             = mIniData[section].begin(),
             itEnd = mIniData[section].end();
             it != itEnd;
             ++it) {
            data << (*it).first << "=" << (*it).second.first << "\n";
        }

        data << "\n";
    }
}

namespace N2D2 {
template <>
std::string IniParser::getProperty<std::string>(const std::string& name)
{
    const std::map<std::string, std::pair<std::string, bool> >::iterator it
        = mIniData[mCurrentSection].find(name);

    if (it == mIniData[mCurrentSection].end())
        throw std::runtime_error("Property " + name + " not found in section ["
                                 + mIniSections[mCurrentSection] + "]");

    (*it).second.second = true;
    return getPropertyValue((*it).second.first);
}

template <>
std::string IniParser::getProperty
    <std::string>(const std::string& name, const std::string& defaultValue)
{
    const std::map<std::string, std::pair<std::string, bool> >::iterator it
        = mIniData[mCurrentSection].find(name);

    std::string value(defaultValue);

    if (it != mIniData[mCurrentSection].end()) {
        value = getPropertyValue((*it).second.first);
        (*it).second.second = true;
    }

    return value;
}

template <>
std::vector<std::string>
IniParser::getSectionsWithProperty(const std::string& name,
                                   const std::string& requiredValue)
{
    std::vector<std::string> sections;

    for (unsigned int section = 0, nbSections = mIniSections.size();
         section < nbSections;
         ++section) {
        const std::map<std::string, std::pair<std::string, bool> >::iterator it
            = mIniData[section].find(name);

        if (it != mIniData[section].end() && (*it).second.first
                                             == requiredValue) {
            sections.push_back(mIniSections[section]);
            (*it).second.second = true;
        }
    }

    return sections;
}
}

std::string N2D2::IniParser::getPropertyValue(const std::string& value) const
{
    // Value is actually identical to another property value in the INI file
    if (!value.empty() && *(value.begin()) == '[') {
        const size_t posCloseBracket = value.find_first_of(']', 1);

        if (posCloseBracket != std::string::npos) {
            const std::string sectionName
                = value.substr(1, posCloseBracket - 1);
            std::vector<std::string>::const_iterator itSection = std::find(
                mIniSections.begin(), mIniSections.end(), sectionName);

            if (itSection == mIniSections.end())
                throw std::runtime_error("Section [" + sectionName
                                         + "] does not exist in INI file "
                                         + mFileName);

            const unsigned int sectionIndex = itSection - mIniSections.begin();
            const std::string propertyName = value.substr(posCloseBracket + 1);

            const std::map
                <std::string, std::pair<std::string, bool> >::const_iterator it
                = mIniData[sectionIndex].find(propertyName);

            if (it == mIniData[sectionIndex].end()) {
                throw std::runtime_error("Property " + propertyName
                                         + " not found in section ["
                                         + mIniSections[sectionIndex] + "]");
            }

            return getPropertyValue((*it).second.first);
        }
    }

    return value;
}

void N2D2::IniParser::loadTplIni(const std::string& tplIni) {
    // Process templated sub INI
    TemplateParser parser;
    parser.addParameter("SECTION_NAME",
                        mIniSections[mCurrentSection]);
    parser.addParameter("SECTION_FILE_NAME", tplIni);

    for (std::map
         <std::string, std::pair<std::string, bool> >
         ::const_iterator it = mIniData[mCurrentSection].begin(),
         itEnd = mIniData[mCurrentSection].end();
         it != itEnd;
         ++it)
    {
        parser.addParameter((*it).first, (*it).second.first);
    }

    mCurrentSection = 0;

    const std::string parentFileName = mFileName;
    std::istringstream str(parser.renderFile(tplIni));
    load(str);
    mFileName = parentFileName;
}

N2D2::IniParser::~IniParser()
{
    for (unsigned int section = 0, nbSections = mIniSections.size();
         section < nbSections;
         ++section) {
        bool unknownSection = true;

        for (std::map
             <std::string, std::pair<std::string, bool> >::const_iterator it
             = mIniData[section].begin(),
             itEnd = mIniData[section].end();
             it != itEnd;
             ++it) {
            if ((*it).second.second) {
                unknownSection = false;
                break;
            }
        }

        if (unknownSection)
            std::cout << "Notice: Unused section " + mIniSections[section]
                         + " in INI file" << std::endl;
    }
}
