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

#ifndef N2D2_INIPARSER_H
#define N2D2_INIPARSER_H

#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include "utils/Utils.hpp"
#include "utils/TemplateParser.hpp"

namespace N2D2 {
class IniParser {
public:
    /**
     * Constructor
    */
    IniParser();

    /**
     * Load an INI file.
     *
     * @param fileName          Name of the INI file
    */
    void load(const std::string& fileName);

    /**
     * Load from an input stream.
     *
     * @param data              Input stream
    */
    void load(std::istream& data);

    /**
     * Check if a section exists.
     *
     * @param name              Name of the section
     * @return True if the section exists
    */
    bool isSection(const std::string& name);

    /**
     * Return the number of sections in the INI file (at least 1, the implicit
     *section).
     *
     * @return Number of sections in the INI file
    */
    unsigned int getNbSections() const;

    const std::vector<std::string>& getSections() const;
    std::vector<std::string> getSections(const std::string& section) const;

    /**
     * Return an entire section.
     *
     * @param section           Name of the section
     * @param unreadOnly        Return only unread (and non-ignored) properties
     * @return Map of (property, value) pairs
    */
    std::map<std::string, std::string> getSection(const std::string& section,
                                                  bool unreadOnly = false);

    /**
     * Change the current section.
     *
     * @param section           Name of the section
     * @param checkForUnknown   Check if there is unkown properties in this
     *section
     * @return True if the section already existed.
    */
    bool currentSection(const std::string& section = "",
                        bool checkForUnknown = true);

    /**
     * Get the current section name.
     *
     * @return Name of the current section.
    */
    std::string getCurrentSection() const;

    /**
     * Return the number of properties in the current section.
     *
     * @return Number of properties in the current section
    */
    unsigned int getNbProperties() const;

    /**
     * Check if a property exists.
     *
     * @param name              Name of the property (support '*' wildcard)
     * @return True if the property exists
     *
     * @exception std::runtime_error Property not found in section
    */
    bool isProperty(const std::string& name);

    /**
     * Ignore a property (so it is not counted as unkown property in section
     *check for unknown).
     *
     * @param name              Name of the property (support '*' wildcard)
    */
    void ignoreProperty(const std::string& name);

    /**
     * Return the value associated to a property in the current section, and
     *throw an exception if the property is not found.
     *
     * @param name              Name of the property
     * @return Value associated to the property, in the desired type
     *
     * @exception std::runtime_error Property not found in section
    */
    template <class T> T getProperty(const std::string& name);

    /**
     * Return the value associated to a property in the current section.
     *
     * @param name              Name of the property
     * @param defaultValue      Default value to use if the property is not
     *found
     * @return Value associated to the property, in the desired type
    */
    template <class T>
    T getProperty(const std::string& name, const T& defaultValue);

    /**
     * Return all the sections containing a given property.
     *
     * @param name              Name of the property
     * @return Vector containing the name of the sections containing the
     *property
    */
    std::vector<std::string> getSectionsWithProperty(const std::string& name);

    /**
     * Return all the sections containing a given property and a given
     *associated value.
     *
     * @param name              Name of the property
     * @param requiredValue     Required value associated to the property
     * @return Vector containing the name of the sections containing the
     *(property, value) pair
    */
    template <class T>
    std::vector<std::string> getSectionsWithProperty(const std::string& name,
                                                     const T& requiredValue);

    /**
     * Erase an entire section.
     *
     * @param section           Name of the section
     * @return True if the section existed (meaning that we actually erased
     *something!)
    */
    bool eraseSection(const std::string& section);

    /**
     * Set the value associated to a property in the current section.
     *
     * @param name              Name of the property
     * @param value             Value associated to the property
    */
    template <class T>
    void setProperty(const std::string& name, const T& value);

    /**
     * Erase a property and its associated value in the current section.
     *
     * @param name              Name of the property
     * @return True if the property existed (meaning that we actually erased
     *something!)
    */
    bool eraseProperty(const std::string& name);

    /**
     * Save the current configuration in an INI file.
     *
     * @param fileName          Name of the INI file
    */
    void save(const std::string& fileName) const;

    const std::string& getFileName() const
    {
        return mFileName;
    };

    /// Destructor
    virtual ~IniParser();

private:
    std::string getPropertyValue(const std::string& value) const;
    void loadTplIni(const std::string& tplIni);

    std::string mFileName;
    unsigned int mCurrentSection;
    bool mCheckForUnknown;
    std::vector<std::string> mIniSections;
    std::vector<std::map<std::string, std::pair<std::string, bool> > > mIniData;
};
}

template <class T> T N2D2::IniParser::getProperty(const std::string& name)
{
    const std::map<std::string, std::pair<std::string, bool> >::iterator it
        = mIniData[mCurrentSection].find(name);

    if (it == mIniData[mCurrentSection].end())
        throw std::runtime_error("Property " + name + " not found in section ["
                                 + mIniSections[mCurrentSection] + "]");

    T value;
    std::stringstream strVal(getPropertyValue((*it).second.first));
    strVal.imbue(Utils::locale);

    if (!(Utils::signChecked<T>(strVal) >> value) || !strVal.eof())
        throw std::runtime_error("Unreadable property: " + name
                                 + " in section ["
                                 + mIniSections[mCurrentSection] + "]");

    (*it).second.second = true;

    return value;
}

template <class T>
T N2D2::IniParser::getProperty(const std::string& name, const T& defaultValue)
{
    const std::map<std::string, std::pair<std::string, bool> >::iterator it
        = mIniData[mCurrentSection].find(name);

    T value = defaultValue;

    if (it != mIniData[mCurrentSection].end()) {
        std::stringstream strVal(getPropertyValue((*it).second.first));
        strVal.imbue(Utils::locale);

        if (!(Utils::signChecked<T>(strVal) >> value) || !strVal.eof())
            throw std::runtime_error("Unreadable property: " + name
                                     + " in section ["
                                     + mIniSections[mCurrentSection] + "]");

        (*it).second.second = true;
    }

    return value;
}

template <class T>
std::vector<std::string>
N2D2::IniParser::getSectionsWithProperty(const std::string& name,
                                         const T& requiredValue)
{
    std::vector<std::string> sections;

    for (unsigned int section = 0, nbSections = mIniSections.size();
         section < nbSections;
         ++section) {
        const std::map<std::string, std::pair<std::string, bool> >::iterator it
            = mIniData[section].find(name);

        if (it != mIniData[section].end()) {
            T value;
            std::stringstream strVal((*it).second.first);
            strVal.imbue(Utils::locale);

            if ((Utils::signChecked<T>(strVal) >> value) && strVal.eof()
                && value == requiredValue) {
                sections.push_back(mIniSections[section]);
                (*it).second.second = true;
            }
        }
    }

    return sections;
}

template <class T>
void N2D2::IniParser::setProperty(const std::string& name, const T& value)
{
    std::stringstream strVal;
    strVal.imbue(Utils::locale);
    strVal << std::showpoint << value;

    mIniData[mCurrentSection][name] = std::make_pair(strVal.str(), true);
}

namespace N2D2 {
template <>
std::string IniParser::getProperty<std::string>(const std::string& name);

template <>
std::string IniParser::getProperty
    <std::string>(const std::string& name, const std::string& defaultValue);

template <>
std::vector<std::string>
IniParser::getSectionsWithProperty(const std::string& name,
                                   const std::string& requiredValue);
}

#endif // N2D2_INIPARSER_H
