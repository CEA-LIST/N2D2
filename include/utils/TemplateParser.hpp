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

#ifndef N2D2_TEMPLATEPARSER_H
#define N2D2_TEMPLATEPARSER_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "utils/Utils.hpp"

namespace N2D2 {
class TemplateParser {
public:
    class Section {
    public:
        virtual void render(std::ostream& output,
                            std::map<std::string, std::string>& params);
        void push_back(Section* section);
        virtual ~Section();

    protected:
        std::vector<Section*> mSections;
    };

    class CodeSection : public Section {
    public:
        CodeSection(const std::string& templateCode)
            : mTemplateCode(templateCode)
        {
        }
        void render(std::ostream& output,
                    std::map<std::string, std::string>& params);

    private:
        const std::string mTemplateCode;
    };

    class IfSection : public Section {
    public:
        IfSection(const std::string& varName,
                  const std::string& op,
                  const std::string& value)
            : mVarName(varName), mOp(op), mValue(value)
        {
        }
        void render(std::ostream& output,
                    std::map<std::string, std::string>& params);

    private:
        const std::string mVarName;
        const std::string mOp;
        const std::string mValue;
    };

    class ForSection : public Section {
    public:
        ForSection(const std::string& varName,
                   const std::string& start,
                   const std::string& stop)
            : mVarName(varName), mStart(start), mStop(stop)
        {
        }
        void render(std::ostream& output,
                    std::map<std::string, std::string>& params);

    private:
        const std::string mVarName;
        const std::string mStart;
        const std::string mStop;
    };

    class BlockSection : public Section {
    public:
        BlockSection(const std::string& varName) : mVarName(varName)
        {
        }
        void render(std::ostream& output,
                    std::map<std::string, std::string>& params);

    private:
        const std::string mVarName;
    };

    template <class T>
    void addParameter(const std::string& name, const T& value);
    template <class T>
    void appendToParameter(const std::string& name,
                           const T& value,
                           const std::string& separator = "");
    template <class T>
    void addBlockParameter(const std::string& blockName,
                           const std::string& name,
                           const T& value);
    template <class T>
    void setBlockParameter(const std::string& blockName,
                           unsigned int num,
                           const std::string& name,
                           const T& value);
    template <class T>
    void setParameter(const std::string& name,
                      const T& value,
                      bool ignoreNotExists = false);
    bool isParameter(const std::string& name) const;
    std::string getParameter(const std::string& name,
                             bool ignoreNotExists = false) const;
    void render(std::ostream& output, const std::string& source);
    std::string renderFile(const std::string& fileName);
    void renderFile(std::ostream& output, const std::string& fileName);

private:
    size_t processSection(const std::string& source,
                          size_t startPos,
                          Section* section);

    std::map<std::string, std::string> mParameters;
};
}

template <class T>
void N2D2::TemplateParser::addParameter(const std::string& name, const T& value)
{
    std::ostringstream valueStr;
    valueStr << value;

    bool newInsert;
    std::tie(std::ignore, newInsert)
        = mParameters.insert(std::make_pair(name, valueStr.str()));

    if (!newInsert)
        throw std::runtime_error(
            "TemplateParser::addParameter(): Parameter already exists: "
            + name);
}

template <class T>
void N2D2::TemplateParser::appendToParameter(const std::string& name,
                                             const T& value,
                                             const std::string& separator)
{
    std::ostringstream valueStr;
    valueStr << value;

    std::map<std::string, std::string>::iterator it;

    bool newInsert;
    std::tie(it, newInsert)
        = mParameters.insert(std::make_pair(name, valueStr.str()));

    if (!newInsert)
        (*it).second+= separator + valueStr.str();
}

template <class T>
void N2D2::TemplateParser::addBlockParameter(const std::string& blockName,
                                             const std::string& name,
                                             const T& value)
{
    std::ostringstream nameStr, valueStr;
    valueStr << value;

    // Find next num
    unsigned int num = 0;

    while (true) {
        nameStr.str(std::string());
        nameStr << blockName << "[" << num << "]" << "." << name;

        std::map<std::string, std::string>::iterator it
            = mParameters.find(nameStr.str());

        if (it == mParameters.end())
            break;

        ++num;
    }

    bool newInsert;
    std::tie(std::ignore, newInsert)
        = mParameters.insert(std::make_pair(nameStr.str(), valueStr.str()));

    setParameter(blockName, num + 1, true);

    assert(newInsert);
}

template <class T>
void N2D2::TemplateParser::setBlockParameter(const std::string& blockName,
                                             unsigned int num,
                                             const std::string& name,
                                             const T& value)
{
    std::ostringstream nameStr, valueStr;
    nameStr << blockName << "[" << num << "]." << name;
    valueStr << value;

    bool newInsert;
    std::tie(std::ignore, newInsert)
        = mParameters.insert(std::make_pair(nameStr.str(), valueStr.str()));

    if (!newInsert)
        throw std::runtime_error(
            "TemplateParser::addParameter(): Parameter already exists: "
            + nameStr.str());
}

template <class T>
void N2D2::TemplateParser::setParameter(const std::string& name,
                                        const T& value,
                                        bool ignoreNotExists)
{
    std::ostringstream valueStr;
    valueStr << value;

    std::map<std::string, std::string>::iterator it = mParameters.find(name);

    if (it == mParameters.end()) {
        if (ignoreNotExists)
            mParameters.insert(std::make_pair(name, valueStr.str()));
        else {
            throw std::runtime_error(
                "TemplateParser::setParameter(): Parameter does not exist: "
                + name);
        }
    }
    else
        (*it).second = valueStr.str();
}

#endif // N2D2_TEMPLATEPARSER_H
