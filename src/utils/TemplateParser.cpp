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

#include "utils/TemplateParser.hpp"

void N2D2::TemplateParser::Section::render(std::ostream& output,
                                           std::map
                                           <std::string, std::string>& params)
{
    for (std::vector<Section*>::iterator it = mSections.begin(),
                                         itEnd = mSections.end();
         it != itEnd;
         ++it)
        (*it)->render(output, params);
}

void N2D2::TemplateParser::Section::push_back(Section* section)
{
    mSections.push_back(section);
}

N2D2::TemplateParser::Section::~Section()
{
    std::for_each(mSections.begin(), mSections.end(), Utils::Delete());
}

void N2D2::TemplateParser::CodeSection::render(
    std::ostream& output, std::map<std::string, std::string>& params)
{
    size_t startPos = 0;
    size_t endPos = 0;

    size_t pos;
    while ((pos = mTemplateCode.find("{{", startPos)) != std::string::npos) {
        output << mTemplateCode.substr(endPos, pos - endPos);

        endPos = mTemplateCode.find("}}", pos + 2);

        if (endPos == std::string::npos)
            throw std::runtime_error("Missing closing braces near: "
                                     + mTemplateCode.substr(startPos, 20));

        const std::string varName
            = mTemplateCode.substr(pos + 2, endPos - pos - 2);

        std::map<std::string, std::string>::const_iterator it
            = params.find(varName);

        if (it == params.end())
            throw std::runtime_error("Undefined variable name: " + varName);

        output << (*it).second;

        endPos += 2;
        startPos = endPos;
    }

    output << mTemplateCode.substr(startPos, mTemplateCode.length() - startPos);
}

void N2D2::TemplateParser::IfSection::render(std::ostream& output,
                                             std::map
                                             <std::string, std::string>& params)
{
    std::map<std::string, std::string>::const_iterator it
        = params.find(mVarName);

    bool predicate;

    if (mOp == "exists")
        predicate = (it != params.end());
    else if (mOp == "not_exists")
        predicate = (it == params.end());
    else {
        if (it == params.end())
            throw std::runtime_error("Undefined variable name: " + mVarName
                                     + " in if statement");

        if (mOp == "==")
            predicate = ((*it).second == mValue);
        else if (mOp == "!=")
            predicate = ((*it).second != mValue);
        else
            throw std::runtime_error("Unknown operator: " + mOp
                                     + " in if statement");
    }

    if (predicate) {
        for (std::vector<Section*>::iterator it = mSections.begin(),
                                             itEnd = mSections.end();
             it != itEnd;
             ++it)
            (*it)->render(output, params);
    }
}

void N2D2::TemplateParser::ForSection::render(
    std::ostream& output, std::map<std::string, std::string>& params)
{
    if (params.find(mVarName) != params.end())
        throw std::runtime_error("Variable " + mVarName
                                 + " already exists in this context");

    double loopStart;
    double loopStop;

    std::map<std::string, std::string>::const_iterator itStart
        = params.find(mStart);
    std::stringstream value((itStart != params.end()) ? (*itStart).second
                                                      : mStart);

    if (!(value >> loopStart) || !value.eof())
        throw std::runtime_error("Unreadable loop range value: " + value.str());

    if (!mStop.empty()) {
        std::map<std::string, std::string>::const_iterator itStop
            = params.find(mStop);

        value.clear();
        value.str((itStop != params.end()) ? (*itStop).second : mStop);

        if (!(value >> loopStop) || !value.eof())
            throw std::runtime_error("Unreadable loop range value: "
                                     + value.str());
    } else {
        loopStop = loopStart;
        loopStart = 0;
    }

    for (double i = loopStart; i < loopStop; ++i) {
        std::stringstream iStr;
        iStr << i;

        params[mVarName] = iStr.str();

        for (std::vector<Section*>::iterator it = mSections.begin(),
                                             itEnd = mSections.end();
             it != itEnd;
             ++it)
            (*it)->render(output, params);
    }

    params.erase(mVarName);
}

void N2D2::TemplateParser::BlockSection::render(
    std::ostream& output, std::map<std::string, std::string>& params)
{
    std::map<std::string, std::string>::const_iterator it
        = params.find(mVarName);

    if (it == params.end())
        throw std::runtime_error("Undefined variable name: " + mVarName
                                 + " in block statement");

    int size;
    std::stringstream value((*it).second);

    if (!(value >> size) || !value.eof())
        throw std::runtime_error("Unreadable size value: " + (*it).second
                                 + " for block variable " + mVarName);

    for (int i = 0; i < size; ++i) {
        std::stringstream itemStr;
        itemStr << mVarName << "[" << i << "]";

        std::stringstream numStr;
        numStr << i;

        std::map<std::string, std::string> blockParams(params);
        blockParams["#"] = numStr.str();

        for (std::map<std::string, std::string>::const_iterator itVar
             = params.lower_bound(itemStr.str()),
             itVarEnd = params.end();
             itVar != itVarEnd;
             ++itVar) {
            // std::map elements are ordered
            if ((*itVar).first.compare(0, itemStr.str().length(), itemStr.str())
                != 0)
                break;

            const std::string varName
                = (*itVar).first.substr(itemStr.str().length());
            blockParams[varName] = (*itVar).second;
        }

        for (std::vector<Section*>::iterator it = mSections.begin(),
                                             itEnd = mSections.end();
             it != itEnd;
             ++it)
            (*it)->render(output, blockParams);
    }
}

void N2D2::TemplateParser::render(std::ostream& output,
                                  const std::string& source)
{
    Section rootSection;
    size_t endPos = processSection(source, 0, &rootSection);

    if (endPos != source.length())
        throw std::runtime_error("Source code remaining at end");

    rootSection.render(output, mParameters);
}

std::string N2D2::TemplateParser::renderFile(const std::string& fileName)
{
    std::stringstream output;
    renderFile(output, fileName);
    return output.str();
}

void N2D2::TemplateParser::renderFile(std::ostream& output,
                                      const std::string& fileName)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not open TPL file: " + fileName);

    std::stringstream buffer;
    buffer << data.rdbuf();
    render(output, buffer.str());
}

bool N2D2::TemplateParser::isParameter(const std::string& name) const {
    return (mParameters.find(name) != mParameters.end());
}

std::string N2D2::TemplateParser::getParameter(const std::string& name,
                                               bool ignoreNotExists) const
{
    const std::map<std::string, std::string>::const_iterator it
        = mParameters.find(name);

    if (it == mParameters.end()) {
        if (!ignoreNotExists) {
            throw std::runtime_error(
                "TemplateParser::getParameter(): Parameter does not exist: "
                + name);
        }

        return "";
    }
    else
        return (*it).second;
}

size_t N2D2::TemplateParser::processSection(const std::string& source,
                                            size_t startPos,
                                            Section* section)
{
    std::string ifVarName;
    std::string ifOp;
    std::string ifValue;

    while (true) {
        size_t controlStartPos = source.find("{%", startPos);

        if (controlStartPos == std::string::npos) {
            // Code only
            const std::string templateCode
                = source.substr(startPos, source.length() - startPos);
            CodeSection* codeSection = new CodeSection(templateCode);
            section->push_back(codeSection);

            if (dynamic_cast<BlockSection*>(section) != NULL)
                throw std::runtime_error("Missing endblock");

            if (dynamic_cast<ForSection*>(section) != NULL)
                throw std::runtime_error("Missing endfor");

            if (dynamic_cast<IfSection*>(section) != NULL)
                throw std::runtime_error("Missing endif");

            return source.length();
        } else {
            // Code, followed by control
            const std::string templateCode
                = source.substr(startPos, controlStartPos - startPos);
            CodeSection* codeSection = new CodeSection(templateCode);
            section->push_back(codeSection);

            // Control
            size_t controlEndPos = source.find("%}", controlStartPos + 2);

            if (controlEndPos == std::string::npos)
                throw std::runtime_error("Unterminated section");

            const std::string control = source.substr(
                controlStartPos + 2, controlEndPos - controlStartPos - 2);
            std::vector<std::string> controlArgs
                = Utils::split(control, " (),", true);

            if (controlArgs.empty())
                throw std::runtime_error("Empty control");

            if (controlArgs[0] == "endblock") {
                if (controlArgs.size() > 1)
                    throw std::runtime_error("Bad endblock control section");

                if (dynamic_cast<BlockSection*>(section) == NULL)
                    throw std::runtime_error("endblock while not in block!");

                return controlEndPos + 2;
            } else if (controlArgs[0] == "block") {
                if (controlArgs.size() != 2)
                    throw std::runtime_error(
                        "Bad block syntax control section");

                const std::string varName = controlArgs[1];

                BlockSection* blockSection = new BlockSection(varName);
                section->push_back(blockSection);

                startPos
                    = processSection(source, controlEndPos + 2, blockSection);
            } else if (controlArgs[0] == "endfor") {
                if (controlArgs.size() > 1)
                    throw std::runtime_error("Bad endfor control section");

                if (dynamic_cast<ForSection*>(section) == NULL)
                    throw std::runtime_error("endfor while not in for loop!");

                return controlEndPos + 2;
            } else if (controlArgs[0] == "for") {
                if (controlArgs.size() < 5)
                    throw std::runtime_error("Bad for syntax control section");

                const std::string loopVarName = controlArgs[1];

                if (controlArgs[2] != "in")
                    throw std::runtime_error(
                        "In control section {% " + control
                        + " unexpected: second word should be 'in'");

                if (controlArgs[3] != "range")
                    throw std::runtime_error(
                        "In control section {% " + control
                        + " unexpected: third word should be 'range'");

                const std::string loopStart = controlArgs[4];
                const std::string loopStop
                    = (controlArgs.size() > 5) ? controlArgs[5] : "";

                ForSection* forSection
                    = new ForSection(loopVarName, loopStart, loopStop);
                section->push_back(forSection);

                startPos
                    = processSection(source, controlEndPos + 2, forSection);
            } else if (controlArgs[0] == "endif") {
                if (controlArgs.size() > 1)
                    throw std::runtime_error("Bad endfor control section");

                if (dynamic_cast<IfSection*>(section) == NULL)
                    throw std::runtime_error(
                        "endif while not in if conditional section!");

                ifVarName.clear();

                return controlEndPos + 2;
            } else if (controlArgs[0] == "if") {
                if (controlArgs.size() != 3 && controlArgs.size() != 4)
                    throw std::runtime_error("Bad if syntax control section");

                ifVarName = controlArgs[1];
                ifOp = controlArgs[2];
                ifValue = (controlArgs.size() == 4) ? controlArgs[3] : "";

                IfSection* ifSection = new IfSection(ifVarName, ifOp, ifValue);
                section->push_back(ifSection);

                startPos = processSection(source, controlEndPos + 2, ifSection);
            } else if (controlArgs[0] == "else") {
                if (controlArgs.size() > 1)
                    throw std::runtime_error("Bad else control section");

                if (dynamic_cast<IfSection*>(section) == NULL) {
                    if (ifVarName.empty())
                        throw std::runtime_error(
                            "Found else control statement without if");

                    ifOp = (ifOp == "==") ? "!=" :
                           (ifOp == "!=") ? "==" :
                           (ifOp == "exists") ? "not_exists" :
                           (ifOp == "not_exists") ? "exists" :
                           "";

                    IfSection* ifSection
                        = new IfSection(ifVarName, ifOp, ifValue);
                    section->push_back(ifSection);

                    startPos
                        = processSection(source, controlEndPos + 2, ifSection);
                } else {
                    // Exit the inside of the if block
                    return controlStartPos;
                }
            } else if (controlArgs[0] == "include") {
                const std::string fileName = controlArgs[1];

                std::ifstream incTempl(fileName.c_str());

                if (!incTempl.good())
                    throw std::runtime_error("Could not open template file: "
                                             + fileName);

                const std::string templ(
                    (std::istreambuf_iterator<char>(incTempl)),
                    std::istreambuf_iterator<char>());
                incTempl.close();

                size_t endPos = processSection(templ, 0, section);

                if (endPos != templ.length())
                    throw std::runtime_error(
                        "Source code remaining at end for included file: "
                        + fileName);

                startPos = controlEndPos + 2;
            } else
                throw std::runtime_error("Unknown control section");
        }
    }
}
