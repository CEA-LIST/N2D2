/*
    (C) Copyright 2012 CEA LIST. All Rights Reserved.
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

#include "utils/ProgramOptions.hpp"

N2D2::ProgramOptions::ProgramOptions(int argc,
                                     char* argv[],
                                     bool abortOnError,
                                     bool saveArgs)
    : mAbortOnError(abortOnError), mCommand(argv[0])
{
    std::copy(argv + 1, argv + argc, std::back_inserter(mArgs));

    if (saveArgs) {
        // Save argument list
        std::ostringstream fileName;
        fileName << argv[0] << ".args";

        std::ofstream data(fileName.str().c_str(), std::ofstream::app);

        if (!data.good())
            throw std::runtime_error("Could not create args file: "
                                     + fileName.str());

        std::copy(argv, argv + argc, std::ostream_iterator<char*>(data, " "));
        data << "\n";
        data.close();
    }

    mHelp = parse("-h", "show possible options and exit");
}

bool N2D2::ProgramOptions::parse(const std::string& option,
                                 const std::string& desc)
{
    bool value = false;
    std::vector<std::string>::iterator it
        = std::find(mArgs.begin(), mArgs.end(), option);

    if (it != mArgs.end()) {
        value = true;
        mArgs.erase(it);

        std::cout << "Option " << option << ": " << desc << std::endl;
    }

    bool newInsert;
    std::tie(std::ignore, newInsert)
        = mOptions.insert(std::make_pair(option, std::make_pair(desc, "")));

    if (!newInsert)
        throw std::runtime_error("Option already parsed: " + option);

    return value;
}

std::map<std::string, std::string>
N2D2::ProgramOptions::grab(const std::string& prefix, const std::string& desc)
{
    std::map<std::string, std::string> params;
    std::vector<std::string>::iterator it = mArgs.begin();
    const unsigned len = prefix.size();

    while (it != mArgs.end() && it + 1 != mArgs.end()) {
        if ((*it).compare(0, len, prefix) == 0) {
            const std::string name((*it).substr(len));
            it = mArgs.erase(it);

            bool newInsert;
            std::tie(std::ignore, newInsert)
                = params.insert(std::make_pair(name, *it));

            if (!newInsert || mOptions.find(name) != mOptions.end())
                throw std::runtime_error("Option already parsed: " + prefix
                                         + name);

            std::cout << "Option " << prefix << ": " << desc << " [" << name
                      << "=" << (*it) << "]" << std::endl;
            it = mArgs.erase(it);
        } else
            ++it;
    }

    bool newInsert;
    std::tie(std::ignore, newInsert)
        = mOptions.insert(std::make_pair(prefix, std::make_pair(desc, "")));

    if (!newInsert)
        throw std::runtime_error("Option already parsed: " + prefix);

    return params;
}

void N2D2::ProgramOptions::done()
{
    if (!mArgs.empty()) {
        std::cout << "Unknown option(s): ";
        std::copy(mArgs.begin(),
                  mArgs.end(),
                  std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << std::endl;

        mArgs.clear();

        if (mAbortOnError)
            mHelp = true;
    }

    if (mHelp) {
        std::cout << "Usage: " << Utils::baseName(mCommand) << " ";

        for (std::vector
             <std::tuple
              <std::string, std::string, std::string> >::const_iterator it
             = mPosOptions.begin(),
             itEnd = mPosOptions.end();
             it != itEnd;
             ++it) {
            std::cout << std::get<0>(*it) << " ";
        }

        std::cout << "[options] ..." << std::endl;

        for (std::vector
             <std::tuple
              <std::string, std::string, std::string> >::const_iterator it
             = mPosOptions.begin(),
             itEnd = mPosOptions.end();
             it != itEnd;
             ++it) {
            if (std::get<1>(*it) != "" || std::get<2>(*it) != "")
                displayHelp(
                    std::get<0>(*it), std::get<1>(*it), std::get<2>(*it));
        }

        std::cout << "Options:" << std::endl;

        for (std::map
             <std::string, std::pair<std::string, std::string> >::const_iterator
                 it = mOptions.begin(),
                 itEnd = mOptions.end();
             it != itEnd;
             ++it) {
            displayHelp((*it).first, (*it).second.first, (*it).second.second);
        }

        std::cout << std::endl;
        std::exit(0);
    }

    // Clear useless memory
    mOptions.clear();
    mPosOptions.clear();
}

void N2D2::ProgramOptions::displayHelp(const std::string& name,
                                       const std::string& desc,
                                       const std::string& defaultValue)
{
    std::cout << "  " << name;

    if (name.size() < 18)
        std::cout << std::string(18 - name.size(), ' ');
    else
        std::cout << std::endl << std::string(20, ' ');

    const std::string descFull
        = (defaultValue != "") ? desc + " [" + defaultValue + "]" : desc;

    // Word-wrap to display in the console
    for (unsigned int i = 0, start = 0, space = 0, len = descFull.size();
         i != len;
         ++i) {
        if (descFull[i] == '\n') {
            if (start > 0)
                std::cout << std::endl << std::string(20, ' ');

            std::cout << descFull.substr(start, i - start); // no +1, the \n
            // character will be
            // part of the next
            // std::cout command
            start = i + 1;
        } else if (std::isspace(descFull[i]))
            space = i;

        if (i == start + 59 || i == len - 1) {
            if (space <= start || i == len - 1) // No space found, force the
                // wrap in the middle of the
                // (very long) word
                space = i;

            if (start > 0)
                std::cout << std::endl << std::string(20, ' ');

            std::cout << descFull.substr(start, space - start + 1);
            start = space + 1;
        }
    }

    std::cout << std::endl;
}

namespace N2D2 {
template <>
std::string ProgramOptions::grab<std::string>(const std::string& defaultValue,
                                              const std::string& name,
                                              const std::string& desc)
{
    std::string value = defaultValue;

    if (!mArgs.empty()) {
        value = mArgs[0];
        mArgs.erase(mArgs.begin());
    }

    mPosOptions.push_back(std::make_tuple(name, desc, value));

    return value;
}
}
