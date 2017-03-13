/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Victor GACOIN
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_REGISTRAR_H
#define N2D2_REGISTRAR_H

#include <cstdarg>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace N2D2 {
class BaseCommand {
public:
    virtual ~BaseCommand()
    {
    }
};

typedef std::map<std::string, BaseCommand*> RegistryMap_T;

template <typename C> struct Registrar {
    struct Command : public BaseCommand {
        typename C::RegistryCreate_T mFunc;
        Command(typename C::RegistryCreate_T func) : mFunc(func)
        {
        }
    };

    Registrar(const std::string& key, typename C::RegistryCreate_T func)
    {
        if (C::registry().find(key) != C::registry().end())
            throw std::runtime_error("Registrar \"" + key
                                     + "\" already exists");

        C::registry().insert(std::make_pair(key, new Command(func)));
    }

    Registrar(typename C::RegistryCreate_T func, const char* key, ...)
    {
        va_list args;
        va_start(args, key);

        while (key != NULL) {
            if (C::registry().find(key) != C::registry().end())
                throw std::runtime_error("Registrar \"" + std::string(key)
                                         + "\" already exists");

            C::registry().insert(std::make_pair(key, new Command(func)));

            key = va_arg(args, const char*);
        }

        va_end(args);
    }

    static bool exists(const std::string& key)
    {
        return (C::registry().find(key) != C::registry().end());
    }

    static typename C::RegistryCreate_T create(const std::string& key)
    {
        const RegistryMap_T::const_iterator it = C::registry().find(key);

        if (it == C::registry().end()) {
            // throw std::runtime_error("Invalid registrar key \"" + key +
            // "\"");
            std::cout << "Invalid registrar key \"" << key << "\"" << std::endl;
#ifdef WIN32
            return nullptr; // Required by Visual C++
#else
            return NULL; // but nullptr is not supported on GCC 4.4
#endif
        }

        return static_cast<Command*>(it->second)->mFunc;
    }
};

template <typename C, typename F = typename C::RegistryCreate_T>
struct RegistrarCustom {
    struct Command : public BaseCommand {
        F mFunc;
        Command(F func) : mFunc(func)
        {
        }
    };

    RegistrarCustom(const std::string& key, F func)
    {
        if (C::registry().find(key) != C::registry().end())
            throw std::runtime_error("Registrar \"" + key
                                     + "\" already exists");

        C::registry().insert(std::make_pair(key, new Command(func)));
    }

    RegistrarCustom(F func, const char* key, ...)
    {
        va_list args;
        va_start(args, key);

        while (key != NULL) {
            if (C::registry().find(key) != C::registry().end())
                throw std::runtime_error("Registrar \"" + std::string(key)
                                         + "\" already exists");

            C::registry().insert(std::make_pair(key, new Command(func)));

            key = va_arg(args, const char*);
        }

        va_end(args);
    }

    static bool exists(const std::string& key)
    {
        return (C::registry().find(key) != C::registry().end());
    }

    static F create(const std::string& key)
    {
        const RegistryMap_T::const_iterator it = C::registry().find(key);

        if (it == C::registry().end()) {
            // throw std::runtime_error("Invalid registrar key \"" + key +
            // "\"");
            std::cout << "Invalid registrar key \"" << key << "\"" << std::endl;
#ifdef WIN32
            return nullptr; // Required by Visual C++
#else
            return NULL; // but nullptr is not supported on GCC 4.4
#endif
        }

        return static_cast<Command*>(it->second)->mFunc;
    }
};
}

#endif // N2D2_REGISTRAR_H
