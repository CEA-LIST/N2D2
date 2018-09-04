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

typedef std::map<std::string,
                 std::map<const std::type_info*, BaseCommand*> > RegistryMap_T;

template <typename C, typename F = typename C::RegistryCreate_T>
struct Registrar {
    struct Command : public BaseCommand {
        F mFunc;
        Command(F func) : mFunc(func)
        {
        }
    };

    template <class T>
    struct Type {};

    template <class T = void>
    Registrar(const std::string& key,
              F func,
              const Type<T>& /*type*/ = Type<T>())
    {
        RegistryMap_T::iterator it;
        bool newInsert;
        std::tie(it, std::ignore)
            = C::registry().insert(std::make_pair(key,
                        std::map<const std::type_info*, BaseCommand*>()));
        std::tie(std::ignore, newInsert)
            = (*it).second.insert(std::make_pair(&typeid(T),
                                                 new Command(func)));

        if (!newInsert) {
            throw std::runtime_error("Registrar \"" + key
                                     + "\" already exists");
        }
    }

    template <class T = void>
    Registrar(std::initializer_list<std::string> keys,
              F func,
              const Type<T>& /*type*/ = Type<T>())
    {
        for (auto keyIt = keys.begin(), keyItEnd = keys.end();
            keyIt != keyItEnd; ++keyIt)
        {
            RegistryMap_T::iterator it;
            bool newInsert;
            std::tie(it, std::ignore)
                = C::registry().insert(std::make_pair(*keyIt,
                            std::map<const std::type_info*, BaseCommand*>()));
            std::tie(std::ignore, newInsert)
                = (*it).second.insert(std::make_pair(&typeid(T),
                                                     new Command(func)));

            if (!newInsert) {
                throw std::runtime_error("Registrar \"" + (*keyIt)
                                         + "\" already exists");
            }
        }
    }

    static bool exists(const std::string& key)
    {
        return (C::registry().find(key) != C::registry().end());
    }

    template <class T = void>
    static bool exists(const std::string& key)
    {
        const RegistryMap_T::const_iterator it = C::registry().find(key);

        if (it == C::registry().end())
            return false;

        return (it->second.find(&typeid(T)) != it->second.end());
    }

    template <class T = void>
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

        const std::map<const std::type_info*, BaseCommand*>::const_iterator
            itType = it->second.find(&typeid(T));

        if (itType == it->second.end()) {
            std::cout << "Invalid registrar key type (" << typeid(T).name()
                << ") for key \"" << key << "\"" << std::endl;
#ifdef WIN32
            return nullptr; // Required by Visual C++
#else
            return NULL; // but nullptr is not supported on GCC 4.4
#endif
        }

        return static_cast<Command*>(itType->second)->mFunc;
    }
};
}

#endif // N2D2_REGISTRAR_H
