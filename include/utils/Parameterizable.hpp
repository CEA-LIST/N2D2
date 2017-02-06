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

#ifndef N2D2_PARAMETERIZABLE_H
#define N2D2_PARAMETERIZABLE_H

#include <cctype>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class Parameter_T {
public:
    template <class T> explicit Parameter_T(T& value);
    template <class T> Parameter_T& operator=(const T& value);
    Parameter_T& operator=(const Parameter_T& value);
    template <class T> T get() const;
    template <class T> void copy(const Parameter_T& value);
    template <class T> std::ostream& print(std::ostream& os) const;
    template <class T> std::istream& read(std::istream& is) const;
    virtual ~Parameter_T() {};

    friend std::ostream& operator<<(std::ostream&, const Parameter_T&);
    friend std::istream& operator>>(std::istream&, const Parameter_T&);

private:
    typedef void (Parameter_T::*Copy_PT)(const Parameter_T&);
    typedef std::ostream& (Parameter_T::*Print_PT)(std::ostream&) const;
    typedef std::istream& (Parameter_T::*Read_PT)(std::istream&) const;

    void* mValue;
    const std::type_info* mType;
    Copy_PT mCopy;
    Print_PT mPrint;
    Read_PT mRead;
};

class Percent {
public:
    explicit Percent(double value) : mValue(value) {};
    operator double() const
    {
        return mValue;
    }

private:
    double mValue;
};

template <class T> class Spread {
public:
    Spread(T mean, double stdDev)
        : mValue(mean),
          mValueMean(mean),
          mValueSpread(stdDev),
          mRelativeSpread(false) {};

    Spread(T mean, Percent relStdDev = Percent(0))
        : mValue(mean),
          mValueMean(mean),
          mValueSpread(relStdDev),
          mRelativeSpread(true) {};

    Spread& operator=(const T& value)
    {
        mValue = value;
        mValueMean = value;

        if (!mRelativeSpread)
            mValueSpread = 0.0;

        return *this;
    }

    T& spreadNormal()
    {
        return (mValue = Random::randNormal(mValueMean, stdDev()));
    }

    T& spreadNormal(T vmin)
    {
        return (mValue = Random::randNormal(mValueMean, stdDev(), vmin));
    }

    T& spreadNormal(T vmin, T vmax)
    {
        return (mValue = Random::randNormal(mValueMean, stdDev(), vmin, vmax));
    }

    T& spreadLogNormal(bool logMean = false)
    {
        return (mValue = Random::randLogNormal(
                    (logMean) ? std::log(mValueMean) : mValueMean, stdDev()));
    }

    T spreadNormal() const
    {
        return Random::randNormal(mValueMean, stdDev());
    }

    T spreadNormal(T vmin) const
    {
        return Random::randNormal(mValueMean, stdDev(), vmin);
    }

    T spreadNormal(T vmin, T vmax) const
    {
        return Random::randNormal(mValueMean, stdDev(), vmin, vmax);
    }

    T spreadLogNormal(bool logMean = false) const
    {
        return Random::randLogNormal(
            (logMean) ? std::log(mValueMean) : mValueMean, stdDev());
    }

    void setSpread(double stdDev)
    {
        mValueSpread = stdDev;
        mRelativeSpread = false;
    }

    void setSpread(Percent relStdDev = Percent(0))
    {
        mValueSpread = relStdDev;
        mRelativeSpread = true;
    }

    operator T() const
    {
        return mValue;
    }
    T mean() const
    {
        return mValueMean;
    }

    double stdDev() const
    {
        return (mRelativeSpread) ? std::fabs(static_cast<double>(mValueMean))
                                   * mValueSpread / 100.0
                                 : mValueSpread;
    }

    template <class U>
    friend std::ostream& operator<<(std::ostream& os, const Spread<U>& spread);
    template <class U>
    friend std::istream& operator>>(std::istream& is, Spread<U>& spread);

private:
    T mValue;
    T mValueMean;
    double mValueSpread;
    bool mRelativeSpread;
};

template <class T> class Parameter;

template <class T> class ParameterWithSpread;

/**
* This class handles a map with string keys and parameters (Parameter_T*) as
* values.
*/
class Parameterizable {
public:
    bool isParameter(const std::string& name) const;
    template <class T> void setParameter(const std::string& name, T value);
    template <class T>
    void setParameter(const std::string& name, T mean, Percent relStdDev);
    template <class T>
    void setParameter(const std::string& name, T mean, double stdDev);
    template <class T>
    void setParameterSpread(const std::string& name,
                            Percent relStdDev = Percent(0));
    template <class T>
    void setParameterSpread(const std::string& name, double stdDev);
    void setParameter(const std::string& name, const std::string& value);
    unsigned int setParameters(const std::map<std::string, std::string>& params,
                               bool ignoreUnknown = false);
    unsigned int setPrefixedParameters(const std::map
                                       <std::string, std::string>& params,
                                       const std::string& prefix,
                                       bool ignoreUnknown = false);
    unsigned int setPrefixedParameters(std::map
                                       <std::string, std::string>& params,
                                       const std::string& prefix,
                                       bool greedy = true,
                                       bool ignoreUnknown = false);
    template <class T> T getParameter(const std::string& name) const;
    std::string getParameter(const std::string& name) const;

    /**
     * Load parameters from file @p fileName.
     * This function supports single-line comments in the configuration file
     *(comments start with #).
     *
     * @param fileName                  Configuration file name to load
     *parameters from
     * @param ignoreNotExists           If true, no error is thrown if the file
     *does not exist
     * @param ignoreUnknown             If true, no error is thrown if there is
     *an unknown parameter in the config file
     * @return Number of valid parameters loaded
     *
     * @exception std::runtime_error Could not open configuration file (only if
     *@p ignoreNotExists is false)
     * @exception std::runtime_error Unknown parameter in config file (only if
     *@p ignoreUnknown is false)
     * @exception std::runtime_error Unreadable parameter in config file
     * @exception std::runtime_error Missing value for parameter in config file
    */
    unsigned int loadParameters(const std::string& fileName,
                                bool ignoreNotExists = false,
                                bool ignoreUnknown = false);
    void saveParameters(const std::string& fileName) const;
    void copyParameters(const Parameterizable& from);
    virtual ~Parameterizable() {};

    template <class T> friend class Parameter;
    template <class T> friend class ParameterWithSpread;

private:
    std::map<std::string, Parameter_T*> mParameters;
};

template <class T> class Parameter : public Parameter_T {
public:
    // No default constructor is created, this ensures that the Parameter has to
    // be attached to a Parameterizable object.
    // Additionally, it forces at compile time the Parameter to be initialized
    // in the constructor initializer list, making it
    // impossible to forget its initialization.
    Parameter(Parameterizable* p, const std::string& name, T value)
        : Parameter_T(mValue), mValue(value)
    {
        if ((*p).mParameters.find(name) != (*p).mParameters.end())
            throw std::runtime_error("Parameter already exists: " + name);

        (*p).mParameters[name] = this;
    }

    Parameter& operator=(const T& value)
    {
        mValue = value;
        return *this;
    }

    // Compound assignment operators
    T& operator+=(const T& value)
    {
        return (mValue += value);
    }
    T& operator-=(const T& value)
    {
        return (mValue -= value);
    }
    T& operator*=(const T& value)
    {
        return (mValue *= value);
    }
    T& operator/=(const T& value)
    {
        return (mValue /= value);
    }

    // Arithmetic operators
    T& operator++()
    {
        return (++mValue);
    }
    T& operator++(int)
    {
        return (mValue++);
    }
    T& operator--()
    {
        return (--mValue);
    }
    T& operator--(int)
    {
        return (mValue--);
    }
    operator T() const
    {
        return mValue;
    }
    T* operator->() {
        return &mValue;
    }
    T const* operator->() const {
        return &mValue;
    }

private:
    T mValue;
};

template <class T> class ParameterWithSpread : public Parameter_T {
public:
    // No default constructor is created, this ensures that the Parameter has to
    // be attached to a Parameterizable object.
    // Additionally, it forces at compile time the Parameter to be initialized
    // in the constructor initializer list, making it
    // impossible to forget its initialization.
    ParameterWithSpread(Parameterizable* p,
                        const std::string& name,
                        const Spread<T>& value)
        : Parameter_T(mValue), mValue(value)
    {
        if ((*p).mParameters.find(name) != (*p).mParameters.end())
            throw std::runtime_error("Parameter already exists: " + name);

        (*p).mParameters[name] = this;
    }

    ParameterWithSpread(Parameterizable* p,
                        const std::string& name,
                        T mean,
                        Percent relStdDev = Percent(0))
        : Parameter_T(mValue), mValue(Spread<T>(mean, relStdDev))
    {
        if ((*p).mParameters.find(name) != (*p).mParameters.end())
            throw std::runtime_error("Parameter already exists: " + name);

        (*p).mParameters[name] = this;
    }

    ParameterWithSpread(Parameterizable* p,
                        const std::string& name,
                        T mean,
                        double stdDev)
        : Parameter_T(mValue), mValue(Spread<T>(mean, stdDev))
    {
        if ((*p).mParameters.find(name) != (*p).mParameters.end())
            throw std::runtime_error("Parameter already exists: " + name);

        (*p).mParameters[name] = this;
    }

    ParameterWithSpread& operator=(const Spread<T>& value)
    {
        mValue = value;
        return *this;
    }

    ParameterWithSpread& operator=(const T& value)
    {
        mValue = value;
        return *this;
    }

    void setSpread(double stdDev)
    {
        mValue.setSpread(stdDev);
    }

    void setSpread(Percent relStdDev = Percent(0))
    {
        mValue.setSpread(relStdDev);
    }

    T& spreadNormal()
    {
        return (mValue.spreadNormal());
    }

    T& spreadNormal(T vmin)
    {
        return (mValue.spreadNormal(vmin));
    }

    T& spreadNormal(T vmin, T vmax)
    {
        return (mValue.spreadNormal(vmin, vmax));
    }

    T& spreadLogNormal(bool logMean = false)
    {
        return (mValue.spreadLogNormal(logMean));
    }

    T spreadNormal() const
    {
        return (mValue.spreadNormal());
    }

    T spreadNormal(T vmin) const
    {
        return (mValue.spreadNormal(vmin));
    }

    T spreadNormal(T vmin, T vmax) const
    {
        return (mValue.spreadNormal(vmin, vmax));
    }

    T spreadLogNormal(bool logMean = false) const
    {
        return (mValue.spreadLogNormal(logMean));
    }

    operator T() const
    {
        return static_cast<T>(mValue);
    }

    operator Spread<T>() const
    {
        return mValue;
    }

    T mean() const
    {
        return mValue.mean();
    }

    double stdDev() const
    {
        return mValue.stdDev();
    }

private:
    Spread<T> mValue;
};
}

template <class T>
N2D2::Parameter_T::Parameter_T(T& value)
    : mValue(&value),
      mType(&typeid(T)),
      mCopy(&Parameter_T::copy<T>),
      mPrint(&Parameter_T::print<T>),
      mRead(&Parameter_T::read<T>)
{
    // ctor
}

template <class T>
N2D2::Parameter_T& N2D2::Parameter_T::operator=(const T& value)
{
    if (*mType == typeid(T))
        *((T*)mValue) = value;
    else if (*mType == typeid(Spread<T>))
        *((Spread<T>*)mValue) = value;
    else {
        throw std::runtime_error("Incompatible type [operator =(): "
                                 + std::string((*mType).name()) + " != "
                                 + std::string(typeid(T).name()) + " or "
                                 + std::string(typeid(Spread<T>).name()) + "]");
    }

    return *this;
}

template <class T> T N2D2::Parameter_T::get() const
{
    if (mType == NULL || *mType != typeid(T)) {
        throw std::runtime_error("Incompatible type [get(): "
                                 + std::string((*mType).name()) + " != "
                                 + std::string(typeid(T).name()) + "]");
    }

    return *((T*)mValue);
}

template <class T> void N2D2::Parameter_T::copy(const Parameter_T& value)
{
    if (*mType == *value.mType)
        *((T*)mValue) = *((T*)value.mValue);
    else {
        throw std::runtime_error("Incompatible type [copy(): "
                                 + std::string((*mType).name()) + " != "
                                 + std::string(typeid(T).name()) + "]");
    }
}

template <class T>
std::ostream& N2D2::Parameter_T::print(std::ostream& os) const
{
    return os << *((T*)mValue);
}

template <class T> std::istream& N2D2::Parameter_T::read(std::istream& is) const
{
    return Utils::signChecked<T>(is) >> *((T*)mValue);
}

template <class T>
void N2D2::Parameterizable::setParameter(const std::string& name, T value)
{
    if (mParameters.find(name) != mParameters.end())
        (*mParameters[name]) = value;
    else
        throw std::runtime_error("Parameter does not exist: " + name);
}

namespace N2D2 {
template <class T>
std::ostream& operator<<(std::ostream& os, const Spread<T>& spread)
{
    os << spread.mValue << "; " << spread.mValueMean << "; "
       << spread.mValueSpread;

    if (spread.mRelativeSpread)
        os << "%";

    return os;
}
using ::operator<<;
}

namespace N2D2 {
template <class T> std::istream& operator>>(std::istream& is, Spread<T>& spread)
{
    if (!(is >> spread.mValue))
        throw std::runtime_error("Unreadable Spread stream data");

    if (is.eof()) {
        spread.mValueMean = spread.mValue;
        spread.mValueSpread = 0.0;
        spread.mRelativeSpread = true;
    } else {
        std::string dummy;
        std::getline(is >> std::ws, dummy, ';');

        if (!dummy.empty())
            throw std::runtime_error(
                "Unreadable Spread stream data (mean value)");

        const std::streampos isPos = is.tellg();
        std::stringstream meanSpread;
        meanSpread << is.rdbuf();
        is.seekg(isPos);

        if (meanSpread.str().find(';') != std::string::npos) {
            // Mean value that may be different from value
            if (!(is >> spread.mValueMean))
                throw std::runtime_error(
                    "Unreadable Spread stream data (mean value)");

            std::getline(is >> std::ws, dummy, ';');

            if (!dummy.empty())
                throw std::runtime_error(
                    "Unreadable Spread stream data (mean value)");
        } else
            spread.mValueMean = spread.mValue;

        if (!(is >> spread.mValueSpread))
            throw std::runtime_error(
                "Unreadable Spread stream data (spread value)");

        if (!is.eof() && is.peek() == '%') {
            is.get(); // Discard '%'
            is.peek(); // Set EOF flag if end of stream
            spread.mRelativeSpread = true;
        } else
            spread.mRelativeSpread = false;
    }

    return is;
}
using ::operator>>;
}

template <class T>
void N2D2::Parameterizable::setParameter(const std::string& name,
                                         T mean,
                                         Percent relStdDev)
{
    if (mParameters.find(name) != mParameters.end()) {
        ParameterWithSpread<T>* spreadParam = dynamic_cast
            <ParameterWithSpread<T>*>(mParameters[name]);

        if (spreadParam != NULL)
            (*spreadParam) = Spread<T>(mean, relStdDev);
        else
            throw std::runtime_error("Parameter " + name
                                     + " is not a parameter with spread");
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

template <class T>
void N2D2::Parameterizable::setParameter(const std::string& name,
                                         T mean,
                                         double stdDev)
{
    if (mParameters.find(name) != mParameters.end()) {
        ParameterWithSpread<T>* spreadParam = dynamic_cast
            <ParameterWithSpread<T>*>(mParameters[name]);

        if (spreadParam != NULL)
            (*spreadParam) = Spread<T>(mean, stdDev);
        else
            throw std::runtime_error("Parameter " + name
                                     + " is not a parameter with spread");
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

template <class T>
void N2D2::Parameterizable::setParameterSpread(const std::string& name,
                                               Percent relStdDev)
{
    if (mParameters.find(name) != mParameters.end()) {
        ParameterWithSpread<T>* spreadParam = dynamic_cast
            <ParameterWithSpread<T>*>(mParameters[name]);

        if (spreadParam != NULL)
            (*spreadParam).setSpread(relStdDev);
        else
            throw std::runtime_error("Parameter " + name
                                     + " is not a parameter with spread");
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

template <class T>
void N2D2::Parameterizable::setParameterSpread(const std::string& name,
                                               double stdDev)
{
    if (mParameters.find(name) != mParameters.end()) {
        ParameterWithSpread<T>* spreadParam = dynamic_cast
            <ParameterWithSpread<T>*>(mParameters[name]);

        if (spreadParam != NULL)
            (*spreadParam).setSpread(stdDev);
        else
            throw std::runtime_error("Parameter " + name
                                     + " is not a parameter with spread");
    } else
        throw std::runtime_error("Parameter does not exist: " + name);
}

template <class T>
T N2D2::Parameterizable::getParameter(const std::string& name) const
{
    const std::map<std::string, Parameter_T*>::const_iterator it
        = mParameters.find(name);

    if (it != mParameters.end())
        return (*it).second->get<T>();
    else
        throw std::runtime_error("Parameter does not exist: " + name);
}

#endif // N2D2_PARAMETERIZABLE_H
