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

#ifndef N2D2_TANHACTIVATION_H
#define N2D2_TANHACTIVATION_H

#include "Activation/Activation.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$TanhActivation_Frame@M@N2D2@@0U?$Registrar@V?$TanhActivation@M@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$TanhActivation_Frame_CUDA@M@N2D2@@0U?$Registrar@V?$TanhActivation@M@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
template <class T> class TanhActivation : public Activation<T> {
public:
    typedef std::function
        <std::shared_ptr<TanhActivation<T> >()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    /**
     * Alpha value for Y. LeCun recommanded sigmoid function, normalized between
     * -1 and 1.
     * See Y. LeCun, "Generalization and network design strategies", Connections
     * in Perspective, Amsterdam, 1989
     * \f$f(x) = tanh(1.7159 \frac{2}{3} x)\f$
    */
    static const double AlphaLeCun;

    TanhActivation();
    const char* getType() const
    {
        return Type;
    };
    virtual ~TanhActivation() {};

protected:
    /// Alpha
    Parameter<double> mAlpha;
};
}

template <class T> const char* N2D2::TanhActivation<T>::Type = "Tanh";

template <class T>
const double N2D2::TanhActivation<T>::AlphaLeCun = 1.7159 * (2.0 / 3.0);

template <class T>
N2D2::TanhActivation<T>::TanhActivation()
    : mAlpha(this, "Alpha", 1.0)
{
    // ctor
}

#endif // N2D2_TANHACTIVATION_H
