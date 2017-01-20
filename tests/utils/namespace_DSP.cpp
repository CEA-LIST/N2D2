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

#include "utils/DSP.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(DSP, toComplex, (std::string xStr), std::make_tuple("1 2 3 4"))
{
    std::vector<double> x;
    x << xStr;

    const std::vector<std::complex<double> > y = DSP::toComplex(x);

    for (unsigned int i = 0; i < x.size(); ++i) {
        ASSERT_EQUALS(y[i].real(), x[i]);
        ASSERT_EQUALS(y[i].imag(), 0.0);
    }
}

TEST_DATASET(DSP,
             fft,
             (std::string xStr, std::string yStr),
             std::make_tuple(
                 "1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0",
                 "(4,0) (1,-2.414214) (0,0) (1,-0.4142136) (0,0) (1,0.4142136) "
                 "(0,0) (1,2.414214)"))
{
    std::vector<std::complex<double> > x;
    x << xStr;

    DSP::fft(x);

    std::vector<std::complex<double> > y;
    y << yStr;

    for (unsigned int i = 0; i < x.size(); ++i) {
        ASSERT_EQUALS_DELTA(x[i].real(), y[i].real(), 1.0e-6);
        ASSERT_EQUALS_DELTA(x[i].imag(), y[i].imag(), 1.0e-6);
    }
}

TEST_DATASET(DSP,
             ifft,
             (std::string xStr, std::string yStr),
             std::make_tuple(
                 "(4,0) (1,-2.414214) (0,0) (1,-0.4142136) (0,0) "
                 "(1,0.4142136) (0,0) (1,2.414214)",
                 "(1.0,0.0) (1.0,0.0) (1.0,0.0) (1.0,0.0) (0.0,0.0) (0.0,0.0) "
                 "(0.0,0.0) (0.0,0.0)"))
{
    std::vector<std::complex<double> > x;
    x << xStr;

    DSP::ifft(x);

    std::vector<std::complex<double> > y;
    y << yStr;

    for (unsigned int i = 0; i < x.size(); ++i) {
        ASSERT_EQUALS_DELTA(x[i].real(), y[i].real(), 1.0e-6);
        ASSERT_EQUALS_DELTA(x[i].imag(), y[i].imag(), 1.0e-6);
    }
}

TEST_DATASET(DSP,
             hilbert,
             (std::string xStr, std::string yStr),
             std::make_tuple("1 2 3 4", "(1,1) (2,-1) (3,-1) (4,1)"))
{
    std::vector<std::complex<double> > x;
    x << xStr;

    DSP::hilbert(x);

    std::vector<std::complex<double> > y;
    y << yStr;

    for (unsigned int i = 0; i < x.size(); ++i) {
        ASSERT_EQUALS_DELTA(x[i].real(), y[i].real(), 1.0e-6);
        ASSERT_EQUALS_DELTA(x[i].imag(), y[i].imag(), 1.0e-6);
    }
}

RUN_TESTS()
