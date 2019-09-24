/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <algorithm>
#include <vector>
#include "Histogram.hpp"
#include "utils/UnitTest.hpp"


using namespace N2D2;


TEST(Histogram, test_out_of_range) {
    const std::size_t nbBins = 3;
    Histogram hist(-10, 10, nbBins);
    hist(0);
    hist(-4);
    hist(3);
    hist(10);
    hist(-10);

    ASSERT_THROW(hist(-11), std::out_of_range);
    ASSERT_THROW(hist(11), std::out_of_range);
    ASSERT_THROW(hist(450), std::out_of_range);
}

TEST(Histogram, test_out_of_range_positive) {
    const std::size_t nbBins = 3;
    Histogram hist(0, 10, nbBins);
    hist(0);
    hist(3);
    hist(10);

    ASSERT_THROW(hist(-1), std::out_of_range);
    ASSERT_THROW(hist(-11), std::out_of_range);
    ASSERT_THROW(hist(11), std::out_of_range);
    ASSERT_THROW(hist(450), std::out_of_range);
}

TEST(Histogram, test_get) {
    const std::vector<double> values = {3.4, 19, -10, 20, 41, -23.4, -23.39, 
                                        7.2, -8.2, 3.4, 20.1, 22.3, 0, 1};
    const auto minMax = std::minmax_element(values.begin(), values.end());
    const std::size_t nbBins = 11;

    Histogram hist(*minMax.first, *minMax.second, nbBins);
    for(auto v: values) {
        hist(v);
    }
    hist(1, 20);

    ASSERT_TRUE(hist.getBins() == 
                std::vector<std::size_t>({2, 0, 2, 1, 23, 1, 0, 4, 0, 0, 1}));


    ASSERT_EQUALS(hist.getMinVal(), -23.4);
    ASSERT_EQUALS(hist.getMaxVal(), 41);

    ASSERT_EQUALS(hist.getNbBins(), nbBins);
    ASSERT_EQUALS_DELTA(hist.getBinWidth(), (41.0+23.4)/nbBins, 0.0001);

    ASSERT_EQUALS_DELTA(hist.getBinValue(0), -23.4 + hist.getBinWidth()*0.5, 0.0001);
    ASSERT_EQUALS_DELTA(hist.getBinValue(1), -23.4 + hist.getBinWidth()*1.5, 0.0001);
    ASSERT_EQUALS_DELTA(hist.getBinValue(10), 41 - hist.getBinWidth()*0.5, 0.0001);
    ASSERT_THROW(hist.getBinValue(11), std::out_of_range);
    ASSERT_THROW(hist.getBinValue(100), std::out_of_range);


    ASSERT_EQUALS(hist.getBinIdx(-23.4), 0);
    ASSERT_EQUALS(hist.getBinIdx(-100), 0);
    ASSERT_EQUALS(hist.getBinIdx(41), 10);
    ASSERT_EQUALS(hist.getBinIdx(100), 10);

    ASSERT_EQUALS(hist.getBinIdx(-23.41 + hist.getBinWidth()), 0);
    ASSERT_EQUALS(hist.getBinIdx(-23.4 + hist.getBinWidth()), 1);
    ASSERT_EQUALS(hist.getBinIdx(-23.39 + hist.getBinWidth()), 1);

    ASSERT_EQUALS(hist.getBinIdx(-23.4 + hist.getBinWidth()*2), 2);
    ASSERT_EQUALS(hist.getBinIdx(-23.4 + hist.getBinWidth()*5), 5);
}

TEST(Histogram, test_enlarge_symetric) {
    const std::vector<double> values = {3.4, 19, -10, 20, 41, -23.4, -23.39, 
                                        7.2, -8.2, 3.4, 20.1, 22.3, 0, 1};
    const auto minMax = std::minmax_element(values.begin(), values.end());
    const std::size_t nbBins = 11;

    Histogram hist(*minMax.first, *minMax.second, nbBins);
    for(auto v: values) {
        hist(v);
    }

    const double oldBinWidth = hist.getBinWidth();
    ASSERT_TRUE(hist.getBins() == 
                std::vector<std::size_t>({2, 0, 2, 1, 3, 1, 0, 4, 0, 0, 1}));

    // Enlarge, should at 2 bins at the begining and 2 bins at the end
    hist.enlarge(-30, true);

    ASSERT_EQUALS_DELTA(oldBinWidth, hist.getBinWidth(), 1e-6);
    ASSERT_EQUALS(nbBins + 2*2, hist.getNbBins());
    
    ASSERT_EQUALS_DELTA(hist.getMinVal(), *minMax.first - 2*oldBinWidth, 1e-6);
    ASSERT_EQUALS_DELTA(hist.getMaxVal(), *minMax.second + 2*oldBinWidth, 1e-6);

    ASSERT_TRUE(hist.getBins() == 
                std::vector<std::size_t>({0, 0, 2, 0, 2, 1, 3, 1, 0, 4, 0, 0, 1, 0, 0}));
}

TEST(Histogram, test_enlarge_asymetric_min) {
    const std::vector<double> values = {3.4, 19, -10, 20, 41, -23.4, -23.39, 
                                        7.2, -8.2, 3.4, 20.1, 22.3, 0, 1};
    const auto minMax = std::minmax_element(values.begin(), values.end());
    const std::size_t nbBins = 11;

    Histogram hist(*minMax.first, *minMax.second, nbBins);
    for(auto v: values) {
        hist(v);
    }

    const double oldBinWidth = hist.getBinWidth();
    // Enlarge, should at 2 bins at the begining 
    hist.enlarge(-30, false);

    ASSERT_EQUALS_DELTA(oldBinWidth, hist.getBinWidth(), 1e-6);
    ASSERT_EQUALS(nbBins + 2, hist.getNbBins());
    
    ASSERT_EQUALS_DELTA(hist.getMinVal(), *minMax.first - 2*oldBinWidth, 1e-6);
    ASSERT_EQUALS_DELTA(hist.getMaxVal(), *minMax.second, 1e-6);

    ASSERT_TRUE(hist.getBins() == 
                std::vector<std::size_t>({0, 0, 2, 0, 2, 1, 3, 1, 0, 4, 0, 0, 1}));
}

TEST(Histogram, test_enlarge_asymetric_max) {
    const std::vector<double> values = {3.4, 19, -10, 20, 41, -23.4, -23.39, 
                                        7.2, -8.2, 3.4, 20.1, 22.3, 0, 1};
    const auto minMax = std::minmax_element(values.begin(), values.end());
    const std::size_t nbBins = 11;

    Histogram hist(*minMax.first, *minMax.second, nbBins);
    for(auto v: values) {
        hist(v);
    }

    const double oldBinWidth = hist.getBinWidth();
    // Enlarge, should at 2 bins at the end 
    hist.enlarge(50, false);

    ASSERT_EQUALS_DELTA(oldBinWidth, hist.getBinWidth(), 1e-6);
    ASSERT_EQUALS(nbBins + 2, hist.getNbBins());
    
    ASSERT_EQUALS_DELTA(hist.getMinVal(), *minMax.first, 1e-6);
    ASSERT_EQUALS_DELTA(hist.getMaxVal(), *minMax.second + 2*oldBinWidth, 1e-6);

    ASSERT_TRUE(hist.getBins() == 
                std::vector<std::size_t>({2, 0, 2, 1, 3, 1, 0, 4, 0, 0, 1, 0, 0}));
}

RUN_TESTS()