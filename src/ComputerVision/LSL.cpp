/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "ComputerVision/LSL.hpp"

void N2D2::ComputerVision::LSL::process(const Matrix<unsigned char>& frame)
{
    mWidth = frame.cols();
    mHeight = frame.rows();

    // ER an associative table holding the relative labels er associated
    std::vector<unsigned int> ER, ER1;
    // ERA an associative table holding the association between er and ea: ea =
    // ERA[er]
    std::vector<unsigned int> ERA, ERA1;
    // EQ the table holding the equivalence classes, before transitive closure
    std::vector<unsigned int> EQ(1, 0);

    ER.reserve(mWidth);
    ER1.resize(mWidth, 0);

    unsigned int nea = 0;

    mRLC.resize(mHeight);
    mLEA.resize(mHeight);

    for (unsigned int i = 0; i < mHeight; ++i) {
        mRLC[i].clear(); // mRLC[i] max size = mWidth + 1
        mLEA[i].clear();

        // Step #1: relative labeling of each line
        //////////////////////////////////////////

        unsigned int ner = 0; // initial label
        bool x1 = false; // previous value of x

        for (unsigned int j = 0; j < mWidth; ++j) {
            const bool x0 = (bool)frame(i, j);

            if (x0 != x1) {
                // Begin/end of segment
                // x0 = 1 => begin of segment, x1 = 0
                // x0 = 0 => end of segment, x1 = 1
                mRLC[i].push_back(j - x1); //  store
                ++ner; // segment label incrementation
            }

            ER.push_back(ner);
            x1 = x0;
        }

        if (x1) {
            // Close the segment
            mRLC[i].push_back(mWidth - 1);
        }

        // Step #2: equivalence construction
        ////////////////////////////////////

        for (unsigned int er = 1; er <= ner; er += 2) {
            // Boundaries [j0,j1] of the segment er
            unsigned int j0 = mRLC[i][er - 1];
            unsigned int j1 = mRLC[i][er];

            // check extension in case of 8-connect algorithm
            if (j0 > 0)
                --j0;
            if (j1 < mWidth - 1)
                ++j1;

            // relative labels of every adjacent segment in the previous line:
            // er0 is the label of the first segment and er1 the label of the
            // last segment
            int er0 = ER1[j0];
            int er1 = ER1[j1];

            // background slices are labeled with even numbers
            // check label parity: segments are odd
            if (er0 % 2 == 0)
                ++er0;
            if (er1 % 2 == 0)
                --er1; // er1 can be equal to -1

            if (er1 >= er0) {
                unsigned int ea = ERA1[(er0 - 1) / 2];
                unsigned int a = EQ[ea];

                for (unsigned int erk = er0 + 2; erk <= (unsigned int)er1;
                     erk += 2) {
                    const unsigned int eak = ERA1[(erk - 1) / 2];
                    const unsigned int ak = EQ[eak];

                    // min extraction and propagation
                    if (a < ak)
                        EQ[eak] = a;
                    else {
                        a = ak;
                        EQ[ea] = a;
                        ea = eak;
                    }
                }

                ERA.push_back(a); // the global min
                mLEA[i].push_back(a);
            } else {
                // no adjacency -> new label
                ++nea;
                EQ.push_back(nea);
                ERA.push_back(nea);
                mLEA[i].push_back(nea);
            }
        }
        /*
                // Debug
                std::cout << "--- line #" << i << "\n";

                for (unsigned int er = 0, size = ERA.size(); er < size; ++er)
                    std::cout << "    " << er << " -> " << ERA[er] << "\n";

                std::cout << std::endl;
        */
        // Step #3: segment first absolute labeling
        ///////////////////////////////////////////

        // No step #3 as we work on mLEA instead of EA

        /*
                // Debug
                for (unsigned int j = 0; j < mWidth; ++j)
                    std::cout << frame(i,j) << " ";

                std::cout << "\n";

                for (unsigned int j = 0; j < mWidth; ++j)
                    std::cout << ER[j] << " ";

                std::cout << "\n";
                std::cout << std::endl;
                // End Debug
        */
        ER1.swap(ER);
        ER.clear();

        ERA1.swap(ERA);
        ERA.clear();
    }

    // Step #4: resolution of the equivalence classes
    /////////////////////////////////////////////////

    unsigned int na = 0;

    for (unsigned int e = 1, ne = EQ.size(); e < ne; ++e) {
        if (EQ[e] != e)
            EQ[e] = EQ[EQ[e]];
        else {
            ++na;
            EQ[e] = na;
        }
    }

// Step #5: second absolute labeling
////////////////////////////////////

#pragma omp parallel for if (mHeight > 32)
    for (int i = 0; i < (int)mHeight; ++i) {
        for (unsigned int k = 0, size = mLEA[i].size(); k < size; ++k)
            mLEA[i][k] = EQ[mLEA[i][k]];
    }
}

N2D2::Matrix<unsigned int> N2D2::ComputerVision::LSL::getEA() const
{
    assert(mHeight == mRLC.size());
    assert(mHeight == mLEA.size());

    Matrix<unsigned int> EA(mHeight, mWidth, 0);

#pragma omp parallel for if (mHeight > 32)
    for (int i = 0; i < (int)mHeight; ++i) {
        for (unsigned int er = 1, ner = mRLC[i].size(); er <= ner; er += 2) {
            // Boundaries [j0,j1] of the segment er
            const unsigned int j0 = mRLC[i][er - 1];
            const unsigned int j1 = mRLC[i][er];

            for (unsigned int j = j0; j <= j1; ++j)
                EA(i, j) = mLEA[i][(er - 1) / 2];
        }
    }

    return EA;
}
