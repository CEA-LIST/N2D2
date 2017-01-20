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

#ifndef N2D2_COMPUTERVISION_LSL_H
#define N2D2_COMPUTERVISION_LSL_H

#include <tuple>
#include <vector>

#include "containers/Matrix.hpp"

namespace N2D2 {
namespace ComputerVision {
    /**
     * Implementation of the "Light Speed Labeling" algorithm.
     * See L. Lacassagne, B. Zavidovique, "Light speed labeling: efficient
     * connected component labeling on RISC architectures,"
     * Journal of Real-Time Image Processing, Volume 6, Issue 2, pp. 117-135,
     * 2011.
    */
    class LSL {
    public:
        void process(const Matrix<unsigned char>& frame);
        Matrix<unsigned int> getEA() const;

    private:
        // Processed image width (necessary for getEA())
        unsigned int mWidth;
        // Processed image height (necessary for getEA())
        unsigned int mHeight;
        // 2D list holding the run length coding of segments of every line
        std::vector<std::vector<unsigned int> > mRLC;
        // 2D list of absolute labels of every line
        std::vector<std::vector<unsigned int> > mLEA;
    };
}
}

#endif // N2D2_COMPUTERVISION_LSL_H
