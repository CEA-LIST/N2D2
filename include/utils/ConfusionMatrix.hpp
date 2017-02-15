/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CONFUSIONMATRIX_H
#define N2D2_CONFUSIONMATRIX_H

#include <type_traits>

#include "containers/Matrix.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
template <class T>
class ConfusionMatrix : public Matrix<T> {
public:
    ConfusionMatrix() : Matrix<T>()
    {
    }
    ConfusionMatrix(unsigned int nbRows) : Matrix<T>(nbRows)
    {
    }
    ConfusionMatrix(unsigned int nbRows,
                    unsigned int nbCols,
                    const T& value = T())
        : Matrix<T>(nbRows, nbCols, value)
    {
    }
    void log(const std::string& fileName,
             const std::vector<std::string>& labels = std::vector
             <std::string>()) const;
    virtual ~ConfusionMatrix()
    {
    }
};

template <class T>
class ConfusionTable {
public:
    ConfusionTable() : mTp(0), mTn(0), mFp(0), mFn(0) {};
    double precision() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFp)) : 0.0;
    };
    double recall() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFn)) : 0.0;
    };
    double F1Score() const
    {
        return (mTp > 0) ? (2 * mTp / (double)(2 * mTp + mFp + mFn)) : 0.0;
    };
    void tp(T tp)
    {
        mTp += tp;
    };
    void tn(T tn)
    {
        mTn += tn;
    };
    void fp(T fp)
    {
        mFp += fp;
    };
    void fn(T fn)
    {
        mFn += fn;
    };
    T tp() const
    {
        return mTp;
    };
    T tn() const
    {
        return mTn;
    };
    T fp() const
    {
        return mFp;
    };
    T fn() const
    {
        return mFn;
    };

private:
    T mTp;
    T mTn;
    T mFp;
    T mFn;
};
}

template <class T>
void N2D2::ConfusionMatrix<T>::log(const std::string& fileName,
                                const std::vector<std::string>& labels) const
{
    std::ofstream confData(fileName);

    if (!confData.good())
        throw std::runtime_error("Could not save confusion matrix data file: "
                                 + fileName);

    confData << "target estimated # %\n";

    std::stringstream tics;
    tics << "(";

    T total = 0;
    T totalCorrect = 0;

    const unsigned int nbTargets = this->rows();
    std::vector<ConfusionTable<T> > conf(nbTargets, ConfusionTable<T>());

    for (unsigned int target = 0; target < nbTargets; ++target) {
        const std::vector<T>& row = this->row(target);
        const T targetCount = std::accumulate(row.begin(), row.end(), (T)0);

        total += targetCount;
        totalCorrect += (*this)(target, target);

        for (unsigned int estimated = 0; estimated < nbTargets; ++estimated) {
            if (target == estimated) {
                conf[target].tp((*this)(target, estimated));
            } else {
                conf[target].fn((*this)(target, estimated));
                conf[target].fp((*this)(estimated, target));

                for (unsigned int other = 0; other < nbTargets; ++other) {
                    if (other != target)
                        conf[target].tn((*this)(other, estimated));
                }
            }

            confData << target << " " << estimated << " "
                     << (*this)(target, estimated) << " "
                     << ((targetCount > 0) ? ((*this)(target, estimated)
                                              / (double)targetCount)
                                           : 0.0) << "\n";
        }

        if (target > 0)
            tics << ", ";

        tics << "\"";

        if (!labels.empty())
            tics << labels[target];
        else
            tics << target;

        tics << "\" " << target;
    }

    tics << ")";
    confData.close();

    const std::string confFile = Utils::fileBaseName(fileName) + "_score."
                                 + Utils::fileExtension(fileName);
    confData.open(confFile);

    if (!confData.good())
        throw std::runtime_error("Could not save confusion data file: "
                                 + confFile);

    confData << "target precision recall F1-score\n";

    for (unsigned int target = 0; target < nbTargets; ++target) {
        confData << target << " " << conf[target].precision() << " "
                 << conf[target].recall() << " " << conf[target].F1Score()
                 << "\n";
    }

    confData.close();

    Gnuplot::setDefaultOutput("png", "size 800,600 tiny", "png");

    Gnuplot gnuplot;
    gnuplot.set("key off").unset("colorbox");

    typedef typename std::make_signed<T>::type T_Signed;

    std::stringstream xlabel;
    xlabel << "Estimated class ("
              "total correct: " << totalCorrect << ", "
              "total misclassified: "
            << ((T_Signed)total - (T_Signed)totalCorrect)
           << ", error rate: " << std::fixed << std::setprecision(2)
           << ((total > 0.0)
               ? (100.0 * (1.0 - (T_Signed)totalCorrect / (double)total))
               : 0.0) << "%)";

    gnuplot.setXlabel(xlabel.str());
    gnuplot.setYlabel("Target (actual) class");
    gnuplot.set("xtics rotate by 90", tics.str());
    gnuplot.set("ytics", tics.str());
    gnuplot.set(
        "palette",
        "defined (-2 'red', -1.01 '#FFEEEE', -1 'white', 0 'red', 1 'cyan')");
    gnuplot.set("yrange", "[] reverse");
    gnuplot.set("cbrange", "[-2:1]");

    std::stringstream plotCmd;
    //plotCmd << "every ::1 using 2:1:($1==$2 ? $4 : (-1.0-$4)) with image";
    plotCmd << "using 2:1:($1==$2 ? $4 : (-1.0-$4)) with image";

    if (nbTargets <= 10) {
        plotCmd << ", \"\" using 2:1:($3 > 0 ? "
                   "sprintf(\"%.f\\n%.02f%%\",$3,100.0*$4) : \"\") "
                   "with labels";
    }
    else if (nbTargets <= 50) {
        plotCmd << ", \"\" using 2:1:($3 > 0 ? sprintf(\"%.f\",$3) : \"\") "
                   "with labels";
    }

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName, plotCmd.str());

    Gnuplot::setDefaultOutput();
}

#endif // N2D2_CONFUSIONMATRIX_H
