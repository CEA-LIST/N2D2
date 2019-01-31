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
enum ConfusionTableMetric {
    Sensitivity,
    Specificity,
    Precision,
    NegativePredictiveValue,
    MissRate,
    FallOut,
    FalseDiscoveryRate,
    FalseOmissionRate,
    Accuracy,
    F1Score,
    Informedness,
    Markedness
};

template <class T>
class ConfusionTable {
public:
    ConfusionTable() : mTp(0), mTn(0), mFp(0), mFn(0) {};
    // Base metrics
    /// Sensitivity, recall, hit rate, or true positive rate (TPR)
    double sensitivity() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFn)) : 0.0;
    };
    /// Specificity, selectivity or true negative rate (TNR)
    double specificity() const
    {
        return (mTn > 0) ? (mTn / (double)(mTn + mFp)) : 0.0;
    };
    /// Precision or positive predictive value (PPV)
    double precision() const
    {
        return (mTp > 0) ? (mTp / (double)(mTp + mFp)) : 0.0;
    };
    /// Negative predictive value (NPV)
    double negativePredictiveValue() const
    {
        return (mTn > 0) ? (mTn / (double)(mTn + mFn)) : 0.0;
    };
    /// Miss rate or false negative rate (FNR)
    double missRate() const
    {
        return (1.0 - sensitivity());
    };
    /// Fall-out or false positive rate (FPR)
    double fallOut() const
    {
        return (1.0 - specificity());
    };
    /// False discovery rate (FDR)
    double falseDiscoveryRate() const
    {
        return (1.0 - precision());
    };
    /// False omission rate (FOR)
    double falseOmissionRate() const
    {
        return (1.0 - negativePredictiveValue());
    };
    // Combined metrics
    /// Accuracy (ACC)
    double accuracy() const
    {
        return ((mTp + mTn) > 0)
            ? ((mTp + mTn) / (double)(mTp + mTn + mFp + mFn)) : 0.0;
    };
    /// F-score
    double fScore(double beta = 1.0) const
    {
        return (mTp > 0) ? (1.0 + beta * beta) * (precision() * sensitivity())
                            / (beta * beta * precision() + sensitivity()) : 0.0;
    };
    /// Informedness or Bookmaker Informedness (BM)
    double informedness() const
    {
        return (sensitivity() + specificity() - 1.0);
    };
    /// Markedness (MK)
    double markedness() const
    {
        return (precision() + negativePredictiveValue() - 1.0);
    };
    // Get any metric
    double getMetric(ConfusionTableMetric metric) const {
        switch (metric) {
        // Base metrics
        case Sensitivity:
            return sensitivity();
        case Specificity:
            return specificity();
        case Precision:
            return precision();
        case NegativePredictiveValue:
            return negativePredictiveValue();
        case MissRate:
            return missRate();
        case FallOut:
            return fallOut();
        case FalseDiscoveryRate:
            return falseDiscoveryRate();
        case FalseOmissionRate:
            return falseOmissionRate();
        // Combined metrics
        case Accuracy:
            return accuracy();
        case F1Score:
            return fScore();
        case Informedness:
            return informedness();
        case Markedness:
            return markedness();
        default:
            throw std::runtime_error("ConfusionTable::getMetric(): "
                                     "unknown metric");
        }
    }
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
    /// true positive (TP), eqv. with hit
    T tp() const
    {
        return mTp;
    };
    /// true negative (TN), eqv. with correct rejection
    T tn() const
    {
        return mTn;
    };
    /// false positive (FP), eqv. with false alarm, Type I error
    T fp() const
    {
        return mFp;
    };
    /// false negative (FN), eqv. with miss, Type II error
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
    ConfusionTable<T> getConfusionTable(unsigned int target) const;
    std::vector<ConfusionTable<T> > getConfusionTables() const;
    void log(const std::string& fileName,
             const std::vector<std::string>& labels = std::vector
             <std::string>()) const;
    virtual ~ConfusionMatrix()
    {
    }
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ConfusionTableMetric>::data[]
    = {"Sensitivity",
    "Specificity",
    "Precision",
    "NegativePredictiveValue",
    "MissRate",
    "FallOut",
    "FalseDiscoveryRate",
    "FalseOmissionRate",
    "Accuracy",
    "F1Score",
    "Informedness",
    "Markedness"};
}

template <class T>
N2D2::ConfusionTable<T>
N2D2::ConfusionMatrix<T>::getConfusionTable(unsigned int target) const
{
    const unsigned int nbTargets = this->rows();
    ConfusionTable<T> conf;

    for (unsigned int estimated = 0; estimated < nbTargets; ++estimated) {
        if (target == estimated) {
            // True Positives
            conf.tp((*this)(target, estimated));
        }
        else {
            // False Negatives
            conf.fn((*this)(target, estimated));
            // False Positives
            conf.fp((*this)(estimated, target));

            // True Negatives
            for (unsigned int other = 0; other < nbTargets; ++other) {
                if (other != target)
                    conf.tn((*this)(other, estimated));
            }
        }
    }

    return conf;
}

template <class T>
std::vector<N2D2::ConfusionTable<T> >
N2D2::ConfusionMatrix<T>::getConfusionTables() const
{
    const unsigned int nbTargets = this->rows();
    std::vector<ConfusionTable<T> > confs;

    for (unsigned int target = 0; target < nbTargets; ++target)
        confs.push_back(getConfusionTable(target));

    return confs;
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

    for (unsigned int target = 0; target < nbTargets; ++target) {
        const std::vector<T>& row = this->row(target);
        const T targetCount = std::accumulate(row.begin(), row.end(), (T)0);

        total += targetCount;
        totalCorrect += (*this)(target, target);

        for (unsigned int estimated = 0; estimated < nbTargets; ++estimated) {
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

    {
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
            "defined (-2 'red', -1.01 '#FFEEEE', -1 'white',"
                " 0 'red', 1 'cyan')");
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

    const std::string confFile = Utils::fileBaseName(fileName) + "_score."
                                 + Utils::fileExtension(fileName);
    confData.open(confFile);

    if (!confData.good())
        throw std::runtime_error("Could not save confusion data file: "
                                 + confFile);

    confData << "Target TargetName Sensitivity Specificity Precision"
                " Accuracy F1-score Informedness\n";

    const std::vector<ConfusionTable<T> > conf = getConfusionTables();
    double avgSensitivity = 0.0;
    double avgSpecificity = 0.0;
    double avgPrecision = 0.0;
    double avgAccuracy = 0.0;
    double avgF1Score = 0.0;
    double avgInformedness = 0.0;
    unsigned int maxLabelSize = 3;

    for (unsigned int target = 0; target < nbTargets; ++target) {
        avgSensitivity += conf[target].sensitivity();
        avgSpecificity += conf[target].specificity();
        avgPrecision += conf[target].precision();
        avgAccuracy += conf[target].accuracy();
        avgF1Score += conf[target].fScore();
        avgInformedness += conf[target].informedness();

        std::stringstream labelStr;

        if (!labels.empty())
            labelStr << labels[target];
        else
            labelStr << target;

        if (labelStr.str().size() > maxLabelSize)
            maxLabelSize = labelStr.str().size();

        confData << target << " \"" << labelStr.str() << "\""
            << " " << conf[target].sensitivity()
            << " " << conf[target].specificity()
            << " " << conf[target].precision()
            << " " << conf[target].accuracy()
            << " " << conf[target].fScore()
            << " " << conf[target].informedness() << "\n";
    }

    avgSensitivity /= nbTargets;
    avgSpecificity /= nbTargets;
    avgPrecision /= nbTargets;
    avgAccuracy /= nbTargets;
    avgF1Score /= nbTargets;
    avgInformedness /= nbTargets;

    confData << "\n";
    confData << "- AVG"
        << " " << avgSensitivity
        << " " << avgSpecificity
        << " " << avgPrecision
        << " " << avgAccuracy
        << " " << avgF1Score
        << " " << avgInformedness << "\n";

    confData.close();

    {
        std::stringstream outputStr;
        outputStr << "size " << ((nbTargets + 1) * 100 + 150)
                            << ",600 enhanced";

        Gnuplot::setDefaultOutput("png", outputStr.str(), "png");

        Gnuplot gnuplot(confFile + ".gnu");
        gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
                   "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
        gnuplot.set("key outside");
        gnuplot.set("style histogram cluster gap 3");
        gnuplot.set("style data histograms");
        gnuplot.set("style fill pattern 1.00 border");
        gnuplot.set("mytics 10");
        gnuplot.set("grid");
        gnuplot.set("tmargin", 4);
        gnuplot.set("bmargin", (maxLabelSize / 2) + 2);
        gnuplot.set("xtics rotate");
        gnuplot.set("boxwidth 1.0");

        std::stringstream plotCmd;
        plotCmd << "i 0 using 3:xticlabels(wrap(stringcolumn(2),"
                << maxLabelSize << ")) ti col, "
                     "'' i 0 using 4 ti col, "
                     "'' i 0 using 5 ti col, "
                     "'' i 0 using 6 ti col, "
                     "'' i 0 using 7 ti col, "
                     "'' i 0 using 8 ti col";

        gnuplot.saveToFile(confFile);
        gnuplot.plot(confFile, plotCmd.str());

        Gnuplot::setDefaultOutput();
    }
}

#endif // N2D2_CONFUSIONMATRIX_H
