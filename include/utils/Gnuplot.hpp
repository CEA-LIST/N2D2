/*
    (C) Copyright 2011 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_GNUPLOT_H
#define N2D2_GNUPLOT_H

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#ifndef WIN32
#include <sys/stat.h>
#endif

#include "utils/Utils.hpp"

namespace N2D2 {
/**
 * Wrapper for using gnuplot interactively (through a pipe).
*/
class Gnuplot {
public:
    /**
     * Open a pipe with Gnuplot.
     *
     * @param fileName      If not empty, the commands sent to gnuplot are also
     *saved in the file named @p fileName
    */
    Gnuplot(const std::string& fileName = "");

    Gnuplot& readCmd(const std::string& fileName);

    /**
     * Open a multiplot environment.
     * Send the gnuplot command "set multiplot" and redirect the pipe from any
     *Gnuplot object created afterward in the pipe used
     * by this object. The calls to showOnScreen() and saveToFile() methods are
     *also ignored for any sub Gnuplot object created
     * afterward. Thus, it is possible to capture plots created by function call
     *and include them as a subplot of this Gnuplot
     * object.
     *
     * @note Has no effect when called in a multiplot environment opened by an
     *other Gnuplot object.
    */
    Gnuplot& setMultiplot(unsigned int rows = 0, unsigned int cols = 0);

    /**
     * Close the multiplot environment.
     * Send the gnuplot command "unset multiplot".
     *
     * @note Has no effect when called in a multiplot environment opened by an
     *other Gnuplot object.
    */
    Gnuplot& unsetMultiplot();
    bool isMultiplot() const
    {
        return mSubPipe;
    };
    Gnuplot& setSize(double width, double height);
    Gnuplot& setOrigin(double x, double y);
    Gnuplot& set(const std::string& setCmd)
    {
        return (*this << ("set " + setCmd));
    };
    Gnuplot& unset(const std::string& unsetCmd)
    {
        return (*this << ("unset " + unsetCmd));
    };
    template <class T> Gnuplot& set(const std::string& setCmd, const T& setArg);
    Gnuplot& setTitle(const std::string& title);
    Gnuplot& setXlabel(const std::string& xlabel,
                       const std::string& optArgs = "");
    Gnuplot& setYlabel(const std::string& ylabel,
                       const std::string& optArgs = "");
    Gnuplot& setY2label(const std::string& ylabel,
                        const std::string& optArgs = "");
    Gnuplot&
    setXrange(double xmin, double xmax, const std::string& optArgs = "");
    Gnuplot&
    setYrange(double ymin, double ymax, const std::string& optArgs = "");
    Gnuplot&
    setY2range(double ymin, double ymax, const std::string& optArgs = "");
    Gnuplot& plot(const std::string& dataFile, const std::string& plotCmd = "");
    Gnuplot& splot(const std::string& dataFile,
                   const std::string& plotCmd = "");
    Gnuplot& showOnScreen();
    void close();

    /**
     * Specify the output file for the next plot command of Gnuplot.
     * The output file format is determined by the @p fileName extension, or @p
     *suffix extension if present. If no extension is
     * present, or if the extension is not a valid output format for Gnuplot,
     *the default format is used
     * (see Gnuplot::setDefaultOutput).
     *
     * Supported file extensions are "png", "ps" and "eps".
     *
     * @param fileName Name of the output file.
     * @param suffix If not empty, the file extension in @p fileName is replaced
     *by @p suffix, including the dot (".").
    */
    Gnuplot& saveToFile(const std::string& fileName,
                        const std::string& suffix = "");
    virtual ~Gnuplot();

    Gnuplot& operator<<(const std::string& cmd);
    template <class T1, class T2>
    Gnuplot& operator<<(const std::pair<T1, T2>& data);

    static void setDefaultOutput(const std::string& terminal = "png",
                                 const std::string& options
                                 = "size 800,600 enhanced large",
                                 const std::string& format = "png");

private:
    Gnuplot(const Gnuplot&); // non construction-copyable
    const Gnuplot& operator=(const Gnuplot&); // non-copyable

    bool mSubPipe;
    FILE* mCmdPipe;
    std::ofstream mCmdFile;

    static FILE* mMasterCmdPipe;
    static std::tuple<std::string, std::string, std::string> mDefaultOutput;
};
}

template <class T>
N2D2::Gnuplot& N2D2::Gnuplot::set(const std::string& setCmd, const T& setArg)
{
    std::ostringstream cmdStr;
    cmdStr << "set " << setCmd << " " << setArg;
    return (*this << cmdStr.str());
}

template <class T1, class T2>
N2D2::Gnuplot& N2D2::Gnuplot::operator<<(const std::pair<T1, T2>& data)
{
    std::stringstream cmd;
    cmd << data.first << " " << data.second << std::endl;

    if (mCmdFile.is_open())
        mCmdFile << cmd.str();

    fputs(cmd.str().c_str(), mCmdPipe);
    return *this;
}

#endif // N2D2_GNUPLOT_H
