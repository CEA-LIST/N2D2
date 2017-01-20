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

#include "utils/Gnuplot.hpp"

FILE* N2D2::Gnuplot::mMasterCmdPipe = NULL;
std::tuple<std::string, std::string, std::string> N2D2::Gnuplot::mDefaultOutput
    = std::make_tuple<std::string, std::string, std::string>(
        "png", "size 800,600 enhanced large", "png");

N2D2::Gnuplot::Gnuplot(const std::string& fileName) : mSubPipe(false)
{
    if (mMasterCmdPipe == NULL) {
#ifdef WIN32
        mCmdPipe = _popen("pgnuplot.exe", "w");
#else
        mCmdPipe = popen("gnuplot", "w");
#endif

        if (mCmdPipe == NULL)
            throw std::runtime_error(
                "Couldn't open connection to gnuplot (is it in the PATH?)");
    } else {
        mSubPipe = true;
        mCmdPipe = mMasterCmdPipe;
    }

    if (!fileName.empty()) {
        mCmdFile.open(fileName.c_str());
#ifndef WIN32
        // Render the command file executable to allow on-click plot
        // regeneration
        mCmdFile << "#!/usr/bin/gnuplot" << std::endl;
        chmod(fileName.c_str(), 0755);
#endif

        if (!mCmdFile.good())
            throw std::runtime_error("Couldn't create gnuplot command file");
    }
}

N2D2::Gnuplot& N2D2::Gnuplot::readCmd(const std::string& fileName)
{
    std::ifstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not open command file: " + fileName);

    std::string line;

    while (std::getline(dataFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        *this << line;
    }

    return *this;
}

N2D2::Gnuplot& N2D2::Gnuplot::setMultiplot(unsigned int rows, unsigned int cols)
{
    if (mSubPipe)
        return *this;

    mMasterCmdPipe = mCmdPipe;

    if (rows > 0 && cols > 0) {
        std::ostringstream cmdStr;
        cmdStr << "if (!exists(\"multiplot\")) set multiplot layout " << rows
               << "," << cols;
        *this << cmdStr.str();
    } else
        *this << "if (!exists(\"multiplot\")) set multiplot";

    return (*this << "if (!exists(\"multiplot\")) multiplot=1");
}

N2D2::Gnuplot& N2D2::Gnuplot::unsetMultiplot()
{
    if (mSubPipe)
        return *this;

    mMasterCmdPipe = NULL;
    return (*this << "unset multiplot");
}

N2D2::Gnuplot& N2D2::Gnuplot::setSize(double width, double height)
{
    if (mSubPipe)
        return *this;

    std::ostringstream cmdStr;
    cmdStr << "set size " << width << "," << height;
    return (*this << cmdStr.str());
}

N2D2::Gnuplot& N2D2::Gnuplot::setOrigin(double x, double y)
{
    if (mSubPipe)
        return *this;

    std::ostringstream cmdStr;
    cmdStr << "set origin " << x << "," << y;
    return (*this << cmdStr.str());
}

N2D2::Gnuplot& N2D2::Gnuplot::setTitle(const std::string& title)
{
    std::string cmdStr = "set title \"";
    cmdStr += title + "\"";
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::setXlabel(const std::string& xlabel,
                                        const std::string& optArgs)
{
    std::string cmdStr = "set xlabel \"";
    cmdStr += xlabel + "\" ";
    cmdStr += optArgs;
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::setYlabel(const std::string& ylabel,
                                        const std::string& optArgs)
{
    std::string cmdStr = "set ylabel \"";
    cmdStr += ylabel + "\" ";
    cmdStr += optArgs;
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::setY2label(const std::string& ylabel,
                                         const std::string& optArgs)
{
    std::string cmdStr = "set y2label \"";
    cmdStr += ylabel + "\" ";
    cmdStr += optArgs;
    return (*this << cmdStr);
}

N2D2::Gnuplot&
N2D2::Gnuplot::setXrange(double xmin, double xmax, const std::string& optArgs)
{
    std::ostringstream cmdStr;
    cmdStr << "set xrange [" << xmin << ":" << xmax << "] " << optArgs;
    return (*this << cmdStr.str());
}

N2D2::Gnuplot&
N2D2::Gnuplot::setYrange(double ymin, double ymax, const std::string& optArgs)
{
    std::ostringstream cmdStr;
    cmdStr << "set yrange [" << ymin << ":" << ymax << "] " << optArgs;
    return (*this << cmdStr.str());
}

N2D2::Gnuplot&
N2D2::Gnuplot::setY2range(double ymin, double ymax, const std::string& optArgs)
{
    std::ostringstream cmdStr;
    cmdStr << "set y2range [" << ymin << ":" << ymax << "] " << optArgs;
    return (*this << cmdStr.str());
}

N2D2::Gnuplot& N2D2::Gnuplot::plot(const std::string& dataFile,
                                   const std::string& plotCmd)
{
    std::string cmdStr = "plot \"";
    cmdStr += dataFile + "\" ";
    cmdStr += plotCmd;
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::splot(const std::string& dataFile,
                                    const std::string& plotCmd)
{
    std::string cmdStr = "splot \"";
    cmdStr += dataFile + "\" ";
    cmdStr += plotCmd;
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::showOnScreen()
{
#if defined(WIN32)
    *this << "if (!exists(\"multiplot\")) set term windows";
#elif defined(__APPLE__)
    *this << "if (!exists(\"multiplot\")) set term aqua";
#else
    *this << "if (!exists(\"multiplot\")) set term x11";
#endif

    return (*this << "if (!exists(\"multiplot\")) set output");
}

void N2D2::Gnuplot::close()
{
    if (mCmdFile.is_open())
        mCmdFile.close();

    if (!mSubPipe && mCmdPipe != NULL) {
        mMasterCmdPipe = NULL;
#ifdef WIN32
        std::thread(_pclose, mCmdPipe).detach();
#else
        std::thread(pclose, mCmdPipe).detach();
#endif
        mCmdPipe = NULL;
    }
}

N2D2::Gnuplot& N2D2::Gnuplot::saveToFile(const std::string& fileName,
                                         const std::string& suffix)
{
    std::string newName, fileFormat;

    if (suffix.empty()) {
        newName = fileName;
        fileFormat = Utils::fileExtension(fileName);
    } else {
        newName = Utils::fileBaseName(fileName) + suffix;
        fileFormat = Utils::fileExtension(suffix);
    }

    std::transform(
        fileFormat.begin(), fileFormat.end(), fileFormat.begin(), ::tolower);

    if (fileFormat == "ps")
        *this
            << "if (!exists(\"multiplot\")) set term postscript enhanced color";
    else if (fileFormat == "eps")
        *this << "if (!exists(\"multiplot\")) set term postscript eps enhanced "
                 "color";
    else if (fileFormat == "png")
        *this
            << "if (!exists(\"multiplot\")) set term png size 800,600 enhanced";
    else {
        std::string cmdStr = "if (!exists(\"multiplot\")) set term ";
        cmdStr += std::get<0>(mDefaultOutput) + " ";
        cmdStr += std::get<1>(mDefaultOutput);

        *this << cmdStr;

        // Append default extension to the output file name
        if (!fileFormat.empty())
            newName = Utils::fileBaseName(newName);

        newName += ".";
        newName += std::get<2>(mDefaultOutput);
    }

    std::string cmdStr = "if (!exists(\"multiplot\")) set output \"";
    cmdStr += newName + "\" ";
    return (*this << cmdStr);
}

N2D2::Gnuplot& N2D2::Gnuplot::operator<<(const std::string& cmd)
{
    if (mCmdFile.is_open())
        mCmdFile << cmd << "\n";

    fputs((cmd + "\n").c_str(), mCmdPipe);
    fflush(mCmdPipe);

    return *this;
}

void N2D2::Gnuplot::setDefaultOutput(const std::string& terminal,
                                     const std::string& options,
                                     const std::string& format)
{
    mDefaultOutput = std::make_tuple(terminal, options, format);
}

N2D2::Gnuplot::~Gnuplot()
{
    close();
}
