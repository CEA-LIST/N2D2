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

#include "utils/Utils.hpp"

std::tuple<double, double, double>
N2D2::Utils::hsvToHsl(double hsvH, double hsvS, double hsvV)
{
    // Convert HSV to HSL
    double hslH = hsvH;
    double hslL = (2.0 - hsvS) * hsvV;
    double hslS = hsvS * hsvV;

    if (hslL <= 1.0) {
        if (hslL != 0.0)
            hslS /= hslL;
    } else if (hslL != 2.0)
        hslS /= 2.0 - hslL;

    hslL /= 2.0;

    return std::make_tuple(hslH, hslS, hslL);
}

std::tuple<double, double, double>
N2D2::Utils::hsvToRgb(double hsvH, double hsvS, double hsvV)
{
    if (hsvH < 0.0 || hsvH >= 360.0)
        throw std::domain_error(
            "hsvH is out of range (must be >= 0.0 and < 360.0)");

    if (hsvS < 0.0 || hsvS > 1.0)
        throw std::domain_error(
            "hsvS is out of range (must be >= 0.0 and <= 1.0)");

    if (hsvV < 0.0 || hsvV > 1.0)
        throw std::domain_error(
            "hsvV is out of range (must be >= 0.0 and <= 1.0)");

    const int hi = ((int)(hsvH / 60.0)) % 6;
    const double f = (hsvH / 60.0) - hi;
    const double l = hsvV * (1.0 - hsvS);
    const double m = hsvV * (1.0 - f * hsvS);
    const double n = hsvV * (1.0 - (1.0 - f) * hsvS);

    switch (hi) {
    case 0:
        return std::make_tuple(hsvV, n, l);
    case 1:
        return std::make_tuple(m, hsvV, l);
    case 2:
        return std::make_tuple(l, hsvV, n);
    case 3:
        return std::make_tuple(l, m, hsvV);
    case 4:
        return std::make_tuple(n, l, hsvV);
    case 5:
    default:
        return std::make_tuple(hsvV, l, m);
    }
}

std::tuple<double, double, double>
N2D2::Utils::rgbToHsv(double rgbR, double rgbG, double rgbB)
{
    if (rgbR < 0.0 || rgbR > 1.0)
        throw std::domain_error(
            "rgbR is out of range (must be >= 0.0 and <= 1.0)");

    if (rgbG < 0.0 || rgbG > 1.0)
        throw std::domain_error(
            "rgbG is out of range (must be >= 0.0 and <= 1.0)");

    if (rgbB < 0.0 || rgbB > 1.0)
        throw std::domain_error(
            "rgbB is out of range (must be >= 0.0 and <= 1.0)");

    const double max = std::max(rgbR, std::max(rgbG, rgbB));
    const double min = std::min(rgbR, std::min(rgbG, rgbB));

    const double hsvH
        = (max == min)
              ? 0.0
              : (max == rgbR)
                    ? std::fmod(60.0 * (rgbG - rgbB) / (max - min) + 360.0,
                                360.0)
                    : (max == rgbG)
                          ? 60.0 * (rgbB - rgbR) / (max - min) + 120.0
                          : 60.0 * (rgbR - rgbG) / (max - min) + 240.0;

    const double hsvS = (max == 0.0) ? 0.0 : (1.0 - min / max);
    const double hsvV = max;

    return std::make_tuple(hsvH, hsvS, hsvV);
}

std::tuple<double, double, double>
N2D2::Utils::rgbToYuv(double rgbR, double rgbG, double rgbB, bool normalize)
{
    if (rgbR < 0.0 || rgbR > 1.0)
        throw std::domain_error(
            "rgbR is out of range (must be >= 0.0 and <= 1.0)");

    if (rgbG < 0.0 || rgbG > 1.0)
        throw std::domain_error(
            "rgbG is out of range (must be >= 0.0 and <= 1.0)");

    if (rgbB < 0.0 || rgbB > 1.0)
        throw std::domain_error(
            "rgbB is out of range (must be >= 0.0 and <= 1.0)");

    // Constants
    const double wr = 0.299;
    const double wb = 0.114;
    const double wg = 1.0 - wr - wb;
    const double uMax = 0.436;
    const double vMax = 0.615;

    // YUV computation
    const double yuvY = wr * rgbR + wg * rgbG + wb * rgbB;
    const double yuvU = (normalize) ? ((rgbB - yuvY) / (1.0 - wb) + 1.0) / 2.0
                                    : uMax * (rgbB - yuvY) / (1.0 - wb);
    const double yuvV = (normalize) ? ((rgbR - yuvY) / (1.0 - wr) + 1.0) / 2.0
                                    : vMax * (rgbR - yuvY) / (1.0 - wr);

    return std::make_tuple(yuvY, yuvU, yuvV);
}

/**
 * See:
 * -
 * http://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
 * - http://docs.opencv.org/modules/core/doc/clustering.html
 * - opencv/samples/cpp/kmeans.cpp
*/
void N2D2::Utils::colorReduce(cv::Mat& img, unsigned int nbColors)
{
    if (img.depth() != CV_8U)
        throw std::runtime_error(
            "Utils::colorReduce(): only unsigned 8 bit frames are supported");

    const unsigned int rows = img.rows;
    const unsigned int cols = img.cols;
    const int type = img.type();

    // cv::kmeans() requires CV_32F
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    cv::Mat imgLab;

    if (img.channels() > 1) {
        // convert the image from the RGB color space to the L*a*b* color space
        // -- since we will be clustering using k-means which is
        // based on the euclidean distance, we'll use the L*a*b* color space
        // where the euclidean distance implies perceptual meaning
        cv::cvtColor(imgFloat, imgLab, CV_BGR2Lab);
    } else
        imgLab = imgFloat;

    // reshape the image into a feature vector so that k-means can be applied
    const cv::Mat imgVec = imgLab.reshape(0, cols * rows);

    // apply k-means using the specified number of clusters and then create the
    // quantized image based on the predictions
    cv::Mat bestLabels;
    const cv::TermCriteria criteria(
        CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0);
    const int attempts = 3;
    const int flags = cv::KMEANS_PP_CENTERS;
    cv::Mat centers;

#if CV_MAJOR_VERSION < 2 || (CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION < 3)
    cv::kmeans(
        imgVec, nbColors, bestLabels, criteria, attempts, flags, &centers);
#else
    // cv::kmeans() interface changed in 2.3.0
    // (http://upstream.rosalinux.ru/diffs/opencv/2.2.0_to_2.3.0/diff.html)
    cv::kmeans(
        imgVec, nbColors, bestLabels, criteria, attempts, flags, centers);
#endif

    bestLabels = bestLabels.reshape(1, rows);

    for (unsigned int i = 0; i < rows; ++i) {
        int* data = bestLabels.ptr<int>(i);

        for (unsigned int j = 0; j < cols; ++j) {
            const cv::Mat row = centers.row(data[j]);

            if (imgLab.channels() > 1)
                imgLab.at<cv::Vec3f>(i, j) = cv::Vec3f(row.at<float>(0, 0),
                                                       row.at<float>(0, 1),
                                                       row.at<float>(0, 2));
            else
                imgLab.at<float>(i, j) = row.at<float>(0, 0);
        }
    }

    if (img.channels() > 1)
        cv::cvtColor(imgLab, imgFloat, CV_Lab2BGR);
    else
        imgFloat = imgLab;

    imgFloat.convertTo(img, type, 255.0);
}

void N2D2::Utils::colorDiscretize(cv::Mat& img, unsigned int nbLevels)
{
    if (img.depth() != CV_8U)
        throw std::runtime_error("Utils::colorDiscretize(): only unsigned 8 "
                                 "bit frames are supported");

    const unsigned int rows = img.rows;
    const unsigned int elements = img.cols * img.channels();

    for (unsigned int i = 0; i < rows; ++i) {
        unsigned char* data = img.ptr<unsigned char>(i);

        for (unsigned int e = 0; e < elements; ++e)
            data[e] = 255 * Utils::round((nbLevels - 1) * data[e] / 255.0)
                      / (nbLevels - 1);
    }
}

std::string N2D2::Utils::cvMatDepthToString(int depth)
{
    switch (depth) {
    case CV_8U:
        return "8U";
    case CV_8S:
        return "8S";
    case CV_16U:
        return "16U";
    case CV_16S:
        return "16S";
    case CV_32S:
        return "32S";
    case CV_32F:
        return "32F";
    case CV_64F:
        return "64F";
    default:
        throw std::runtime_error(
            "Utils::cvMatDepthToString(): unknown cv::Mat depth");
    }
}

double N2D2::Utils::cvMatDepthUnityValue(int depth)
{
    switch (depth) {
    case CV_8U:
        return std::numeric_limits<unsigned char>::max();
    case CV_8S:
        return std::numeric_limits<char>::max();
    case CV_16U:
        return std::numeric_limits<unsigned short>::max();
    case CV_16S:
        return std::numeric_limits<short>::max();
    case CV_32S:
        return std::numeric_limits<int>::max();
    case CV_32F:
        return 1.0;
    case CV_64F:
        return 1.0;
    default:
        throw std::runtime_error(
            "Utils::cvMatDepthUnityValue(): unknown cv::Mat depth");
    }
}

std::string N2D2::Utils::searchAndReplace(const std::string& value,
                                          const std::string& search,
                                          const std::string& replace)
{
    std::string newValue(value);

    for (std::string::size_type next = newValue.find(search);
         next != std::string::npos;
         next = newValue.find(search, next)) {
        newValue.replace(next, search.length(), replace);
        next += replace.length();
    }

    return newValue;
}

std::string N2D2::Utils::escapeBinary(const std::string& value)
{
    std::string newValue;

    for (std::string::const_iterator it = value.begin(), itEnd = value.end();
         it != itEnd;
         ++it) {
        if (std::isprint(*it))
            newValue += *it;
        else {
            std::stringstream escaped;
            escaped << "\\x" << std::hex << std::uppercase << std::setfill('0')
                    << std::setw(2) << (int)*it;
            newValue += escaped.str();
        }
    }

    return newValue;
}

std::vector<std::string> N2D2::Utils::split(const std::string& value,
                                            const std::string& delimiters,
                                            bool trimEmpty)
{
    std::vector<std::string> result;
    std::string::size_type current = 0;
    std::string::size_type next;

    do {
        next = value.find_first_of(delimiters, current);
        const std::string chunk = value.substr(current, next - current);

        if (!trimEmpty || !chunk.empty())
            result.push_back(chunk);

        current = next + 1;
    } while (next != std::string::npos);

    return result;
}

std::string N2D2::Utils::upperCase(const std::string& str)
{
    std::string upperStr;
    std::transform(str.begin(),
                   str.end(),
                   std::back_inserter(upperStr),
                   std::bind(&::toupper, std::placeholders::_1));
    return upperStr;
}

std::string N2D2::Utils::lowerCase(const std::string& str)
{
    std::string lowerStr;
    std::transform(str.begin(),
                   str.end(),
                   std::back_inserter(lowerStr),
                   std::bind(&::tolower, std::placeholders::_1));
    return lowerStr;
}

std::string N2D2::Utils::ltrim(std::string s)
{
    s.erase(s.begin(),
            std::find_if(s.begin(),
                         s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// The main function that checks if two given strings
// match. The first string may contain wildcard characters
bool N2D2::Utils::match(const std::string& first, const std::string& second)
{
    // If we reach at the end of both strings, we are done
    if (first.empty() && second.empty())
        return true;

    // Make sure that the characters after '*' are present
    // in second string. This function assumes that the first
    // string will not contain two consecutive '*'
    if (first[0] == '*' && first.size() > 1 && second.empty())
        return false;

    // If the first string contains '?', or current characters
    // of both strings match
    if (first[0] == '?' || first[0] == second[0])
        return match(first.substr(1), second.substr(1));

    // If there is *, then there are two possibilities
    // a) We consider current character of second string
    // b) We ignore current character of second string.
    if (first[0] == '*')
        return match(first.substr(1), second) || match(first, second.substr(1));

    return false;
}

std::string N2D2::Utils::expandEnvVars(std::string str)
{
    size_t startPos = 0;

    while ((startPos = str.find("${", startPos)) != std::string::npos) {
        size_t endPos = str.find("}", startPos + 2);

        if (endPos == std::string::npos)
            return str;

        const std::string varName
            = str.substr(startPos + 2, endPos - startPos - 2);
        const char* varValue = std::getenv(varName.c_str());

        str.replace(startPos,
                    endPos - startPos + 1,
                    (varValue != NULL) ? varValue : "");
        startPos = endPos + 1;
    }

    return str;
}

bool N2D2::Utils::createDirectories(const std::string& dirName)
{
    std::stringstream path(dirName);
    std::string dir;
    std::string pathToDir("");
    int status = 0;

    while (std::getline(path, dir, '/') && status == 0) {
#ifdef WIN32
        pathToDir += dir;
#else
        pathToDir += dir + "/";
#endif
        struct stat fileStat;

        if (stat(pathToDir.c_str(), &fileStat) != 0) {
// Directory does not exist
#ifdef WIN32
            status = _mkdir(pathToDir.c_str());
#else
#ifdef S_IRWXU
            status = mkdir(pathToDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#else
            status = mkdir(pathToDir.c_str());
#endif
#endif
        } else if (!S_ISDIR(fileStat.st_mode))
            status = -1;

#ifdef WIN32
        pathToDir += "/";
#endif
    }

    return (status == 0);
}

std::string N2D2::Utils::dirName(const std::string& filePath)
{
    const size_t slashPos = filePath.find_last_of("/\\");
    return (slashPos == std::string::npos) ? "."
                                           : filePath.substr(0, slashPos + 1);
}

std::string N2D2::Utils::baseName(const std::string& filePath)
{
    const size_t slashPos = filePath.find_last_of("/\\");
    return (slashPos == std::string::npos) ? filePath
                                           : filePath.substr(slashPos + 1);
}

std::string N2D2::Utils::fileBaseName(const std::string& filePath)
{
    const size_t dotPos = filePath.find_last_of(".");
    if (dotPos == std::string::npos || dotPos == 0)
        return filePath;

    const size_t slashPos = filePath.find_last_of("/\\");
    return (slashPos != std::string::npos && dotPos <= slashPos + 1)
               ? filePath
               : filePath.substr(0, dotPos);
}

std::string N2D2::Utils::fileExtension(const std::string& filePath)
{
    const size_t dotPos = filePath.find_last_of(".");
    if (dotPos == std::string::npos || dotPos == 0)
        return "";

    const size_t slashPos = filePath.find_last_of("/\\");
    return (slashPos != std::string::npos && dotPos <= slashPos + 1)
               ? ""
               : filePath.substr(dotPos + 1);
}

bool N2D2::Utils::isNotValidIdentifier(int c) {
    return (!isalnum(c) && c != '_');
}

std::string N2D2::Utils::CIdentifier(const std::string& str) {
    std::string identifier(str);
    std::replace_if(identifier.begin(), identifier.end(),
                    Utils::isNotValidIdentifier, '_');

    if (!identifier.empty() && !isalpha(identifier[0]))
        identifier = "_" + identifier;

    return identifier;
}

double N2D2::Utils::normalInverse(double p)
{
    if (p < 0.0 || p > 1.0)
        throw std::domain_error(
            "p is out of range (must be >= 0.0 and <= 1.0)");

    const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                         -2.759285104469687e+02, 1.383577518672690e+02,
                         -3.066479806614716e+01, 2.506628277459239e+00};
    const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                         -1.556989798598866e+02, 6.680131188771972e+01,
                         -1.328068155288572e+01};
    const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                         -2.400758277161838e+00, -2.549732539343734e+00,
                         4.374664141464968e+00,  2.938163982698783e+00};
    const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                         2.445134137142996e+00, 3.754408661907416e+00};

    if (p == 0.0)
        return -std::numeric_limits<double>::infinity();
    else if (p == 1.0)
        return std::numeric_limits<double>::infinity();

    const double q = std::min(p, 1.0 - p);
    double u;

    if (q > 0.02425) {
        // Rational approximation for central region.
        u = q - 0.5;
        const double t = u * u;
        u *= (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t
              + a[5]) / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t
                          + b[4]) * t + 1.0);
    } else {
        // Rational approximation for tail region.
        const double t = std::sqrt(-2.0 * std::log(q));
        u = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t
             + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }

    return (p > 0.5 ? -u : u);
}

double N2D2::Utils::dPrime(unsigned int hits,
                           unsigned int yesTrials,
                           unsigned int falseAlarms,
                           unsigned int noTrials)
{
    double hitRate = hits / (double)yesTrials;
    double falseAlarmRate = falseAlarms / (double)noTrials;

    // Adjustments to avoid infinite values
    if (hitRate == 0.0)
        hitRate = 1.0 / (2 * yesTrials);
    else if (hitRate == 1.0)
        hitRate = 1.0 - 1.0 / (2 * yesTrials);

    if (falseAlarmRate == 0.0)
        falseAlarmRate = 1.0 / (2 * noTrials);
    else if (falseAlarmRate == 1.0)
        falseAlarmRate = 1.0 - 1.0 / (2 * noTrials);

    return (normalInverse(hitRate) - normalInverse(falseAlarmRate));
}

double N2D2::Utils::normalizedAngle(double angle, AngularRange range)
{
    // Preserve continuity of sin(x) and cos(x) better than modulo for high
    // angle values.
    double ang = std::atan2(std::sin(angle), std::cos(angle));

    // Output range of atan2() is [-pi,pi], but we want [-pi,pi[ = [-pi,pi)
    if (ang == M_PI)
        ang = -M_PI;

    if (range == ZeroToTwoPi && ang < 0)
        ang += 2.0 * M_PI;

    return ang;
}

#if CV_MINOR_VERSION < 2
void cv::vconcat(const std::vector<cv::Mat>& src, cv::Mat& dst)
{
    if (src.empty())
        return;

    int totalRows = 0, rows = 0;

    for (size_t i = 0; i < src.size(); ++i) {
        assert(src[i].cols == src[0].cols && src[i].type() == src[0].type());
        totalRows += src[i].rows;
    }

    dst = cv::Mat(totalRows, src[0].cols, src[0].type());

    for (size_t i = 0; i < src.size(); ++i) {
        cv::Mat dpart(dst, cv::Rect(0, rows, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        rows += src[i].rows;
    }
}
#endif
