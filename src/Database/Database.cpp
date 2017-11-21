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

#include "Database/Database.hpp"

const std::locale
N2D2::Database::csvLocale(std::locale(),
                          new N2D2::Utils::streamIgnore(",; \t"));

N2D2::Database::Database(bool loadDataInMemory)
    : mDefaultLabel(this, "DefaultLabel", ""),
      mROIsMargin(this, "ROIsMargin", 0U),
      mRandomPartitioning(this, "RandomPartitioning", true),
      mLoadDataInMemory(loadDataInMemory),
      mStimuliDepth(-1)
{
    // ctor
}

void N2D2::Database::loadROIs(const std::string& fileName,
                              const std::string& relPath,
                              bool noImageSize)
{
    std::ifstream dataRoi(fileName.c_str());

    if (!dataRoi.good())
        throw std::runtime_error(
            "Database::loadROIs(): could not open ROI data file: " + fileName);

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    // Find all stimuli within the relPath path
    const std::map<std::string, StimulusID> stimuliName
        = getRelPathStimuli(fileName, relPath);

    std::string line;

    while (std::getline(dataRoi, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '#'), line.end());
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        std::stringstream values(line);
        values.imbue(csvLocale);

        std::string name;

        if (!(values >> name))
            throw std::runtime_error("Unreadable line in data file: "
                                     + fileName);

        // There is a ROI
        if (!noImageSize) {
            unsigned int width, height;

            if (!(Utils::signChecked<unsigned int>(values) >> width)
                || !(Utils::signChecked<unsigned int>(values) >> height)) {
                std::cout << Utils::cwarning
                          << "Warning: unreadable image size value on line \""
                          << line << "\" in data file: " << fileName
                          << Utils::cdef << std::endl;
                continue;
            }
        }

        double x1, y1, x2, y2;
        std::string label;

        if (!(values >> x1)
            || !(values >> y1)
            || !(values >> x2)
            || !(values >> y2)
            || !(values >> label)) {
            throw std::runtime_error("Unreadable value in data file: "
                                     + fileName);
        }

        if (x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0) {
            std::cout << Utils::cwarning
                      << "Warning: negative coordinates on line \""
                      << line << "\" in data file: " << fileName
                      << Utils::cdef << std::endl;
        }

        if (!values.eof())
            throw std::runtime_error("Extra data at end of line in data file: "
                                     + fileName);

        // Find corresponding stimulus
        std::map<std::string, StimulusID>::const_iterator it
            = stimuliName.find(name);

        if (it != stimuliName.end()) {
            mStimuli[(*it).second].ROIs.push_back(
                new RectangularROI<int>(labelID(label),
                                        RectangularROI<int>::Point_T(x1, y1),
                                        RectangularROI<int>::Point_T(x2, y2)));
        } else {
            std::cout << Utils::cwarning
                      << "Warning: ignoring ROI for non-existant stimulus: \""
                      << name << "\"" << Utils::cdef << std::endl;
        }
    }
}

void N2D2::Database::loadROIsDir(const std::string& dirName,
                                 const std::string& fileExt,
                                 int depth)
{
    DIR* pDir = opendir(dirName.c_str());

    if (pDir == NULL)
        throw std::runtime_error("Couldn't open ROIs database directory: "
                                 + dirName);

    struct dirent* pFile;
    struct stat fileStat;
    std::vector<std::string> subDirs;
    std::vector<std::string> files;

    std::cout << "Loading directory ROIs database \"" << dirName << "\""
              << std::endl;

    while ((pFile = readdir(pDir))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(dirName + "/" + fileName);

        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        if (S_ISDIR(fileStat.st_mode))
            subDirs.push_back(filePath);
        else {
            // Ignore files with the wrong file extension
            if (!fileExt.empty() && Utils::fileExtension(fileName) != fileExt)
                continue;

            files.push_back(filePath);
        }
    }

    closedir(pDir);

    // Sorting files and subDirs for two reasons:
    // - Deterministic behavior regardless of the OS
    // - Label IDs get attributed to the ROIs in the same order than the stimuli
    // are loaded
    std::sort(files.begin(), files.end());

    for (std::vector<std::string>::const_iterator it = files.begin(),
                                                  itEnd = files.end();
         it != itEnd;
         ++it)
        loadROIs(*it, dirName);

    if (depth != 0) {
        std::sort(subDirs.begin(), subDirs.end());

        for (std::vector<std::string>::const_iterator it = subDirs.begin(),
                                                      itEnd = subDirs.end();
             it != itEnd;
             ++it)
            loadROIsDir(*it, fileExt, depth - 1);
    }
}

void N2D2::Database::saveROIs(const std::string& fileName,
                              const std::string& header) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create data file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    data << "# " << std::asctime(
                        localNow); // std::asctime() already appends end of line
    data << header;

    std::map<int, unsigned int> stats;

    for (std::vector<Stimulus>::const_iterator it = mStimuli.begin(),
                                               itEnd = mStimuli.end();
         it != itEnd;
         ++it) {
        if ((*it).ROIs.empty())
            data << (*it).name << " # No ROI for this stimulus\n";
        else {
            for (std::vector<ROI*>::const_iterator itRoi = (*it).ROIs.begin(),
                                                   itRoiEnd = (*it).ROIs.end();
                 itRoi != itRoiEnd;
                 ++itRoi) {
                const cv::Rect rect = (*itRoi)->getBoundingRect();

                data << (*it).name << " " << rect.x << " " << rect.y << " "
                     << (rect.x + rect.width) << " " << (rect.y + rect.height)
                     << " " << getLabelName((*itRoi)->getLabel()) << "\n";

                std::map<int, unsigned int>::iterator itStats;
                std::tie(itStats, std::ignore)
                    = stats.insert(std::make_pair((*itRoi)->getLabel(), 0U));
                ++(*itStats).second;
            }
        }
    }

    std::cout << "Number of classes: " << stats.size() << std::endl;

    for (std::map<int, unsigned int>::const_iterator it = stats.begin(),
                                                     itEnd = stats.end();
         it != itEnd;
         ++it)
        std::cout << "  class \"" << getLabelName((*it).first)
                  << "\": " << (*it).second << " patterns" << std::endl;
}

void N2D2::Database::logROIsStats(const std::string& sizeFileName,
                                  const std::string& labelFileName,
                                  StimuliSetMask setMask) const
{
    // Stats collection
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> sizeStats;
    unsigned int minWidth = std::numeric_limits<int>::max();
    unsigned int minHeight = std::numeric_limits<int>::max();
    unsigned int maxWidth = 0;
    unsigned int maxHeight = 0;
    unsigned int nbROIs = 0;
    std::map<int, unsigned int> labelStats;

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet) {
        for (unsigned int i = 0, size = mStimuliSets(*itSet).size(); i < size;
             ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            for (std::vector<ROI*>::const_iterator itROIs
                 = mStimuli[id].ROIs.begin(),
                 itROIsEnd = mStimuli[id].ROIs.end();
                 itROIs != itROIsEnd;
                 ++itROIs) {
                const cv::Rect bb = (*itROIs)->getBoundingRect();

                std::map<std::pair<unsigned int, unsigned int>,
                         unsigned int>::iterator itSizeStats;
                std::tie(itSizeStats, std::ignore) = sizeStats.insert(
                    std::make_pair(std::make_pair(bb.width, bb.height), 0U));
                ++(*itSizeStats).second;

                if (bb.width > (int)maxWidth)
                    maxWidth = bb.width;
                if (bb.height > (int)maxHeight)
                    maxHeight = bb.height;
                if (bb.width < (int)minWidth)
                    minWidth = bb.width;
                if (bb.height < (int)minHeight)
                    minHeight = bb.height;

                std::map<int, unsigned int>::iterator itLabelStats;
                std::tie(itLabelStats, std::ignore) = labelStats.insert(
                    std::make_pair((*itROIs)->getLabel(), 0U));
                ++(*itLabelStats).second;

                ++nbROIs;
            }
        }
    }

    if (nbROIs == 0) {
        std::cout << "Database::logROIsStats(): no ROI" << std::endl;
        return;
    }

    // Save size stats
    std::ofstream sizeData(sizeFileName.c_str());

    if (!sizeData.good())
        throw std::runtime_error("Could not create data file: " + sizeFileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    sizeData << "# " << std::asctime(localNow); // std::asctime() already
    // appends end of line

    for (unsigned int w = 0; w <= maxWidth; ++w) {
        for (unsigned int h = 0; h <= maxHeight; ++h) {
            const std::map<std::pair<unsigned int, unsigned int>,
                           unsigned int>::const_iterator itSizeStats
                = sizeStats.find(std::make_pair(w, h));

            if (itSizeStats != sizeStats.end())
                sizeData << w << " " << h << " " << (*itSizeStats).second
                         << "\n";
        }
    }

    sizeData.close();

    // Plot size stats
    Gnuplot sizeGnuplot;
    sizeGnuplot.set("grid");
    sizeGnuplot.unset("key");
    sizeGnuplot.set("view map");
    sizeGnuplot.set("origin -0.05,-0.05");
    sizeGnuplot.set("size 1.1,1.1");

    std::stringstream labelStr;
    labelStr << "ROIs width (range = [" << minWidth << ", " << maxWidth << "])";
    sizeGnuplot.setXlabel(labelStr.str());

    labelStr.str(std::string());
    labelStr << "ROIs height (range = [" << minHeight << ", " << maxHeight
             << "])";
    sizeGnuplot.setYlabel(labelStr.str());

    labelStr.str(std::string());
    labelStr << "title \"Total number of ROIs = " << nbROIs << "\"";
    sizeGnuplot.set(labelStr.str());

    sizeGnuplot.saveToFile(sizeFileName);
    sizeGnuplot.splot(
        sizeFileName,
        "using 1:2:3 with points pointtype 5 pointsize 1 palette linewidth 2");

    sizeGnuplot << "reset";
    sizeGnuplot.set("grid");
    sizeGnuplot.unset("key");
    sizeGnuplot.setYlabel("Number of stimuli (cumulative)");
    sizeGnuplot.setXlabel("ROIs width (pixels)");
    sizeGnuplot.saveToFile(sizeFileName, "-width");
    sizeGnuplot.set("multiplot");
    sizeGnuplot.plot(sizeFileName,
                     "using ($1+$0/1.0e12):3 smooth cumulative with steps");

    sizeGnuplot.set("origin 0.5,0.1");
    sizeGnuplot.set("size 0.45,0.4");
    sizeGnuplot.set("object rectangle from screen 0.5,0.1 to screen 0.95,0.5 "
                    "behind fillcolor rgb 'white'"
                    "fillstyle solid noborder");
    sizeGnuplot.setXrange(minWidth, minWidth + (maxWidth - minWidth)/10);
    sizeGnuplot.unset("xlabel");
    sizeGnuplot.unset("ylabel");
    sizeGnuplot.plot(sizeFileName,
                     "using ($1+$0/1.0e12):3 smooth cumulative with steps");
    sizeGnuplot.unset("multiplot");

    sizeGnuplot << "reset";
    sizeGnuplot.set("grid");
    sizeGnuplot.unset("key");
    sizeGnuplot.setYlabel("Number of stimuli (cumulative)");
    sizeGnuplot.setXlabel("ROIs height (pixels)");
    sizeGnuplot.saveToFile(sizeFileName, "-height");
    sizeGnuplot.set("multiplot");
    sizeGnuplot.plot(sizeFileName,
                     "using ($2+$0/1.0e12):3 smooth cumulative with steps");

    sizeGnuplot.set("origin 0.5,0.1");
    sizeGnuplot.set("size 0.45,0.4");
    sizeGnuplot.set("object rectangle from screen 0.5,0.1 to screen 0.95,0.5 "
                    "behind fillcolor rgb 'white'"
                    "fillstyle solid noborder");
    sizeGnuplot.setXrange(minHeight, minHeight + (maxHeight - minHeight)/10);
    sizeGnuplot.unset("xlabel");
    sizeGnuplot.unset("ylabel");
    sizeGnuplot.plot(sizeFileName,
                     "using ($2+$0/1.0e12):3 smooth cumulative with steps");
    sizeGnuplot.unset("multiplot");

    // Save label stats
    std::ofstream labelData(labelFileName.c_str());

    if (!sizeData.good())
        throw std::runtime_error("Could not create data file: "
                                 + labelFileName);

    // Append date & time to the file.
    labelData << "# " << std::asctime(localNow); // std::asctime() already
    // appends end of line

    for (std::map<int, unsigned int>::const_iterator it = labelStats.begin(),
                                                     itEnd = labelStats.end();
         it != itEnd;
         ++it)
    {
        labelData << "\"" << getLabelName((*it).first) << "\" "
            << (*it).second << "\n";
    }

    labelData.close();

    // Plot label stats
    Gnuplot labelGnuplot;
    labelGnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
                    "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    labelGnuplot.set("style histogram cluster gap 1");
    labelGnuplot.set("style data histograms");
    labelGnuplot.set("style fill pattern 1.00 border");
    labelGnuplot.set("ytics nomirror");
    labelGnuplot.setYlabel("ROIs label occurrence(s)");
    labelGnuplot.set("grid");
    labelGnuplot.set("xtics rotate by 90 right");
    labelGnuplot.set("ytics textcolor lt 1");
    labelGnuplot.set("ylabel textcolor lt 1");
    labelGnuplot.set("bmargin 10");
    labelGnuplot.unset("key");
    labelGnuplot.saveToFile(labelFileName);
    labelGnuplot.plot(
        labelFileName,
        "using ($2):xticlabels(wrap(stringcolumn(1),10)) lt 1,"
        " '' using 0:($2):($2) with labels offset char 0,1 textcolor lt 1");
}

void N2D2::Database::extractROIs()
{
    for (int id = mStimuli.size() - 1; id >= 0; --id) {
        if (!mStimuli[id].ROIs.empty()) {
            std::vector<ROI*>::const_iterator it = mStimuli[id].ROIs.begin();
            mStimuli[id].label = (*it)->getLabel();
            ++it;

            for (std::vector<ROI*>::const_iterator itEnd
                 = mStimuli[id].ROIs.end();
                 it != itEnd;
                 ++it) {
                mStimuli.push_back(Stimulus(mStimuli[id].name,
                                            (*it)->getLabel(),
                                            std::vector<ROI*>(1, *it)));
                mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
            }

            mStimuli[id].ROIs.resize(1);
        } else
            removeStimulus(id);
    }
}

void N2D2::Database::filterROIs(const std::vector<int>& labels,
                                bool filterKeep,
                                bool removeStimuli)
{
    unsigned int nbRoi = 0;
    unsigned int nbRoiRemoved = 0;
    const unsigned int nbStimuli = mStimuli.size();
    unsigned int nbStimuliRemoved = 0;

    for (int id = mStimuli.size() - 1; id >= 0; --id) {
        for (int roiId = mStimuli[id].ROIs.size() - 1; roiId >= 0; --roiId) {
            ++nbRoi;

            if ((filterKeep && std::find(labels.begin(), labels.end(),
                mStimuli[id].ROIs[roiId]->getLabel()) == labels.end())
                || (!filterKeep && std::find(labels.begin(), labels.end(),
                    mStimuli[id].ROIs[roiId]->getLabel()) != labels.end()))
            {
                ++nbRoiRemoved;
                mStimuli[id].ROIs.erase(mStimuli[id].ROIs.begin() + roiId);
            }
        }

        if (mStimuli[id].ROIs.empty() && removeStimuli) {
            ++nbStimuliRemoved;
            removeStimulus(id);
        }
    }

    std::cout << "Database::filterROIs():\n"
        "    Remaining ROIs: " << (nbRoi - nbRoiRemoved) << "/" << nbRoi << "\n"
        << "    Remaining stimuli: " << (nbStimuli - nbStimuliRemoved) << "/"
        << nbStimuli << std::endl;
}

void N2D2::Database::extractLabels(bool removeROIs)
{
    const int defaultLabel = getDefaultLabelID();

    for (int id = mStimuli.size() - 1; id >= 0; --id) {
        if (mStimuli[id].ROIs.size() > 0) {
            if (mStimuli[id].ROIs.size() > 1) {
                std::cout << "Stimulus #" << id << " (" << mStimuli[id].name
                          << ") has " << mStimuli[id].ROIs.size() << " ROIs"
                          << std::endl;

                throw std::runtime_error("Database::extractLabels(): labels "
                                         "can be extracted only if each "
                                         "stimulus has one"
                                         " and only one ROI");
            }

            mStimuli[id].label = mStimuli[id].ROIs[0]->getLabel();

            if (removeROIs) {
                std::for_each(mStimuli[id].ROIs.begin(),
                              mStimuli[id].ROIs.end(),
                              Utils::Delete());
                mStimuli[id].ROIs.clear();
            }
        } else
            mStimuli[id].label = defaultLabel;
    }
}

void N2D2::Database::extractSlices(unsigned int width,
                                   unsigned int height,
                                   unsigned int strideX,
                                   unsigned int strideY,
                                   StimuliSetMask setMask)
{
    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    // For progression visualization
    unsigned int toSlice = 0;

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it) {
        toSlice += getNbStimuli(*it);
    }

    std::cout << "Database: slicing " << toSlice << " stimuli" << std::flush;

    unsigned int sliced = 0;
    unsigned int progress = 0, progressPrev = 0;

    for (std::vector<StimuliSet>::const_iterator itSet = stimuliSets.begin(),
                                                 itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet) {
        const int size = mStimuliSets(*itSet).size();
        std::vector<Stimulus> newStimuli;

#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            if (mStimuli[id].slice != NULL)
                throw std::runtime_error("Database::extractSlices(): some "
                                         "stimuli are already sliced!");

            const cv::Mat data = getStimulusData(id);
            Stimulus fullStimulus(mStimuli[id]);

            for (unsigned int x = 0; x < (unsigned int)data.cols;
                 x += strideX) {
                for (unsigned int y = 0; y < (unsigned int)data.rows;
                     y += strideY) {
                    const RectangularROI<int>::Point_T tl(x, y);
                    const RectangularROI<int>::Point_T br(
                        std::min(x + width, (unsigned int)data.cols),
                        std::min(y + height, (unsigned int)data.rows));

                    Stimulus stimulus(fullStimulus);
                    stimulus.slice = new RectangularROI<int>(-1, tl, br);
                    stimulus.ROIs.clear();

                    for (std::vector<ROI*>::const_iterator itROIs
                         = fullStimulus.ROIs.begin(),
                         itROIsEnd = fullStimulus.ROIs.end();
                         itROIs != itROIsEnd;
                         ++itROIs) {
                        // Copy ROI
                        ROI* roi = (*itROIs)->clonePtr();

                        if (mStimuli[id].label == -1) {
                            // Composite stimuli
                            // Crop ROI
                            roi->padCrop(x, y, width, height);

                            // Check ROI overlaps with current slice
                            const cv::Rect roiRect = roi->getBoundingRect();

                            if (roiRect.tl().x > (int)width
                                || roiRect.tl().y > (int)height
                                || roiRect.br().x < 0 || roiRect.br().y < 0) {
                                // No overlap with current slice, discard ROI
                                delete roi;
                                continue;
                            }
                        }
                        // else
                            // Non-composite stimuli
                            // The ROI is extracted by getStimulusData()
                            // *before* slicing => DON'T CHANGE IT!

                        stimulus.ROIs.push_back(roi);
                    }

                    if (x == 0 && y == 0)
                        mStimuli[id] = stimulus;
                    else {
// Don't push_back to mStimuli directly, because if a reallocation occurs,
// concurrent readings (for
// example with getStimulusData()) will go wrong...
#pragma omp critical
                        newStimuli.push_back(stimulus);
                    }
                }
            }

            std::for_each(fullStimulus.ROIs.begin(),
                          fullStimulus.ROIs.end(),
                          Utils::Delete());

// Progress bar
#pragma omp critical
            {
                progress = (unsigned int)(20.0 * (++sliced) / (double)toSlice);

                if (progress > progressPrev) {
                    std::cout << std::string(progress - progressPrev, '.')
                              << std::flush;
                    progressPrev = progress;
                }
            }
        }

        // push_back new stimuli to mStimuli
        mStimuli.reserve(mStimuli.size() + newStimuli.size());
        mStimuliSets(*itSet)
            .reserve(mStimuliSets(*itSet).size() + newStimuli.size());

        for (std::vector<Stimulus>::const_iterator it = newStimuli.begin(),
                                                   itEnd = newStimuli.end();
             it != itEnd;
             ++it) {
            mStimuli.push_back(*it);
            mStimuliSets(*itSet).push_back(mStimuli.size() - 1);
        }
    }

    std::cout << std::endl;
}

void N2D2::Database::load(const std::string& /*dataPath*/,
          const std::string& labelPath,
          bool /*extractROIs*/)
{

    std::ifstream labelFile(labelPath.c_str());

    if (!labelFile.good())
        throw std::runtime_error("Could not open label file: "
                                 + labelPath);

    std::string labelName;

    while(std::getline( labelFile, labelName ))
        labelID(labelName);

    if(getNbLabels() == 0)
        throw std::runtime_error("No labels specified on the label file");

    std::cout << "Database loaded " << getNbLabels() << " labels from the file "
            << labelPath << std::endl;

}

void N2D2::Database::partitionStimuli(unsigned int nbStimuli, StimuliSet set)
{
    if (set == Unpartitioned)
        return;

    unsigned int maxStimuli = mStimuliSets(Unpartitioned).size();

    if (nbStimuli > maxStimuli)
        throw std::runtime_error("Database::partitionStimuli(): partition size "
                                 "larger than the number of available "
                                 "stimuli.");

    for (unsigned int i = 0; i < nbStimuli; ++i) {
        const unsigned int idx = (mRandomPartitioning)
            ? Random::randUniform(0, maxStimuli - 1) : 0;
        mStimuliSets(set).push_back(mStimuliSets(Unpartitioned)[idx]);
        mStimuliSets(Unpartitioned)
            .erase(mStimuliSets(Unpartitioned).begin() + idx);
        --maxStimuli;
    }

    assert(maxStimuli == mStimuliSets(Unpartitioned).size());
}

void
N2D2::Database::partitionStimuli(double learn, double validation, double test)
{
    if (learn + validation + test > 1.0)
        throw std::runtime_error("Database::partitionStimuli(): total "
                                 "partition ratio cannot be higher than 1.");

    unsigned int nbStimuli = mStimuliSets(Unpartitioned).size();

    const unsigned int nbLearn = Utils::round(nbStimuli * learn, Utils::HalfUp);
    const unsigned int nbValidationTest
        = Utils::round(nbStimuli * (validation + test), Utils::HalfDown);
    const double fracValidation
        = (validation > 0) ? (validation / (validation + test)) : 0.0;

    std::map<StimuliSet, unsigned int> partition;
    partition.insert(std::make_pair(Learn, nbLearn));
    partition.insert(std::make_pair(
        Validation,
        Utils::round(nbValidationTest * fracValidation, Utils::HalfUp)));
    partition.insert(
        std::make_pair(Test,
                       Utils::round(nbValidationTest * (1.0 - fracValidation),
                                    Utils::HalfDown)));

    for (std::map<StimuliSet, unsigned int>::const_iterator it
         = partition.begin(),
         itEnd = partition.end();
         it != itEnd;
         ++it)
        partitionStimuli((*it).second, (*it).first);
}

void N2D2::Database::partitionStimuliPerLabel(unsigned int nbStimuliPerLabel,
                                              StimuliSet set)
{
    if (set == Unpartitioned)
        return;

    // For each label, get the list of stimuli indexes in the unpartitioned set
    // with this label
    std::vector<std::vector<unsigned int> > labelsStimuliUnpartitionedIndexes
        = getLabelsStimuliSetIndexes(Unpartitioned);
    std::vector<unsigned int> partitionedIndexes;

    for (std::vector<std::vector<unsigned int> >::iterator it
         = labelsStimuliUnpartitionedIndexes.begin(),
         itEnd = labelsStimuliUnpartitionedIndexes.end();
         it != itEnd;
         ++it) {
        // Stimuli indexes in the unpartitioned set with this label
        std::vector<unsigned int>& unpartitionedIndexes = (*it);

        partitionIndexes(
            unpartitionedIndexes, partitionedIndexes, nbStimuliPerLabel, set);
    }

    if (partitionedIndexes.empty()) {
        std::cout << Utils::cwarning << "Warning: partitionStimuliPerLabel(): "
                                        "no stimulus were partitioned"
                  << Utils::cdef << std::endl;
    }

    removeIndexesFromSet(partitionedIndexes, Unpartitioned);
}

void N2D2::Database::partitionStimuliPerLabel(double learnPerLabel,
                                              double validationPerLabel,
                                              double testPerLabel,
                                              bool equiLabel)
{
    if (learnPerLabel + validationPerLabel + testPerLabel > 1.0)
        throw std::runtime_error("Database::partitionStimuliPerLabel(): total "
                                 "partition ratio cannot be higher than 1.");

    // For each label, get the list of stimuli indexes in the unpartitioned set
    // with this label
    std::vector<std::vector<unsigned int> > labelsStimuliUnpartitionedIndexes
        = getLabelsStimuliSetIndexes(Unpartitioned);

    unsigned int maxStimuliPerLabel = 0;

    if (equiLabel && !labelsStimuliUnpartitionedIndexes.empty()) {
        std::partial_sort(labelsStimuliUnpartitionedIndexes.begin(),
                          labelsStimuliUnpartitionedIndexes.begin() + 1,
                          labelsStimuliUnpartitionedIndexes.end(),
                          Utils::SizeCompare<std::vector<unsigned int> >());

        maxStimuliPerLabel
            = (*labelsStimuliUnpartitionedIndexes.begin()).size();
    }

    std::vector<unsigned int> partitionedIndexes;

    for (std::vector<std::vector<unsigned int> >::iterator it
         = labelsStimuliUnpartitionedIndexes.begin(),
         itEnd = labelsStimuliUnpartitionedIndexes.end();
         it != itEnd;
         ++it) {
        // Stimuli indexes in the unpartitioned set with this label
        std::vector<unsigned int>& unpartitionedIndexes = (*it);

        const unsigned int nbStimuli
            = (equiLabel) ? maxStimuliPerLabel : unpartitionedIndexes.size();

        const unsigned int nbLearn
            = Utils::round(nbStimuli * learnPerLabel, Utils::HalfUp);
        const unsigned int nbValidationTest = Utils::round(
            nbStimuli * (validationPerLabel + testPerLabel), Utils::HalfDown);
        const double fracValidation
            = (validationPerLabel > 0)
                  ? (validationPerLabel / (validationPerLabel + testPerLabel))
                  : 0.0;

        std::map<StimuliSet, unsigned int> partition;
        partition.insert(std::make_pair(Learn, nbLearn));
        partition.insert(std::make_pair(
            Validation,
            Utils::round(nbValidationTest * fracValidation, Utils::HalfUp)));
        partition.insert(std::make_pair(
            Test,
            Utils::round(nbValidationTest * (1.0 - fracValidation),
                         Utils::HalfDown)));

        for (std::map<StimuliSet, unsigned int>::const_iterator it
             = partition.begin(),
             itEnd = partition.end();
             it != itEnd;
             ++it)
            partitionIndexes(unpartitionedIndexes,
                             partitionedIndexes,
                             (*it).second,
                             (*it).first);
    }

    if (partitionedIndexes.empty()) {
        std::cout << Utils::cwarning << "Warning: partitionStimuliPerLabel(): "
                                        "no stimulus were partitioned"
                  << Utils::cdef << std::endl;
    }

    removeIndexesFromSet(partitionedIndexes, Unpartitioned);
}

void N2D2::Database::addStimulus(const std::string& name,
                                 const std::string& labelName,
                                 StimuliSet set)
{
    mStimuli.push_back(Stimulus(name, labelID(labelName)));
    mStimuliSets(set).push_back(mStimuli.size() - 1);
}

void
N2D2::Database::addStimulus(const std::string& name, int label, StimuliSet set)
{
    mStimuli.push_back(Stimulus(name, label));
    mStimuliSets(set).push_back(mStimuli.size() - 1);
}

void N2D2::Database::removeStimulus(StimulusID id)
{
    std::vector<StimuliSet> stimuliSets;
    stimuliSets.push_back(Learn);
    stimuliSets.push_back(Validation);
    stimuliSets.push_back(Test);
    stimuliSets.push_back(Unpartitioned);

    unsigned int nbFound = 0;

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(), itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
    {
        for (int idx = mStimuliSets(*it).size() - 1; idx >= 0; --idx)
        {
            if (mStimuliSets(*it)[idx] == (StimulusID)id) {
                mStimuliSets(*it)
                    .erase(mStimuliSets(*it).begin() + idx);
                ++nbFound;
            } else if (mStimuliSets(*it)[idx] > (StimulusID)id)
                --mStimuliSets(*it)[idx];
        }
    }

    if (nbFound == 0)
        throw std::runtime_error("Database::removeStimulus(): could not find "
                                 "the stimulus in any of the partition!");

    mStimuli.erase(mStimuli.begin() + id);
}

void N2D2::Database::removeLabel(int label)
{
    for (int id = mStimuli.size() - 1; id >= 0; --id) {
        if (mStimuli[id].label == label)
            removeStimulus(id);
        else if (mStimuli[id].label > label)
            --mStimuli[id].label;
    }

    mLabelsName.erase(mLabelsName.begin() + label);
}
/*
// NO TESTED
void N2D2::Database::mergeLabels(const std::vector<int>& labels, const
std::string& newName) {
    std::vector<int> sortedLabels(labels);
    std::sort(sortedLabels.begin(), sortedLabels.end());
    std::vector<int>::iterator itLabelBegin = sortedLabels.begin();

    for (std::vector<Stimulus>::iterator it = mStimuli.begin(), itEnd =
mStimuli.end(); it != itEnd; ++it) {
        std::vector<int>::const_iterator itLabel =
std::lower_bound(itLabelBegin, sortedLabels.end(), (*it).label);

        if (*itLabel == (*it).label)
            (*it).label = sortedLabels[0];
        else
            (*it).label-= itLabel - itLabelBegin;
    }

    std::string mergedName(newName);

    if (mergedName.empty()) {
        // Concatenate label names for the new name for the merged label
        for (std::vector<int>::const_iterator it = itLabelBegin, itEnd =
sortedLabels.end(); it != itEnd; ++it) {
            if (it != itLabelBegin)
                mergedName+= ";";

            mergedName+= mLabelsName[*it];
        }
    }

    // Merged label name
    mLabelsName[sortedLabels[0]] = mergedName;

    // Remove labels that where merged
    for (int label = sortedLabels.size() - 1; label > 0; --label)
        mLabelsName.erase(mLabelsName.begin() + label);
}
*/

std::string N2D2::Database::getStimulusName(StimulusID id) const
{
    if (mStimuli[id].slice != NULL) {
        const cv::Rect bbRect = mStimuli[id].slice->getBoundingRect();

        std::stringstream nameStr;
        nameStr << mStimuli[id].name << "[" << bbRect.y << "," << bbRect.x
                << "]";

        return nameStr.str();
    } else
        return mStimuli[id].name;
}

std::vector<std::shared_ptr<N2D2::ROI> >
N2D2::Database::getStimulusROIs(StimulusID id) const
{
    std::vector<std::shared_ptr<ROI> > stimulusROIs;
    std::transform(mStimuli[id].ROIs.begin(),
                   mStimuli[id].ROIs.end(),
                   std::back_inserter(stimulusROIs),
                   std::bind(&ROI::clone, std::placeholders::_1));

    if (mStimuli[id].label >= 0 && !stimulusROIs.empty()) {
        if (stimulusROIs.size() != 1) {
            throw std::runtime_error("Database::getStimulusROIs(): "
                                     "number of ROIs should be 1 for "
                                     "non-composite stimuli");
        }

        // Align ROI to extracted data
        const cv::Rect roiOrg = stimulusROIs[0]->getBoundingRect();
        stimulusROIs[0]->padCrop(roiOrg.tl().x,
                                 roiOrg.tl().y,
                                 roiOrg.width,
                                 roiOrg.height);

        if (mStimuli[id].slice != NULL) {
            // Align ROI to slice
            const cv::Rect sliceRect = mStimuli[id].slice->getBoundingRect();

            stimulusROIs[0]->padCrop(sliceRect.tl().x,
                                     sliceRect.tl().y,
                                     sliceRect.width,
                                     sliceRect.height);
        }
    }

    return stimulusROIs;
}

unsigned int N2D2::Database::getNbROIs() const
{
    unsigned int nbROIs = 0;

    for (std::vector<Stimulus>::const_iterator it = mStimuli.begin(),
                                               itEnd = mStimuli.end();
         it != itEnd;
         ++it)
        nbROIs += (*it).ROIs.size();

    return nbROIs;
}

unsigned int N2D2::Database::getNbROIsWithLabel(int label) const
{
    unsigned int nbROIs = 0;

    for (std::vector<Stimulus>::const_iterator it = mStimuli.begin(),
                                               itEnd = mStimuli.end();
         it != itEnd;
         ++it) {
        for (std::vector<ROI*>::const_iterator itROIs = (*it).ROIs.begin(),
                                               itROIsEnd = (*it).ROIs.end();
             itROIs != itROIsEnd;
             ++itROIs) {
            if ((*itROIs)->getLabel() == label)
                ++nbROIs;
        }
    }

    return nbROIs;
}

bool N2D2::Database::isLabel(const std::string& labelName) const
{
    return (std::find(mLabelsName.begin(), mLabelsName.end(), labelName)
            != mLabelsName.end());
}

int N2D2::Database::getLabelID(const std::string& labelName) const
{
    std::vector<std::string>::const_iterator it
        = std::find(mLabelsName.begin(), mLabelsName.end(), labelName);

    if (it == mLabelsName.end())
        throw std::runtime_error(
            "Database::getLabelID(): no label with the name \"" + labelName
            + "\"");

    return (it - mLabelsName.begin());
}

int N2D2::Database::getDefaultLabelID() const {
    return (!((std::string)mDefaultLabel).empty())
        ? getLabelID(mDefaultLabel)
        : -1;
}

cv::Mat N2D2::Database::getStimulusData(StimulusID id)
{
    if (mLoadDataInMemory) {
        if (mStimuliData.empty()) {
#pragma omp critical
            mStimuliData.resize(mStimuli.size());
        }

        if (mStimuliData[id].empty())
            mStimuliData[id] = loadStimulusData(id);

        return mStimuliData[id];
    } else
        return loadStimulusData(id);
}

cv::Mat N2D2::Database::getStimulusLabelsData(StimulusID id)
{
    if (mLoadDataInMemory) {
        if (mStimuliLabelsData.empty()) {
#pragma omp critical
            mStimuliLabelsData.resize(mStimuli.size());
        }

        if (mStimuliLabelsData[id].empty())
            mStimuliLabelsData[id] = loadStimulusLabelsData(id);

        return mStimuliLabelsData[id];
    } else
        return loadStimulusLabelsData(id);
}

std::vector<N2D2::Database::StimuliSet>
N2D2::Database::getStimuliSets(StimuliSetMask setMask) const
{
    std::vector<StimuliSet> stimuliSets;

    if (setMask != NoLearn && setMask != ValidationOnly && setMask != TestOnly)
        stimuliSets.push_back(Learn);

    if (setMask != NoValidation && setMask != LearnOnly && setMask != TestOnly)
        stimuliSets.push_back(Validation);

    if (setMask != NoTest && setMask != LearnOnly && setMask != ValidationOnly)
        stimuliSets.push_back(Test);

    return stimuliSets;
}

N2D2::Database::StimuliSetMask
N2D2::Database::getStimuliSetMask(StimuliSet set) const
{
    return  (set == Learn)      ?   LearnOnly :
            (set == Validation) ?   ValidationOnly :
                                    TestOnly;
}

std::map<std::string, N2D2::Database::StimulusID>
N2D2::Database::getRelPathStimuli(const std::string& fileName,
                                  const std::string& relPath)
{
    // Find all stimuli within the relPath path
    std::map<std::string, StimulusID> stimuliName;

    for (StimulusID id = 0, size = mStimuli.size(); id < size; ++id) {
        bool newInsert = true;

        if (relPath.empty())
            std::tie(std::ignore, newInsert) = stimuliName.insert(
                std::make_pair(Utils::baseName(mStimuli[id].name), id));
        else if (mStimuli[id].name.compare(0, relPath.size(), relPath) == 0)
            std::tie(std::ignore, newInsert)
                = stimuliName.insert(std::make_pair(
                    mStimuli[id].name.substr(relPath.size() + 1), id));

        if (!newInsert)
            throw std::runtime_error("Database::loadROIs(): cannot "
                                     "differentiate some stimuli for the ROI "
                                     "data file: " + fileName);
    }

    return stimuliName;
}

int N2D2::Database::labelID(const std::string& labelName)
{
    const std::vector<std::string>::iterator itBegin = mLabelsName.begin();
    std::vector<std::string>::const_iterator it
        = std::find(itBegin, mLabelsName.end(), labelName);

    if (it != mLabelsName.end())
        return (it - itBegin);
    else {
        mLabelsName.push_back(labelName);
        return (mLabelsName.size() - 1);
    }
}

cv::Mat N2D2::Database::loadStimulusData(StimulusID id)
{
    // Initialize mStimuliDepth using the first stimulus
    if (mStimuliDepth == -1) {
        std::string fileExtension = Utils::fileExtension(mStimuli[0].name);
        std::transform(fileExtension.begin(),
                       fileExtension.end(),
                       fileExtension.begin(),
                       ::tolower);

        std::shared_ptr<DataFile> dataFile = Registrar
            <DataFile>::create(fileExtension)();
        mStimuliDepth = dataFile->read(mStimuli[0].name).depth();

        std::cout << Utils::cnotice << "Notice: stimuli depth is "
                  << Utils::cvMatDepthToString(mStimuliDepth)
                  << " (according to database first stimulus)" << Utils::cdef
                  << std::endl;
    }

    std::string fileExtension = Utils::fileExtension(mStimuli[id].name);
    std::transform(fileExtension.begin(),
                   fileExtension.end(),
                   fileExtension.begin(),
                   ::tolower);

    std::shared_ptr<DataFile> dataFile = Registrar
        <DataFile>::create(fileExtension)();
    cv::Mat data = dataFile->read(mStimuli[id].name);

    // Check stimulus depth
    if (data.depth() != mStimuliDepth) {
        std::cout << Utils::cnotice << "Notice: converting depth from "
                  << Utils::cvMatDepthToString(data.depth()) << " to "
                  << Utils::cvMatDepthToString(mStimuliDepth)
                  << " for stimulus: " << mStimuli[id].name << Utils::cdef
                  << std::endl;

        cv::Mat dataConverted;
        data.convertTo(dataConverted,
                       mStimuliDepth,
                       Utils::cvMatDepthUnityValue(mStimuliDepth)
                       / Utils::cvMatDepthUnityValue(data.depth()));
        data = dataConverted;
    }

    if (mStimuli[id].label >= 0 && !mStimuli[id].ROIs.empty()) {
        // Non-composite stimulus with ROI
        data = mStimuli[id].ROIs[0]->extract(data);
    }

    if (mStimuli[id].slice != NULL)
        data = mStimuli[id].slice->extract(data);

    return data;
}

cv::Mat N2D2::Database::loadStimulusLabelsData(StimulusID id) const
{
    if (mStimuli[id].label == -1) {
        // Composite stimulus
        // Construct the labels matrix with the ROIs
        const cv::Mat stimulus
            = cv::imread(mStimuli[id].name, CV_LOAD_IMAGE_UNCHANGED);

        if (!stimulus.data)
            throw std::runtime_error(
                "Database::loadStimulusLabelsData(): unable to read image: "
                + mStimuli[id].name);

        const int defaultLabel = getDefaultLabelID();
        cv::Mat labels(
            stimulus.rows, stimulus.cols, CV_32SC1, cv::Scalar(defaultLabel));

        if (mStimuli[id].slice != NULL)
            labels = mStimuli[id].slice->extract(labels);

        for (std::vector<ROI*>::const_iterator it = mStimuli[id].ROIs.begin(),
                                               itEnd = mStimuli[id].ROIs.end();
             it != itEnd;
             ++it)
            (*it)->append(labels, mROIsMargin, defaultLabel);

        return labels;
    } else {
        // Non-composite stimulus
        if (!mStimuli[id].ROIs.empty()) {
            if (mStimuli[id].ROIs.size() != 1) {
                throw std::runtime_error("Database::loadStimulusLabelsData(): "
                                         "number of ROIs should be 1 for "
                                         "non-composite"
                                         " stimuli");
            }

            return cv::Mat(
                1, 1, CV_32SC1, cv::Scalar(mStimuli[id].ROIs[0]->getLabel()));
        } else
            return cv::Mat(1, 1, CV_32SC1, cv::Scalar(mStimuli[id].label));
    }
}

std::vector<unsigned int>
N2D2::Database::getLabelStimuliSetIndexes(int label, StimuliSet set) const
{
    std::vector<unsigned int> labelStimuli;

    for (std::vector<StimulusID>::const_iterator it
         = mStimuliSets(set).begin(),
         itBegin = mStimuliSets(set).begin(),
         itEnd = mStimuliSets(set).end();
         it != itEnd;
         ++it) {
        if (mStimuli[(*it)].label == label)
            labelStimuli.push_back(it - itBegin);
    }

    return labelStimuli;
}

std::vector<std::vector<unsigned int> >
N2D2::Database::getLabelsStimuliSetIndexes(StimuliSet set) const
{
    const unsigned int nbLabels = mLabelsName.size();
    std::vector<std::vector<unsigned int> > labelsStimuli(
        nbLabels, std::vector<unsigned int>());

    for (std::vector<StimulusID>::const_iterator it
         = mStimuliSets(set).begin(),
         itBegin = mStimuliSets(set).begin(),
         itEnd = mStimuliSets(set).end();
         it != itEnd;
         ++it) {
        // For each stimulus of the set, get its label
        if (mStimuli[(*it)].label >= 0)
            labelsStimuli[mStimuli[(*it)].label].push_back(it - itBegin);
    }

    return labelsStimuli;
}

void N2D2::Database::partitionIndexes(std::vector
                                      <unsigned int>& unpartitionedIndexes,
                                      std::vector
                                      <unsigned int>& partitionedIndexes,
                                      unsigned int nbStimuli,
                                      StimuliSet set)
{
    unsigned int maxStimuli = unpartitionedIndexes.size();

    if (nbStimuli > maxStimuli)
        throw std::runtime_error("Database::partitionIndexes(): partition size "
                                 "larger than the number of available "
                                 "stimuli.");

    for (unsigned int i = 0; i < nbStimuli; ++i) {
        const unsigned int idx = (mRandomPartitioning)
            ? Random::randUniform(0, maxStimuli - 1) : 0;
        const unsigned int unpartitionedIdx = unpartitionedIndexes[idx];
        const StimulusID id = mStimuliSets(Unpartitioned)[unpartitionedIdx];

        unpartitionedIndexes.erase(unpartitionedIndexes.begin() + idx);
        partitionedIndexes.push_back(unpartitionedIdx);
        mStimuliSets(set).push_back(id);
        --maxStimuli;
    }
}

void N2D2::Database::removeIndexesFromSet(std::vector<unsigned int>& indexes,
                                          StimuliSet set)
{
    // Sort the indexes and then delete those elements from the vector from the
    // highest to the lowest.
    // That way, deleting the highest index on the list will not invalidate the
    // lower indices remaining to be deleted, because only
    // the elements higher than the deleted ones change their index.
    std::sort(indexes.begin(), indexes.end());

    for (int idx = indexes.size() - 1; idx >= 0; --idx)
        mStimuliSets(set).erase(mStimuliSets(set).begin() + indexes[idx]);

    indexes.clear();
}

N2D2::Database::~Database()
{
    for (std::vector<Stimulus>::iterator it = mStimuli.begin(),
                                         itEnd = mStimuli.end();
         it != itEnd;
         ++it)
        std::for_each((*it).ROIs.begin(), (*it).ROIs.end(), Utils::Delete());
}
