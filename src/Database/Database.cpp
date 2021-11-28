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

#include "LabelFile/LabelFile.hpp"
#include "LabelFile/CsvLabelFile.hpp"
#include "Database/Database.hpp"
#include "ROI/RectangularROI.hpp"
#include "utils/Gnuplot.hpp"

#include <regex>

const std::locale
N2D2::Database::csvLocale(std::locale(),
                          new N2D2::Utils::streamIgnore(",; \t"));

N2D2::Database::Database(bool loadDataInMemory)
    : mDefaultLabel(this, "DefaultLabel", ""),
      mROIsMargin(this, "ROIsMargin", 0U),
      mRandomPartitioning(this, "RandomPartitioning", true),
      mDataFileLabel(this, "DataFileLabel", true),
      mCompositeLabel(this, "CompositeLabel", Auto),
      mTargetDataPath(this, "TargetDataPath", ""),
      mMultiChannelMatch(this, "MultiChannelMatch", ""),
      mMultiChannelReplace(this, "MultiChannelReplace",
                           std::vector<std::string>()),
      mLoadDataInMemory(loadDataInMemory),
      mStimuliDepth(-1),
      mStimuliTargetDepth(-1)
{
    // ctor
}

bool N2D2::Database::empty() const {
    return mStimuli.empty();
}

void N2D2::Database::loadROIs(const std::string& fileName,
                              const std::string& relPath,
                              bool noImageSize)
{
    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    // Find all stimuli within the relPath path
    const std::map<std::string, StimulusID> stimuliName
        = getRelPathStimuli(fileName, relPath);

    // Read label
    std::string fileExtension = Utils::fileExtension(fileName);
    std::transform(fileExtension.begin(),
                    fileExtension.end(),
                    fileExtension.begin(),
                    ::tolower);

    std::shared_ptr<LabelFile> labelFile = Registrar
        <LabelFile>::create(fileExtension)();

    if (std::shared_ptr<CsvLabelFile> csvLabelFile
        = std::dynamic_pointer_cast<CsvLabelFile>(labelFile))
    {
        csvLabelFile->noImageSize(noImageSize);
    }

    const std::map<std::string, std::vector<ROI*> >& ROIs
        = labelFile->read(fileName,
                std::bind(&Database::labelID, this, std::placeholders::_1));

    const std::string& multiChannelMatch
        = getParameter<std::string>("MultiChannelMatch");
    std::map<std::string, std::string> matchChannel;

    if (!multiChannelMatch.empty()) {
        for (std::map<std::string, StimulusID>::const_iterator it
            = stimuliName.begin(), itEnd = stimuliName.end(); it != itEnd; ++it)
        {
            const std::regex regexp(multiChannelMatch);

            if (!std::regex_match((*it).first, regexp))
                continue;

            const std::vector<std::string>& multiChannelReplace
                = getParameter<std::vector<std::string> >
                    ("MultiChannelReplace");

            for (size_t ch = 0; ch < multiChannelReplace.size(); ++ch) {
                const std::string chFilePath
                    = std::regex_replace((*it).first, regexp,
                                        multiChannelReplace[ch]);

                matchChannel[chFilePath] = (*it).first;
            }
        }
    }

    for (std::map<std::string, std::vector<ROI*> >::const_iterator
        it = ROIs.begin(), itEnd = ROIs.end(); it != itEnd; ++it)
    {
        const std::map<std::string, std::string>::const_iterator itMatchCh
            = matchChannel.find((*it).first);
        const std::string name = (itMatchCh != matchChannel.end())
            ? (*itMatchCh).first : (*it).first;

        // Find corresponding stimulus
        std::map<std::string, StimulusID>::const_iterator itStimulus
            = stimuliName.find(name);

        // Support for "double extension" for label files, like .box.json
        if (itStimulus == stimuliName.end())
            itStimulus = stimuliName.find(Utils::fileBaseName(name));

        if (itStimulus != stimuliName.end()) {
            std::vector<ROI*>& stimulusROIs
                = mStimuli[(*itStimulus).second].ROIs;

            stimulusROIs.insert(stimulusROIs.end(),
                (*it).second.begin(), (*it).second.end());
        }
        else {
            std::cout << Utils::cwarning
                        << "Warning: ignoring ROI for non-existant stimulus: \""
                        << (*it).first << "\"" << Utils::cdef << std::endl;
        }
    }
}

void N2D2::Database::loadROIsDir(const std::string& dirName,
                                 const std::vector<std::string>& fileExt,
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
            if (!fileExt.empty() && std::find(fileExt.begin(), fileExt.end(),
                Utils::fileExtension(fileName)) == fileExt.end())
            {
                continue;
            }

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

void N2D2::Database::logPartition(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(All);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        const unsigned int size = mStimuliSets(*itSet).size();
        std::ostringstream fileNameStr;
        fileNameStr << dirName << "/" << (*itSet) << ".log";

        std::ofstream stats(fileNameStr.str().c_str());

        if (!stats.good())
            throw std::runtime_error("Could not create partition file: " + fileNameStr.str());

        // Append date & time to the file.
        const time_t now = std::time(0);
        tm* localNow = std::localtime(&now);

        stats << "# " << std::asctime(localNow); // std::asctime() already
        // appends end of line

        for (int i = 0; i < (int)size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            stats << mStimuli[id].name << "\n";
        }
    }
}

void N2D2::Database::logStats(const std::string& sizeFileName,
                              const std::string& labelFileName,
                              StimuliSetMask setMask) const
{
    // Stats collection
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> sizeStats;
    unsigned int minWidth = std::numeric_limits<int>::max();
    unsigned int minHeight = std::numeric_limits<int>::max();
    unsigned int maxWidth = 0;
    unsigned int maxHeight = 0;
    unsigned int nbStimuli = 0;
    std::map<int, unsigned int> labelStats;

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        const unsigned int size = mStimuliSets(*itSet).size();

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)size; ++i){
            const StimulusID id = mStimuliSets(*itSet)[i];

            // Read stimuli
            std::string fileExtension = Utils::fileExtension(mStimuli[id].name);
            std::transform(fileExtension.begin(),
                           fileExtension.end(),
                           fileExtension.begin(),
                           ::tolower);

            std::shared_ptr<DataFile> dataFile = Registrar
                <DataFile>::create(fileExtension)();
            cv::Mat stimulus = dataFile->read(mStimuli[id].name);

            // Stats
            if (stimulus.cols > (int)maxWidth) {
#pragma omp critical(logStats_maxWidth)
                if (stimulus.cols > (int)maxWidth)
                    maxWidth = stimulus.cols;
            }

            if (stimulus.rows > (int)maxHeight) {
#pragma omp critical(logStats_maxHeight)
                if (stimulus.rows > (int)maxHeight)
                    maxHeight = stimulus.rows;
            }

            if (stimulus.cols < (int)minWidth) {
#pragma omp critical(logStats_minWidth)
                if (stimulus.cols < (int)minWidth)
                    minWidth = stimulus.cols;
            }

            if (stimulus.rows < (int)minHeight) {
#pragma omp critical(logStats_minHeight)
                if (stimulus.rows < (int)minHeight)
                    minHeight = stimulus.rows;
            }

            const std::pair<std::pair<unsigned int, unsigned int>, unsigned int>
                sizeStat = std::make_pair(std::make_pair(stimulus.cols,
                                                         stimulus.rows), 0U);

            const std::pair<int, unsigned int> labelStat
                = std::make_pair(mStimuli[id].label, 0U);

            std::map<std::pair<unsigned int, unsigned int>,
                     unsigned int>::iterator itSizeStats;
            std::map<int, unsigned int>::iterator itLabelStats;

#pragma omp critical(logStats)
            {
                std::tie(itSizeStats, std::ignore) = sizeStats.insert(sizeStat);
                ++(*itSizeStats).second;

                std::tie(itLabelStats, std::ignore)
                    = labelStats.insert(labelStat);
                ++(*itLabelStats).second;
            }
        }

        nbStimuli += size;
    }

    if (nbStimuli > 0) {
        plotStats(sizeFileName, labelFileName, nbStimuli,
                  minWidth, maxWidth, minHeight, maxHeight,
                  sizeStats, labelStats);
    }
    else
        std::cout << "Database::logStats(): no stimulus" << std::endl;
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
         ++itSet)
    {
        const unsigned int size = mStimuliSets(*itSet).size();

//#pragma omp parallel for schedule(dynamic) reduction(+:nbROIs)
        for (int i = 0; i < (int)size; ++i)
        {
            const StimulusID id = mStimuliSets(*itSet)[i];

            unsigned int bbMinWidth = std::numeric_limits<int>::max();
            unsigned int bbMinHeight = std::numeric_limits<int>::max();
            unsigned int bbMaxWidth = 0;
            unsigned int bbMaxHeight = 0;

            for (std::vector<ROI*>::const_iterator itROIs
                 = mStimuli[id].ROIs.begin(),
                 itROIsEnd = mStimuli[id].ROIs.end();
                 itROIs != itROIsEnd;
                 ++itROIs) {
                const cv::Rect bb = (*itROIs)->getBoundingRect();

                if (bb.width > (int)bbMaxWidth)
                    bbMaxWidth = bb.width;
                if (bb.height > (int)bbMaxHeight)
                    bbMaxHeight = bb.height;
                if (bb.width < (int)bbMinWidth)
                    bbMinWidth = bb.width;
                if (bb.height < (int)bbMinHeight)
                    bbMinHeight = bb.height;

                const std::pair<std::pair<unsigned int, unsigned int>,
                                unsigned int> sizeStat
                    = std::make_pair(std::make_pair(bb.width, bb.height), 0U);

                const std::pair<int, unsigned int> labelStat
                    = std::make_pair((*itROIs)->getLabel(), 0U);

                std::map<std::pair<unsigned int, unsigned int>,
                         unsigned int>::iterator itSizeStats;
                std::map<int, unsigned int>::iterator itLabelStats;

//#pragma omp critical(logROIsStats)
                {
                    std::tie(itSizeStats, std::ignore)
                        = sizeStats.insert(sizeStat);
                    ++(*itSizeStats).second;

                    std::tie(itLabelStats, std::ignore)
                        = labelStats.insert(labelStat);
                    ++(*itLabelStats).second;
                }
            }

            if (bbMaxWidth > maxWidth) {
//#pragma omp critical(logROIsStats_maxWidth)
//                if (bbMaxWidth > maxWidth)
                    maxWidth = bbMaxWidth;
            }

            if (bbMaxHeight > maxHeight) {
//#pragma omp critical(logROIsStats_maxHeight)
//                if (bbMaxHeight > maxHeight)
                    maxHeight = bbMaxHeight;
            }

            if (bbMinWidth < minWidth) {
//#pragma omp critical(logROIsStats_minWidth)
//                if (bbMinWidth < minWidth)
                    minWidth = bbMinWidth;
            }

            if (bbMinHeight < minHeight) {
//#pragma omp critical(logROIsStats_minHeight)
//                if (bbMinHeight < minHeight)
                    minHeight = bbMinHeight;
            }

            nbROIs += mStimuli[id].ROIs.size();
        }
    }

    if (nbROIs > 0) {
        plotStats(sizeFileName, labelFileName, nbROIs,
                  minWidth, maxWidth, minHeight, maxHeight,
                  sizeStats, labelStats);
    }
    else
        std::cout << "Database::logROIsStats(): no ROI" << std::endl;
}

void N2D2::Database::logMultiChannelStats(const std::string& fileName,
                                          StimuliSetMask setMask) const
{
    if (((std::string)mMultiChannelMatch).empty()) {
        std::cout << Utils::cwarning << "Database::logMultiChannelStats(): "
            "no multi-channel, MultiChannelMatch is empty!" << Utils::cdef
            << std::endl;
    }

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);
    const std::regex regexp((std::string)mMultiChannelMatch);
    const std::vector<std::string>& multiChannelReplace
        = mMultiChannelReplace.get<std::vector<std::string> >();

    unsigned int nbStimuli = 0;
    unsigned int nbMatchMulti = 0;
    std::vector<unsigned int> nbMatch(multiChannelReplace.size(), 0);
    std::vector<unsigned int> nbValid(multiChannelReplace.size(), 0);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        const unsigned int size = mStimuliSets(*itSet).size();

        nbStimuli += size;

#pragma omp parallel for schedule(dynamic) reduction(+:nbMatchMulti)
        for (int i = 0; i < (int)size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            std::string fileExtension = Utils::fileExtension(mStimuli[id].name);
            std::transform(fileExtension.begin(),
                        fileExtension.end(),
                        fileExtension.begin(),
                        ::tolower);

            std::shared_ptr<DataFile> dataFile = Registrar
                <DataFile>::create(fileExtension)();
            cv::Mat data = dataFile->read(mStimuli[id].name);

            if (std::regex_match(mStimuli[id].name, regexp)) {
                ++nbMatchMulti;

                for (size_t ch = 0; ch < multiChannelReplace.size(); ++ch) {
                    const std::string chFileName
                        = std::regex_replace(mStimuli[id].name,
                                             regexp, multiChannelReplace[ch]);
                    cv::Mat chData;

                    if (chFileName == mStimuli[id].name) {
                        #pragma omp atomic
                        ++nbMatch[ch];
                        chData = data;
                    }
                    else if (std::ifstream(chFileName).good()) {
                        #pragma omp atomic
                        ++nbMatch[ch];
                        chData = dataFile->read(chFileName);
                    }

                    if (!chData.empty()
                        && chData.size() == data.size()
                        && chData.depth() == data.depth())
                    {
                        #pragma omp atomic
                        ++nbValid[ch];
                    }
                }
            }
        }
    }

    // Save log file
    std::ofstream stats(fileName.c_str());

    if (!stats.good())
        throw std::runtime_error("Could not create stats file: " + fileName);

    // Append date & time to the file.
    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);

    stats << "# " << std::asctime(localNow); // std::asctime() already
    // appends end of line

    stats << "all " << nbStimuli << " " << nbStimuli << "\n";
    stats << "multi " << nbMatchMulti << " " << nbMatchMulti << "\n";

    for (size_t ch = 0; ch < multiChannelReplace.size(); ++ch) {
        stats << ch << " " << nbMatch[ch] << " " << nbValid[ch] << "\n";
    }

    // Plot stats
    Gnuplot gnuplot;
    gnuplot << "wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:"
                    "maxLength].\"\\n\".wrap(str[maxLength+1:],maxLength)";
    gnuplot.set("style histogram cluster gap 1");
    gnuplot.set("style data histograms");
    gnuplot.set("style fill pattern 1.00 border");
    gnuplot.set("ytics nomirror");
    gnuplot.setYlabel("Valid channels");
    gnuplot.set("grid");
    gnuplot.set("xtics rotate by 90 right");
    gnuplot.set("ytics textcolor lt 1");
    gnuplot.set("bmargin 10");
    gnuplot.set("yrange [0:]");
    gnuplot.unset("key");
    gnuplot.saveToFile(fileName);
    gnuplot.plot(
        fileName,
        "using ($2):xticlabels(wrap(stringcolumn(1),10)) lt 1,"
        " '' using 0:($2):($2) with labels offset char 0,1 textcolor lt 1");
}

void N2D2::Database::plotStats(
    const std::string& sizeFileName,
    const std::string& labelFileName,
    unsigned int totalCount,
    unsigned int minWidth,
    unsigned int maxWidth,
    unsigned int minHeight,
    unsigned int maxHeight,
    const std::map<std::pair<unsigned int, unsigned int>,
                                                    unsigned int>& sizeStats,
    const std::map<int, unsigned int>& labelStats) const
{
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
    labelStr << "Width (range = [" << minWidth << ", " << maxWidth << "])";
    sizeGnuplot.setXlabel(labelStr.str());

    labelStr.str(std::string());
    labelStr << "Height (range = [" << minHeight << ", " << maxHeight
             << "])";
    sizeGnuplot.setYlabel(labelStr.str());

    labelStr.str(std::string());
    labelStr << "title \"Total number = " << totalCount << "\"";
    sizeGnuplot.set(labelStr.str());

    sizeGnuplot.saveToFile(sizeFileName);
    sizeGnuplot.splot(
        sizeFileName,
        "using 1:2:3 with points pointtype 5 pointsize 1 palette linewidth 2");

    sizeGnuplot << "reset";
    sizeGnuplot.set("grid");
    sizeGnuplot.unset("key");
    sizeGnuplot.setYlabel("Number of stimuli (cumulative)");
    sizeGnuplot.setXlabel("Width (pixels)");
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
    sizeGnuplot.setXlabel("Height (pixels)");
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
        const std::string labelName = ((*it).first != -1)
            ? getLabelName((*it).first) : "-1";

        labelData << "\"" << labelName << "\" "
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
    labelGnuplot.setYlabel("Label occurrence(s)");
    labelGnuplot.set("grid");
    labelGnuplot.set("xtics rotate by 90 right");
    labelGnuplot.set("ytics textcolor lt 1");

    std::stringstream yLabelStr;
    yLabelStr << "ylabel \"Label name (total number of labels = "
        << labelStats.size() << ")\" textcolor lt 1";

    labelGnuplot.set(yLabelStr.str());
    labelGnuplot.set("bmargin 10");
    labelGnuplot.set("yrange [0:]");
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
                                   StimuliSetMask setMask,
                                   bool randomShuffle,
                                   bool overlapping)
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

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            if (mStimuli[id].slice != NULL) {
#pragma omp critical
                throw std::runtime_error("Database::extractSlices(): some "
                                         "stimuli are already sliced!");
            }

            const cv::Mat data = getStimulusData(id);
            Stimulus fullStimulus(mStimuli[id]);

            for (unsigned int x = 0; x < (unsigned int)data.cols;
                 x += strideX) {
                for (unsigned int y = 0; y < (unsigned int)data.rows;
                     y += strideY) {
                    RectangularROI<int>::Point_T tl, br;

                    if (overlapping) {
                        tl = RectangularROI<int>::Point_T(
                            std::min((int)x, std::max(0, data.cols - (int)width)),
                            std::min((int)y, std::max(0, data.rows - (int)height)));
                        // br is exclusive, as required by RectangularROI<>()
                        br = RectangularROI<int>::Point_T(
                            std::min(tl.x + width, (unsigned int)data.cols),
                            std::min(tl.y + height, (unsigned int)data.rows));
                    }
                    else {
                        tl = RectangularROI<int>::Point_T(x, y);
                        br = RectangularROI<int>::Point_T(
                            std::min(x + width, (unsigned int)data.cols),
                            std::min(y + height, (unsigned int)data.rows));
                    }

                    const int offsetX = x - tl.x; // = 0 without overlapping
                    const int offsetY = y - tl.y; // = 0 without overlapping

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

                        if (mCompositeLabel != Auto
                            || mStimuli[id].label == -1)
                        {
                            // Composite stimuli
                            // Crop ROI
                            roi->padCrop(x, y, width, height);

                            // Check ROI overlaps with current slice
                            const cv::Rect roiRect = roi->getBoundingRect();

                            if (roiRect.tl().x > (int)width
                                || roiRect.tl().y > (int)height
                                || roiRect.br().x < offsetX
                                || roiRect.br().y < offsetY) {
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

                    if (overlapping) {
                        // Make overlapping area ignored
                        if (offsetX > 0) {
                            stimulus.ROIs.push_back(new RectangularROI<int>(-1,
                                RectangularROI<int>::Point_T(0, 0),
                                RectangularROI<int>::Point_T(offsetX, height)));
                        }

                        if (offsetY > 0) {
                            stimulus.ROIs.push_back(new RectangularROI<int>(-1,
                                RectangularROI<int>::Point_T(0, 0),
                                RectangularROI<int>::Point_T(width, offsetY)));
                        }
                    }

                    if (x == 0 && y == 0)
                        mStimuli[id] = stimulus;
                    else {
// Don't push_back to mStimuli directly, because if a reallocation occurs,
// concurrent readings (for
// example with getStimulusData()) will go wrong...
#pragma omp critical(Database__extractSlices_1)
                        newStimuli.push_back(stimulus);
                    }
                }
            }

            std::for_each(fullStimulus.ROIs.begin(),
                          fullStimulus.ROIs.end(),
                          Utils::Delete());

// Progress bar
            progress = (unsigned int)(20.0 * (sliced + i) / (double)toSlice);

            if (progress > progressPrev) {
#pragma omp critical(Database__extractSlices_2)
                if (progress > progressPrev) {
                    std::cout << std::string(progress - progressPrev, '.')
                              << std::flush;
                    progressPrev = progress;
                }
            }
        }

        sliced += size;

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

        if (randomShuffle) {
            std::random_shuffle(mStimuliSets(*itSet).begin(),
                                mStimuliSets(*itSet).end(),
                                Random::randShuffle);
        }
    }

    std::cout << std::endl;

    assert(sliced == toSlice);
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

void N2D2::Database::save(const std::string& dataPath,
                          StimuliSetMask setMask,
                          CompositeTransformation trans,
                          bool subDirPerClass)
{
    Utils::createDirectories(dataPath);

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    // For progression visualization
    unsigned int toSave = 0;

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(),
         itEnd = stimuliSets.end();
         it != itEnd;
         ++it) {
        toSave += getNbStimuli(*it);
    }

    std::cout << "Database: saving " << toSave << " stimuli to: "
        << dataPath << std::flush;

    unsigned int saved = 0;
    unsigned int progress = 0, progressPrev = 0;

    for (std::vector<StimuliSet>::const_iterator itSet = stimuliSets.begin(),
                                                 itSetEnd = stimuliSets.end();
         itSet != itSetEnd; ++itSet)
    {
        const int size = mStimuliSets(*itSet).size();

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];
            const int labelID = getStimulusLabel(id);

            std::ostringstream fileName;
            fileName << dataPath;

            if (subDirPerClass && labelID >= 0) {
                fileName << "/" << getLabelName(labelID);
                Utils::createDirectories(fileName.str());
            }

            fileName << "/" << Utils::baseName(getStimulusName(id));

            cv::Mat data = getStimulusData(id).clone(); // make sure the
            // database image will not be altered
            trans.apply(data);

            if (!cv::imwrite(fileName.str(), data)) {
#pragma omp critical
                throw std::runtime_error("Unable to write image: "
                                         + fileName.str());
            }

// Progress bar
            progress = (unsigned int)(20.0 * (saved + i) / (double)toSave);

            if (progress > progressPrev) {
#pragma omp critical(Database__save)
                if (progress > progressPrev) {
                    std::cout << std::string(progress - progressPrev, '.')
                              << std::flush;
                    progressPrev = progress;
                }
            }
        }

        saved += size;
    }

    assert(saved == toSave);
}

void N2D2::Database::append(const Database& database) {
    const unsigned int offsetID = mStimuli.size();
    const unsigned int offsetLabelID = mLabelsName.size();

    mStimuli.reserve(offsetID + database.mStimuli.size());

    for (std::vector<Stimulus>::const_iterator it = database.mStimuli.begin(),
        itEnd = database.mStimuli.end(); it != itEnd; ++it)
    {
        mStimuli.push_back((*it));

        if (mStimuli.back().label >= 0)
            mStimuli.back().label += offsetLabelID;

        for (std::vector<ROI*>::const_iterator itROIs
            = mStimuli.back().ROIs.begin(), 
            itROIsEnd = mStimuli.back().ROIs.end(); itROIs != itROIsEnd;
            ++itROIs)
        {
            if ((*itROIs)->getLabel() >= 0)
                (*itROIs)->setLabel((*itROIs)->getLabel() + offsetLabelID);
        }
    }

    mLabelsName.insert(mLabelsName.end(),
        database.mLabelsName.begin(), database.mLabelsName.end());
    mStimuliData.insert(mStimuliData.end(),
        database.mStimuliData.begin(), database.mStimuliData.end());
    mStimuliLabelsData.insert(mStimuliLabelsData.end(),
        database.mStimuliLabelsData.begin(), database.mStimuliLabelsData.end());
    mStimuliTargetData.insert(mStimuliTargetData.end(),
        database.mStimuliTargetData.begin(), database.mStimuliTargetData.end());

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(All);

    for (std::vector<StimuliSet>::const_iterator itSet = stimuliSets.begin(),
                                                 itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        const unsigned int size = database.mStimuliSets(*itSet).size();
        mStimuliSets(*itSet).reserve(mStimuliSets(*itSet).size() + size);

        for (int i = 0; i < (int)size; ++i) {
            const StimulusID id = database.mStimuliSets(*itSet)[i];
            mStimuliSets(*itSet).push_back(id + offsetID);
        }
    }
}

void N2D2::Database::partitionStimulus(StimulusID id, StimuliSet set)
{
    if (set == Unpartitioned)
        return;

    std::vector<StimulusID>::iterator it
        = std::find(mStimuliSets(Unpartitioned).begin(), 
                    mStimuliSets(Unpartitioned).end(), id);

    if (it != mStimuliSets(Unpartitioned).end()) {
        mStimuliSets(set).push_back(id);
        mStimuliSets(Unpartitioned).erase(it);
    }
    else {
        std::ostringstream errorMsg;
        errorMsg << "Database::partitionStimulus(): stimulus ID "
            << id << " does not exist or is already partitionned.";

        throw std::runtime_error(errorMsg.str());
    }
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

    mStimuliSets(set).reserve(mStimuliSets(set).size() + nbStimuli);

    if (!mRandomPartitioning) {
        mStimuliSets(set).insert(mStimuliSets(set).end(),
                            mStimuliSets(Unpartitioned).begin(),
                            mStimuliSets(Unpartitioned).begin() + nbStimuli);
        mStimuliSets(Unpartitioned).erase(mStimuliSets(Unpartitioned).begin(),
                            mStimuliSets(Unpartitioned).begin() + nbStimuli);
    }
    else {
        for (unsigned int i = 0; i < nbStimuli; ++i) {
            const unsigned int idx = Random::randUniform(0, maxStimuli - 1);
            mStimuliSets(set).push_back(mStimuliSets(Unpartitioned)[idx]);
            mStimuliSets(Unpartitioned)
                .erase(mStimuliSets(Unpartitioned).begin() + idx);
            --maxStimuli;
        }

        assert(maxStimuli == mStimuliSets(Unpartitioned).size());
    }
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

        if (maxStimuliPerLabel == 0) {
            std::cout << Utils::cwarning << "Warning:"
                " partitionStimuliPerLabel(): maxStimuliPerLabel is 0 with"
                " equiLabel=true" << Utils::cdef << std::endl;

            std::cout << "StimuliPerLabel are:" << std::endl;

            for (std::vector<std::vector<unsigned int> >::iterator it
                = labelsStimuliUnpartitionedIndexes.begin(),
                itBegin = labelsStimuliUnpartitionedIndexes.begin(),
                itEnd = labelsStimuliUnpartitionedIndexes.end();
                it != itEnd;
                ++it)
            {
                std::cout << "  " << (it - itBegin)
                    << " " << getLabelName(it - itBegin)
                    << " " << (*it).size() << std::endl;
            }
        }
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
        // round() HalfDown alone would work with perfectly represented floating
        // point numbers, but numerical imprecision may lead in some cases to
        // false result, if the actual number is 2.5000000001 or 2.4999999999
        // instead of 2.5. That's why we have the min() in addition.
        const unsigned int nbValidationTest
            = std::min((unsigned int)Utils::round(
            nbStimuli * (validationPerLabel + testPerLabel), Utils::HalfDown),
            nbStimuli - nbLearn);
        const double fracValidation
            = (validationPerLabel > 0)
                  ? (validationPerLabel / (validationPerLabel + testPerLabel))
                  : 0.0;

        const unsigned int nbValidation = Utils::round(
            nbValidationTest * fracValidation, Utils::HalfUp);
        // Same as above for the min()
        const unsigned int nbTest = std::min((unsigned int)Utils::round(
            nbValidationTest * (1.0 - fracValidation), Utils::HalfDown),
            nbStimuli - nbLearn - nbValidation);

        std::map<StimuliSet, unsigned int> partition;
        partition.insert(std::make_pair(Learn, nbLearn));
        partition.insert(std::make_pair(Validation, nbValidation));
        partition.insert(std::make_pair(Test, nbTest));

        for (std::map<StimuliSet, unsigned int>::const_iterator itSet
             = partition.begin(),
             itSetEnd = partition.end();
             itSet != itSetEnd;
             ++itSet)
        {
            partitionIndexes(unpartitionedIndexes,
                             partitionedIndexes,
                             (*itSet).second,
                             (*itSet).first);
        }
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

int N2D2::Database::addLabel(const std::string& labelName) {
    const std::vector<std::string>::iterator itBegin = mLabelsName.begin();
    std::vector<std::string>::const_iterator it
        = std::find(itBegin, mLabelsName.end(), labelName);

    if (it != mLabelsName.end()) {
        throw std::runtime_error(
            "Database::addLabel(): a label with the name \"" + labelName
            + "\" already exists");
    }

    mLabelsName.push_back(labelName);
    return (mLabelsName.size() - 1);
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

void N2D2::Database::removeStimuli(const std::vector<StimulusID>& ids)
{
    std::vector<StimulusID> sortedIds(ids);
    std::sort(sortedIds.begin(), sortedIds.end());

    std::map<int, int> stimuliMapping;
    int offset = 0;

    // mStimuli.erase() is very slow, better create a new vector and swap!
    std::vector<Stimulus> newStimuli;

    for (unsigned int i = 0, size = mStimuli.size(); i < size; ++i) {
        if (std::binary_search(sortedIds.begin(), sortedIds.end(), i)) {
            stimuliMapping.insert(std::make_pair(i, -1));
            //mStimuli.erase(mStimuli.begin() + (i - offset));
            ++offset;
        }
        else {
            stimuliMapping.insert(std::make_pair(i, i - offset));
            newStimuli.push_back(mStimuli[i]);
        }
    }

    mStimuli.swap(newStimuli);

    std::vector<StimuliSet> stimuliSets;
    stimuliSets.push_back(Learn);
    stimuliSets.push_back(Validation);
    stimuliSets.push_back(Test);
    stimuliSets.push_back(Unpartitioned);

    for (std::vector<Database::StimuliSet>::const_iterator it
         = stimuliSets.begin(), itEnd = stimuliSets.end();
         it != itEnd;
         ++it)
    {
        // mStimuliSets(*it).erase() is very slow...
        std::vector<StimulusID> newStimuliSet;

        for (int idx = mStimuliSets(*it).size() - 1; idx >= 0; --idx)
        {
            const int id = mStimuliSets(*it)[idx];
            const int newId = stimuliMapping[id];

            //if (newId == -1) {
            //    mStimuliSets(*it)
            //        .erase(mStimuliSets(*it).begin() + idx);
            //}
            //else
            //    mStimuliSets(*it)[idx] = newId;

            if (newId != -1)
                newStimuliSet.push_back(newId);
        }

        mStimuliSets(*it).swap(newStimuliSet);
    }
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

void N2D2::Database::removeLabels(const std::vector<int>& labels)
{
    std::vector<int> sortedLabels(labels);
    std::sort(sortedLabels.begin(), sortedLabels.end());

    std::map<int, int> labelsMapping;
    int offset = 0;

    for (unsigned int i = 0, size = mLabelsName.size(); i < size; ++i) {
        if (std::binary_search(sortedLabels.begin(), sortedLabels.end(), i)) {
            labelsMapping.insert(std::make_pair(i, -1));
            mLabelsName.erase(mLabelsName.begin() + (i - offset));
            ++offset;
        }
        else
            labelsMapping.insert(std::make_pair(i, i - offset));
    }

    std::vector<StimulusID> stimuliToRemove;

    for (int id = mStimuli.size() - 1; id >= 0; --id) {
        const int label = mStimuli[id].label;

        if (label >= 0) {
            const int newLabel = labelsMapping[label];

            if (newLabel == -1)
                stimuliToRemove.push_back(id);
            else
                mStimuli[id].label = newLabel;
        }
    }

    removeStimuli(stimuliToRemove);
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

void N2D2::Database::sortAndDropLabels(unsigned int nbKeep,
                                       bool ascending,
                                       StimuliSetMask setMask)
{
    std::vector<unsigned int> nbStimuliWithLabel(mLabelsName.size(), 0);

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        const unsigned int size = mStimuliSets(*itSet).size();

        for (int i = 0; i < (int)size; ++i) {
            const StimulusID id = mStimuliSets(*itSet)[i];

            if (mStimuli[id].label >= 0)
                ++nbStimuliWithLabel.at(mStimuli[id].label);
        }
    }

    const unsigned int keep = std::min(nbKeep,
                                       (unsigned int)mLabelsName.size());

    std::vector<int> labelIdx(mLabelsName.size());
    std::iota(labelIdx.begin(), labelIdx.end(), 0);

    // Sort indexes based on comparing values in nbStimuliWithLabel
    if (ascending) {
        std::partial_sort(labelIdx.begin(),
            labelIdx.begin() + keep,
            labelIdx.end(),
            [&nbStimuliWithLabel](int i1, int i2)
                {return nbStimuliWithLabel[i1] < nbStimuliWithLabel[i2];});
    }
    else {
        std::partial_sort(labelIdx.begin(),
            labelIdx.begin() + keep,
            labelIdx.end(),
            [&nbStimuliWithLabel](int i1, int i2)
                {return nbStimuliWithLabel[i1] > nbStimuliWithLabel[i2];});
    }

    removeLabels(std::vector<int>(labelIdx.begin() + keep, labelIdx.end()));
}

std::string N2D2::Database::getStimulusName(StimulusID id,
                                            bool appendSlice) const
{
    assert(id < mStimuli.size());

    if (mStimuli[id].slice != NULL && appendSlice) {
        const cv::Rect bbRect = mStimuli[id].slice->getBoundingRect();

        std::stringstream nameStr;
        nameStr << mStimuli[id].name << "[" << bbRect.y << "," << bbRect.x
                << "]";

        return nameStr.str();
    } else
        return mStimuli[id].name;
}

N2D2::Database::StimuliSet N2D2::Database::getStimulusSet(StimulusID id)
    const
{
    const std::vector<StimuliSet> stimuliSets = getStimuliSets(All);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
         = stimuliSets.begin(),
         itSetEnd = stimuliSets.end();
         itSet != itSetEnd;
         ++itSet)
    {
        if (std::find(mStimuliSets(*itSet).begin(),
            mStimuliSets(*itSet).end(), id) != mStimuliSets(*itSet).end())
        {
            return (*itSet);
        }
    }

    return Unpartitioned;
}

std::vector<std::shared_ptr<N2D2::ROI> >
N2D2::Database::getStimulusROIs(StimulusID id) const
{
    assert(id < mStimuli.size());

    std::vector<std::shared_ptr<ROI> > stimulusROIs;
    std::transform(mStimuli[id].ROIs.begin(),
                   mStimuli[id].ROIs.end(),
                   std::back_inserter(stimulusROIs),
                   std::bind(&ROI::clone, std::placeholders::_1));

    if (mCompositeLabel == Auto
        && mStimuli[id].label >= 0
        && !stimulusROIs.empty())
    {
        unsigned int nbLabelROIs = 0;

        for (std::vector<std::shared_ptr<ROI> >::iterator
            itROIs = stimulusROIs.begin(), itROIsEnd = stimulusROIs.end();
            itROIs != itROIsEnd; ++itROIs)
        {
            if ((*itROIs)->getLabel() >= 0) {
                // Align ROI to extracted data
                const cv::Rect roiOrg = (*itROIs)->getBoundingRect();
                (*itROIs)->padCrop(roiOrg.tl().x,
                                        roiOrg.tl().y,
                                        roiOrg.width,
                                        roiOrg.height);

                if (mStimuli[id].slice != NULL) {
                    // Align ROI to slice
                    const cv::Rect sliceRect
                        = mStimuli[id].slice->getBoundingRect();

                    (*itROIs)->padCrop(sliceRect.tl().x,
                                            sliceRect.tl().y,
                                            sliceRect.width,
                                            sliceRect.height);
                }

                ++nbLabelROIs; // only for ROIs with label >= 0
            }
        }

        if (nbLabelROIs > 1) {
#pragma omp critical(Database__getStimulusROIs)
            throw std::runtime_error("Database::getStimulusROIs(): "
                                     "number of ROIs should be 1 for "
                                     "non-composite stimuli");
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

unsigned int N2D2::Database::getNbStimuliWithLabel(
    int label,
    StimuliSetMask setMask) const
{
    size_t count = 0;
    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
        = stimuliSets.begin(),
        itSetEnd = stimuliSets.end();
        itSet != itSetEnd;
        ++itSet)
    {
        count += std::count_if(
            mStimuliSets(*itSet).begin(),
            mStimuliSets(*itSet).end(),
            std::bind(std::equal_to<int>(),
                    label,
                    std::bind<int>((int (Database::*)(StimulusID) const) &Database::getStimulusLabel, this, std::placeholders::_1)));
    }

    return count;
}

unsigned int N2D2::Database::getNbROIsWithLabel(int label,
    StimuliSetMask setMask) const
{
    unsigned int nbROIs = 0;

    const std::vector<StimuliSet> stimuliSets = getStimuliSets(setMask);

    for (std::vector<Database::StimuliSet>::const_iterator itSet
        = stimuliSets.begin(),
        itSetEnd = stimuliSets.end();
        itSet != itSetEnd;
        ++itSet)
    {
        for (std::vector<unsigned int>::const_iterator it = mStimuliSets(*itSet).begin(),
                                                itEnd = mStimuliSets(*itSet).end();
            it != itEnd;
            ++it)
        {
            for (std::vector<ROI*>::const_iterator itROIs = mStimuli[(*it)].ROIs.begin(),
                                                itROIsEnd = mStimuli[(*it)].ROIs.end();
                itROIs != itROIsEnd;
                ++itROIs) {
                if ((*itROIs)->getLabel() == label)
                    ++nbROIs;
            }
        }
    }

    return nbROIs;
}

bool N2D2::Database::isLabel(const std::string& labelName) const
{
    return (std::find(mLabelsName.begin(), mLabelsName.end(), labelName)
            != mLabelsName.end());
}

bool N2D2::Database::isMatchingLabel(const std::string& labelMask) const
{
    for (std::vector<std::string>::const_iterator it = mLabelsName.begin(),
                                                  itEnd = mLabelsName.end();
         it != itEnd; ++it)
    {
        if (Utils::match(labelMask, *it))
            return true;
    }

    return false;
}

std::vector<int> N2D2::Database::getMatchingLabelsIDs(
    const std::string& labelMask) const
{
    return getMatchingLabelsIDs(std::vector<std::string>(1, labelMask));
}

std::vector<int> N2D2::Database::getMatchingLabelsIDs(
    const std::vector<std::string>& labelMask) const
{
    std::vector<int> labels;

    for (std::vector<std::string>::const_iterator it = mLabelsName.begin(),
        itBegin = mLabelsName.begin(), itEnd = mLabelsName.end();
        it != itEnd; ++it)
    {
        for (std::vector<std::string>::const_iterator
            itMask = labelMask.begin(), itMaskEnd = labelMask.end();
            itMask != itMaskEnd; ++itMask)
        {
            if (Utils::match(*itMask, *it))
                labels.push_back(it - itBegin);
        }
    }

    return labels;
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
    assert(id < mStimuli.size());

    if (mLoadDataInMemory) {
        if (mStimuliData.empty()) {
#pragma omp critical(Database__getStimulusData)
            if (mStimuliData.empty())
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
    assert(id < mStimuli.size());

    if (mLoadDataInMemory) {
        if (mStimuliLabelsData.empty()) {
#pragma omp critical(Database__getStimulusLabelsData)
            if (mStimuliLabelsData.empty())
                mStimuliLabelsData.resize(mStimuli.size());
        }

        if (mStimuliLabelsData[id].empty()) {
            mStimuliLabelsData[id] = loadStimulusLabelsData(id);
        }

        return mStimuliLabelsData[id];
    } else {
        return loadStimulusLabelsData(id);
    }
}

cv::Mat N2D2::Database::getStimulusTargetData(StimulusID id,
    const cv::Mat& /*frame*/,
    const cv::Mat& /*labels*/,
    const std::vector<std::shared_ptr<ROI> >& /*labelsROI*/)
{
    assert(id < mStimuli.size());

    if (mLoadDataInMemory) {
        if (mStimuliTargetData.empty()) {
#pragma omp critical(Database__getStimulusTargetData)
            if (mStimuliTargetData.empty())
                mStimuliTargetData.resize(mStimuli.size());
        }

        if (mStimuliTargetData[id].empty())
            mStimuliTargetData[id] = loadStimulusTargetData(id);

        return mStimuliTargetData[id];
    } else
        return loadStimulusTargetData(id);
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

int N2D2::Database::getStimuliDepth() {
    if (mStimuliDepth == -1) {
        if(mStimuli.empty()) {
            throw std::runtime_error("Can't get the stimuli depth of an empty database.");
        }

        if (mLoadDataInMemory
            && !mStimuliData.empty() && !mStimuliData[0].empty())
        {
            mStimuliDepth = mStimuliData[0].depth();
        }
        else {
            std::string fileExtension = Utils::fileExtension(mStimuli[0].name);
            std::transform(fileExtension.begin(), fileExtension.end(),
                        fileExtension.begin(), ::tolower);

            auto dataFile = Registrar<DataFile>::create(fileExtension)();
            mStimuliDepth = dataFile->read(mStimuli[0].name).depth();

            std::cout << Utils::cnotice << "Notice: stimuli depth is "
                        << Utils::cvMatDepthToString(mStimuliDepth)
                        << " (according to database first stimulus)"
                        << Utils::cdef << std::endl;
        }
    }

    return mStimuliDepth;
}

cv::Mat N2D2::Database::loadStimulusData(StimulusID id)
{
    // Initialize mStimuliDepth using the first stimulus
    if (mStimuliDepth == -1) {
#pragma omp critical(Database__loadStimulusData)
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
                      << " (according to database first stimulus)"
                      << Utils::cdef << std::endl;
        }
    }

    return loadData(id, mStimuliDepth, mStimuli[id].name);
}

cv::Mat N2D2::Database::loadStimulusLabelsData(StimulusID id)
{
    std::string fileExtension = Utils::fileExtension(mStimuli[id].name);
    std::transform(fileExtension.begin(),
                   fileExtension.end(),
                   fileExtension.begin(),
                   ::tolower);

    cv::Mat labels = this->readLabel(id);

    if (mCompositeLabel != None && (mCompositeLabel != Auto
                                    || mStimuli[id].label == -1
                                    || !labels.empty()))
    {
        std::shared_ptr<DataFile> dataFile = Registrar
            <DataFile>::create(fileExtension)();

        const int defaultLabel = getDefaultLabelID();

        // Composite stimulus
        // Construct the labels matrix with the ROIs
        cv::Mat stimulus = dataFile->read(mStimuli[id].name);

        if (labels.empty()) {
            const int labelID = ((mCompositeLabel == Auto
                    && mStimuli[id].label == -1)
                || mCompositeLabel == Default
                || (mCompositeLabel == Disjoint && !mStimuli[id].ROIs.empty()))
                    ? defaultLabel : mStimuli[id].label;

            labels = cv::Mat(stimulus.rows, stimulus.cols, CV_32SC1,
                            cv::Scalar(labelID));
        }
        else if (mStimuli[id].label >= 0) {
            // use labels as a mask for the stimulus label
            labels.setTo(defaultLabel, labels == 0);
            labels.setTo(mStimuli[id].label, labels > 0);
        }

        if (mStimuli[id].slice != NULL)
            labels = mStimuli[id].slice->extract(labels);

        for (std::vector<ROI*>::const_iterator it = mStimuli[id].ROIs.begin(),
                                               itEnd = mStimuli[id].ROIs.end();
             it != itEnd;
             ++it)
        {
            try {
                (*it)->append(labels, mROIsMargin, defaultLabel);
            }
            catch (const std::exception& e)
            {
#pragma omp critical(Database__loadStimulusLabelsData)
                std::cout << Utils::cwarning << "Could not append ROI #"
                    << (it - mStimuli[id].ROIs.begin()) << " to stimulus "
                    << mStimuli[id].name << " (" << stimulus.cols
                    << "x" << stimulus.rows << "):\n" << Utils::cdef
                    << e.what() << std::endl;
            }
        }

        return labels;
    } else {
        // Non-composite stimulus
        if (mCompositeLabel == Auto && !mStimuli[id].ROIs.empty()) {
            if (mStimuli[id].ROIs.size() != 1) {
#pragma omp critical(Database__loadStimulusLabelsData)
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

cv::Mat N2D2::Database::loadStimulusTargetData(StimulusID id) {
    if (((std::string)mTargetDataPath).empty())
        return cv::Mat();

    // Initialize mStimuliTargetDepth using the first stimulus
    if (mStimuliTargetDepth == -1) {
        const std::string fileName0 = Utils::expandEnvVars(mTargetDataPath)
            + "/" + Utils::baseName(mStimuli[0].name);

#pragma omp critical(Database__loadStimulusTargetData)
        if (mStimuliTargetDepth == -1) {
            std::string fileExtension = Utils::fileExtension(fileName0);
            std::transform(fileExtension.begin(),
                           fileExtension.end(),
                           fileExtension.begin(),
                           ::tolower);

            std::shared_ptr<DataFile> dataFile = Registrar
                <DataFile>::create(fileExtension)();
            mStimuliTargetDepth = dataFile->read(fileName0).depth();

            std::cout << Utils::cnotice << "Notice: stimuli depth is "
                      << Utils::cvMatDepthToString(mStimuliTargetDepth)
                      << " (according to database first stimulus)"
                      << Utils::cdef << std::endl;
        }
    }

    const std::string fileName = Utils::expandEnvVars(mTargetDataPath)
        + "/" + Utils::baseName(mStimuli[id].name);

    return loadData(id, mStimuliTargetDepth, fileName);
}

cv::Mat N2D2::Database::loadData(
    StimulusID id,
    int depth,
    const std::string fileName) const
{
    std::string fileExtension = Utils::fileExtension(fileName);
    std::transform(fileExtension.begin(),
                   fileExtension.end(),
                   fileExtension.begin(),
                   ::tolower);

    std::shared_ptr<DataFile> dataFile = Registrar
        <DataFile>::create(fileExtension)();
    cv::Mat data = dataFile->read(fileName);

    if (!((std::string)mMultiChannelMatch).empty()) {
        const std::regex regexp((std::string)mMultiChannelMatch);
        const std::vector<std::string>& multiChannelReplace
            = mMultiChannelReplace.get<std::vector<std::string> >();

        std::vector<cv::Mat> channels;

        for (size_t ch = 0; ch < multiChannelReplace.size(); ++ch) {
            std::string chFileName;
            cv::Mat chData;

            if (std::regex_match(fileName, regexp)) {
                chFileName = std::regex_replace(fileName, regexp,
                                                multiChannelReplace[ch]);

                if (chFileName == fileName)
                    chData = data;
                else if (std::ifstream(chFileName).good())
                    chData = dataFile->read(chFileName);
            }
            //else {
            //    std::cout << Utils::cwarning << "Warning: no match for channel "
            //        "#" << ch << " for stimulus: " << fileName
            //        << Utils::cdef << std::endl;
            //}

            if (chData.empty()) {
                //std::cout << Utils::cnotice << "Notice: missing channel #"
                //        << ch << " data for stimulus: " << fileName
                //        << Utils::cdef << std::endl;

                chData = cv::Mat(data.rows, data.cols, data.type(),
                                 cv::Scalar(0));
            }

            if (chData.size() != data.size() || chData.depth() != data.depth())
            {
                std::stringstream errorStr;
                errorStr << "Database::loadStimulusData():"
                    " channel #" << ch << " \"" << chFileName << "\" size / depth ("
                    << chData.size() << " / " << chData.depth() << ") does not"
                    " match with channel \"" << fileName << "\" ("
                    << data.size() << " / " << data.depth() << ")";

#pragma omp critical(Database__loadData)
                throw std::runtime_error(errorStr.str());
            }

            channels.push_back(chData);
        }

        cv::merge(channels, data);
    }

    // Check stimulus depth
    if (data.depth() != depth) {
        std::cout << Utils::cnotice << "Notice: converting depth from "
                  << Utils::cvMatDepthToString(data.depth()) << " to "
                  << Utils::cvMatDepthToString(depth)
                  << " for stimulus: " << fileName << Utils::cdef
                  << std::endl;

        cv::Mat dataConverted;
        data.convertTo(dataConverted,
                       depth,
                       Utils::cvMatDepthUnityValue(depth).second
                       / Utils::cvMatDepthUnityValue(data.depth()).second);
        data = dataConverted;
    }

    if (mCompositeLabel == Auto
        && mStimuli[id].label >= 0
        && !mStimuli[id].ROIs.empty())
    {
        bool extracted = false;

        for (std::vector<ROI*>::const_iterator
            itROIs = mStimuli[id].ROIs.begin(),
            itROIsEnd = mStimuli[id].ROIs.end();
            itROIs != itROIsEnd; ++itROIs)
        {
            if ((*itROIs)->getLabel() >= 0) {
                // Non-composite stimulus with ROI
                if (!extracted) {
                    data = (*itROIs)->extract(data);
                    extracted = true;
                }
                else {
#pragma omp critical(Database__loadData)
                    throw std::runtime_error("Database::loadStimulusData():"
                        " number of ROIs should be 1 for non-composite"
                        " stimuli");
                }
            }
        }
    }

    if (mStimuli[id].slice != NULL)
        data = mStimuli[id].slice->extract(data);

    return data;
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

    if (nbStimuli > maxStimuli) {
        std::stringstream errorMsg;
        errorMsg << "Database::partitionIndexes():"
            " partition size (" << nbStimuli << ") larger than"
            " the number of available stimuli (" << maxStimuli << ") for the"
            " set " << set << ".";

        throw std::runtime_error(errorMsg.str());
    }

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