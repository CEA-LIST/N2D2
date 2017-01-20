/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
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

#include "Aer.hpp"

N2D2::Aer::Aer(const std::shared_ptr<HeteroEnvironment>& environment)
    : mEnvironment(environment),
      mAerJitter(this, "AerJitter", 0 * TimeS),
      mAerUniformNoise(this, "AerUniformNoise", 0.0)
{
    // ctor
}

N2D2::Aer::Aer(const std::shared_ptr<Environment>& environment)
    : mEnvironment(std::make_shared<HeteroEnvironment>(environment)),
      mAerJitter(this, "AerJitter", 0 * TimeS),
      mAerUniformNoise(this, "AerUniformNoise", 0.0)
{
    // ctor
}

std::pair<N2D2::Time_T, N2D2::Time_T> N2D2::Aer::getTimes(const std::string
                                                          & fileName) const
{
    std::ifstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not open AER file: " + fileName);

    AerEvent event(readVersion(data));

    if (!event.read(data).good())
        throw std::runtime_error("Invalid AER file: " + fileName);

    const Time_T timeStart = event.time;

    data.seekg(-event.size(), std::ios::end);

    if (!event.read(data).good())
        throw std::runtime_error("Invalid AER file: " + fileName);

    return std::make_pair(timeStart, event.time);
}

N2D2::Aer::AerData_T N2D2::Aer::read(const std::string& fileName,
                                     AerEvent::AerFormat format,
                                     bool ret,
                                     Time_T offset,
                                     Time_T start,
                                     Time_T end)
{
    static std::map<std::string, std::pair<Time_T, std::streampos> > history;

    std::ifstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not open AER file: " + fileName);

    AerEvent event(readVersion(data));

    if (history.find(fileName) != history.end() && start
                                                   == history[fileName].first)
        // Start from last position
        data.seekg(history[fileName].second);

    AerData_T events;
    unsigned int nbEvents = 0;
    Time_T lastTime = start;

    while (event.read(data).good()) {
        // Tolerate a lag of 100ms because real AER retina captures are not
        // always non-monotonic
        if (event.time + 100 * TimeMs >= lastTime) {
            if (end > 0 && event.time >= end) {
                history[fileName] = std::make_pair(
                    end, (std::streamoff)data.tellg() - event.size());
                break;
            }

            if (mAerJitter > 0) {
                event.time = (Time_T)Random::randNormal(event.time, mAerJitter);

                if (event.time < start || (end > 0 && event.time >= end))
                    continue;
            }

            if (ret)
                events.push_back(std::make_pair(event.time, event.addr));
            else {
                event.maps(format);
                (*mEnvironment)[event.map]
                    ->getNodeByIndex(event.channel, event.node)
                    ->incomingSpike(NULL, offset + event.time);
            }

            // Take the MAX because event.time can be non-monotonic because of
            // AER lag or added jitter
            lastTime = std::max(event.time, lastTime);
            ++nbEvents;
        } else if (nbEvents > 0) {
            std::cout << "Current event time is " << event.time / TimeUs
                      << " us, last event time was " << lastTime / TimeUs
                      << " us" << std::endl;
            throw std::runtime_error("Non-monotonic AER data in file: "
                                     + fileName);
        }
    }

    if (mAerUniformNoise > 0.0) {
        if (mAerUniformNoise >= 1.0)
            throw std::domain_error(
                "mAerUniformNoise: out of range (must be < 1.0)");

        const Time_T timeEnd = (end > 0) ? end : lastTime;
        const unsigned int nbNoiseEvents
            = (unsigned int)(nbEvents * mAerUniformNoise
                             / (1.0 - mAerUniformNoise));

        for (unsigned int i = 0; i < nbNoiseEvents; ++i) {
            event.time = (Time_T)(Random::randUniform((double)start / TimeUs,
                                                      (double)timeEnd / TimeUs)
                                  * TimeUs);
            event.map = Random::randUniform(0, mEnvironment->size() - 1);
            event.channel = Random::randUniform(
                0, (*mEnvironment)[event.map]->getNbChannels() - 1);
            event.node = Random::randUniform(
                0, (*mEnvironment)[event.map]->getNbNodes(event.channel) - 1);

            if (ret) {
                event.unmaps(format);
                events.push_back(std::make_pair(event.time, event.addr));
            } else
                (*mEnvironment)[event.map]
                    ->getNodeByIndex(event.channel, event.node)
                    ->incomingSpike(NULL, offset + event.time);
        }

        nbEvents += nbNoiseEvents;
    }

    if (ret)
        std::sort(events.begin(),
                  events.end(),
                  Utils::PairFirstPred<Time_T, unsigned int>());

    return events;
}

void N2D2::Aer::merge(const std::string& source1,
                      const std::string& source2,
                      AerEvent::AerFormat format,
                      const std::string& destination)
{
    AerData_T events1 = read(source1, format, true);
    AerData_T events2 = read(source2, format, true);

    AerData_T mergedEvents;
    mergedEvents.reserve(events1.size() + events2.size());

    std::merge(events1.begin(),
               events1.end(),
               events2.begin(),
               events2.end(),
               std::back_inserter(mergedEvents),
               Utils::PairFirstPred<Time_T, unsigned int>());

    save(destination, mergedEvents);
}

void N2D2::Aer::save(const std::string& fileName,
                     const AerData_T& events,
                     bool append,
                     double version)
{
    std::ofstream data(fileName.c_str(),
                       (append) ? (std::fstream::binary | std::fstream::app)
                                : std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not create AER file: " + fileName);

    if (!append) {
        std::stringstream head;
        head << "#!AER-DAT" << version << "\n";
        data.write(head.str().c_str(), head.str().size());
    }

    AerEvent event(version);

    for (AerData_T::const_iterator it = events.begin(), itEnd = events.end();
         it != itEnd;
         ++it) {
        event.time = (*it).first;
        event.addr = (*it).second;

        event.write(data);
    }
}

unsigned int N2D2::Aer::loadVideo(const std::string& fileName,
                                  unsigned int fps,
                                  double threshold,
                                  AerCodingMode mode)
{
    if (threshold <= 0.0)
        throw std::domain_error("Threshold out of range (must be > 0.0)");

    cv::VideoCapture video(fileName);
    video.set(
        CV_CAP_PROP_POS_FRAMES,
        0.0); // Bug in OpenCV 2.4.7 (see http://code.opencv.org/issues/3030)

    if (!video.isOpened())
        throw std::runtime_error("Could not open video file: " + fileName);

    const unsigned int nbMaps = mEnvironment->size();

    cv::Mat retFrame, grayRetFrame, mapFrame;
    std::vector<std::vector<cv::Mat> > frame(nbMaps), prevFrame(nbMaps),
        acc(nbMaps);
    std::vector<std::map<unsigned int, unsigned int> > negFilter(nbMaps);

    for (unsigned int map = 0; map < nbMaps; ++map) {
        const unsigned int width
            = (unsigned int)((*mEnvironment)[map]->getSizeX());
        const unsigned int height
            = (unsigned int)((*mEnvironment)[map]->getSizeY());
        const unsigned int nbChannels = (*mEnvironment)[map]->getNbChannels();

        frame[map].resize(nbChannels);
        prevFrame[map].resize(nbChannels);
        acc[map].reserve(nbChannels);

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            acc[map].push_back(cv::Mat(cv::Size(width, height), CV_32FC1, 0.0));

            if (channel + 1 < nbChannels) {
                for (unsigned int nextChannel = channel + 1;
                     nextChannel < nbChannels;
                     ++nextChannel) {
                    if ((*mEnvironment)[map]->getFilter(channel)->getKernel()
                        == (-(*((*mEnvironment)[map]->getFilter(nextChannel))))
                               .getKernel()) {
                        negFilter[map]
                            .insert(std::make_pair(nextChannel, channel));
                    }
                }
            }
        }
    }

    unsigned int numEvents = 0;
    const std::string aerFileName = Utils::fileBaseName(fileName) + ".dat";

    unsigned int f = 0;

    for (; video.grab(); ++f) {
        video.retrieve(retFrame);
        cv::cvtColor(retFrame, grayRetFrame, CV_RGB2GRAY);
        grayRetFrame.convertTo(grayRetFrame,
                               CV_32FC1,
                               1.0 / 255.0); // Assuming input frame is 8 bits

        AerData_T events;

        // For each scale ...
        for (unsigned int map = 0; map < nbMaps; ++map) {
            const unsigned int width
                = (unsigned int)((*mEnvironment)[map]->getSizeX());
            const unsigned int height
                = (unsigned int)((*mEnvironment)[map]->getSizeY());
            const unsigned int nbChannels
                = (*mEnvironment)[map]->getNbChannels();

            //  NOTE: cv::resize() doesn't work on OpenCV 2.1.0 with CV_64FC1 on
            // tested videos
            cv::resize(grayRetFrame, mapFrame, cv::Size(width, height));

            // ... sample the image and for each channel
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                // ... sample the image and convolve it
                const bool neg
                    = (negFilter[map].find(channel) != negFilter[map].end());

                if (!neg) {
                    const std::shared_ptr<FilterTransformation> filter_
                        = (*mEnvironment)[map]->getFilter(channel);

                    if (filter_->getKernel().size() > 1)
                        cv::filter2D(mapFrame,
                                     frame[map][channel],
                                     -1,
                                     (cv::Mat)filter_->getKernel());
                    else
                        frame[map][channel] = (filter_->getKernel().at(0))
                                              * mapFrame;
                }

                if (f > 0) { // Skip first frame
                    if (!neg) {
                        acc[map][channel] += (mode == AccumulateDiff)
                                                 ? frame[map][channel]
                                                   - prevFrame[map][channel]
                                                 : frame[map][channel];
                    }

                    const double accSign = (neg) ? -1.0 : 1.0;
                    const unsigned int accIndex
                        = (neg) ? negFilter[map][channel] : channel;

                    cv::MatIterator_<float> itBegin = acc[map][accIndex].begin
                                                      <float>();

                    for (cv::MatIterator_<float> it = itBegin,
                                                 itEnd = acc[map][accIndex].end
                                                         <float>();
                         it != itEnd;
                         ++it) {
                        unsigned int i = 1;
                        const double value = accSign * (*it);

                        while (accSign * (*it) >= threshold) {
                            (*it) -= (float)(accSign * threshold);
                            Time_T timestamp = (Time_T)(
                                (f - 1 + i * threshold / value) / fps * TimeS);
                            events.push_back(std::make_pair(
                                timestamp,
                                AerEvent::unmaps(map, channel, it - itBegin)));
                            ++i;
                        }
                    }
                }

                if (!neg)
                    frame[map][channel].copyTo(prevFrame[map][channel]);
            }
        }

        numEvents += events.size();

        std::sort(events.begin(),
                  events.end(),
                  Utils::PairFirstPred<Time_T, unsigned int>());
        save(aerFileName, events, f > 1); // Append if f > 1

        if (numEvents / 100000 > (numEvents - events.size()) / 100000)
            std::cout << "[loadVideo] " << numEvents << " events @ frame #" << f
                      << std::endl;
    }

    std::cout << "[loadVideo] *** " << numEvents << " events generated for "
              << f << " frames (" << f / (double)fps << " s)" << std::endl;

    return numEvents;
}

void N2D2::Aer::viewer(const std::string& fileName,
                       AerEvent::AerFormat format,
                       const std::string& labelFile,
                       unsigned int labelTime,
                       const std::string& videoName)
{
    AerData_T events = read(fileName, format, true);
    std::vector<std::pair<Time_T, std::vector<unsigned int> > > labels;
    std::set<unsigned int> labelsId;

    if (!labelFile.empty()) {
        std::ifstream labelData(labelFile.c_str());

        if (!labelData.good())
            throw std::runtime_error("Could not open label data file: "
                                     + fileName);

        std::string line;

        while (std::getline(labelData, line)) {
            std::istringstream sLine(line);
            Time_T time;

            if (sLine >> time) {
                std::vector<unsigned int> data;
                std::copy(std::istream_iterator<unsigned int>(sLine),
                          std::istream_iterator<unsigned int>(),
                          std::back_inserter(data));

                if (!data.empty())
                    labelsId.insert(data[0]);

                labels.push_back(std::make_pair(time, data));
            }
        }

        std::sort(labels.begin(),
                  labels.end(),
                  Utils::PairFirstPred<Time_T, std::vector<unsigned int> >());
    }

    cv::namedWindow("N2D2", CV_WINDOW_NORMAL);

    AerEvent event;
    Time_T timeLast = 0 * TimeS;
    unsigned int idx = 0, size = events.size();
    unsigned int labelIdx = 0, labelSize = labels.size();

    const unsigned int width = (*mEnvironment)[0]->getSizeX();
    const unsigned int height = (*mEnvironment)[0]->getSizeY();

    if (size > 0)
        timeLast = events[0].first; // Start from the first event (the AER
    // sequence may not start from 0 s)

    cv::VideoWriter video;

    if (!videoName.empty()) {
        video.open(videoName,
                   CV_FOURCC('H', 'F', 'Y', 'U'),
                   50.0,
                   cv::Size(width, height));

        if (!video.isOpened())
            throw std::runtime_error("Unable to write video file: "
                                     + videoName);
    }

    while (idx < size) {
        cv::Mat frame(
            cv::Size(width, height), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
        bool isLabel = false;

        while (idx < size && events[idx].first < 20 * TimeMs + timeLast) {
            event.time = events[idx].first;
            event.addr = events[idx].second;
            event.maps(format);

            const std::shared_ptr<FilterTransformation> filter_
                = (*mEnvironment)[event.map]->getFilter(event.channel);

            assert(event.node / width < height);
            assert(event.node % width < width);

            if (filter_->getKernel().size() > 1) {
                double r, g, b;
                std::tie(r, g, b) = Utils::hsvToRgb(
                    360.0 * filter_->getOrientation(), 0.5, 1.0);

                frame.at<cv::Vec3f>(event.node / width, event.node % width)
                    = cv::Vec3f(b, g, r);
            } else {
                frame.at<cv::Vec3f>(event.node / width, event.node % width)
                    = (filter_->getKernel().at(0) > 0.0)
                          ? cv::Vec3f(1.0, 1.0, 1.0)
                          : cv::Vec3f(0.0, 0.0, 0.0);
            }

            ++idx;
        }

        while (labelIdx < labelSize && labels[labelIdx].first < 20 * TimeMs
                                                                + timeLast) {
            const std::vector<unsigned int>& lData = labels[labelIdx].second;

            if (lData.empty())
                cv::rectangle(frame,
                              cv::Point(0, 0),
                              cv::Point(width - 1, height - 1),
                              cv::Scalar(1.0, 0.0, 0.0));
            else if (lData.size() == 4) {
                cv::rectangle(
                    frame,
                    cv::Point(lData[0], lData[1]),
                    cv::Point(lData[0] + lData[2] - 1, lData[1] + lData[3] - 1),
                    cv::Scalar(1.0, 0.0, 0.0));
            } else if (lData.size() == 5) {
                const std::set<unsigned int>::iterator it
                    = labelsId.find(lData[0]);
                const double hue = 360.0 * std::distance(labelsId.begin(), it)
                                   / (double)labelsId.size();

                double r, g, b;
                std::tie(r, g, b) = Utils::hsvToRgb(hue, 1.0, 1.0);

                cv::rectangle(
                    frame,
                    cv::Point(lData[1], lData[2]),
                    cv::Point(lData[1] + lData[3] - 1, lData[2] + lData[4] - 1),
                    cv::Scalar(b, g, r));
            }

            ++labelIdx;
            isLabel = true;
        }

        cv::imshow("N2D2", frame);
        timeLast += 20 * TimeMs;

#if !defined(WIN32) && !defined(__APPLE__)
        const int excepts = fegetexcept();
        fedisableexcept(FE_UNDERFLOW); // cv::waitKey() can trigger a floating
// exception, apparently an underflow,
// with OpenCV 2.3
#endif

        if (isLabel)
            cv::waitKey(labelTime);
        else {
            if (cv::waitKey(20) >= 0)
                break;
        }

#if !defined(WIN32) && !defined(__APPLE__)
        feenableexcept(excepts);
#endif

        if (!videoName.empty()) {
            cv::Mat frameVideo(cv::Size(width, height), CV_8UC3);
            frame.convertTo(
                frameVideo, CV_8UC3, 255); // Assuming input frame is 8 bits
            video << frameVideo;
        }
    }
}

double N2D2::Aer::readVersion(std::ifstream& data) const
{
    std::string line;
    double version = 1.0; // Default version

    while (data.peek() == '#' && getline(data, line)) {
        if (line.compare(0, 9, "#!AER-DAT") == 0) {
            std::stringstream versionStr(line.substr(9));
            versionStr >> version;
        }
    }

    return version;
}
