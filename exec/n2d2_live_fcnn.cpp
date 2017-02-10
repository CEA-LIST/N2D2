/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include <signal.h>
#include <thread>
#include <mutex>

#include "N2D2.hpp"

#include "DeepNet.hpp"
#include "Target/TargetScore.hpp"
#include "Target/TargetROIs.hpp"
#include "DrawNet.hpp"
#include "Generator/DeepNetGenerator.hpp"
#include "Export/DeepNetExport.hpp"
#include "Export/StimuliProviderExport.hpp"
#include "utils/Key.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#ifdef CUDA
#include "CudaContext.hpp"
#endif

#include "n2d2_list_logo.hpp"

using namespace N2D2;

#define DISPLAY_WIDTH 1280
#define DISPLAY_HEIGHT 720

const std::string frameWindow = "N2D2 LIVE FCNN";
const std::string labelsWindow = "Labels";
std::string timingsWindow = "Timings";

std::mutex captureLock;
bool captureFlag = true;

void capture(cv::VideoCapture& video, cv::Mat& frame)
{
    while (true) {
        captureLock.lock();
        video >> frame;
        const bool flag = captureFlag;
        captureLock.unlock();

        if (!flag)
            return;

#ifdef WIN32
        Sleep(10); // ms
#else
        usleep(10000);
#endif
    }
}

cv::Mat drawTimings(const std::vector<std::pair<std::string, double> >& timings,
                    unsigned int width = 480,
                    unsigned int height = 360)
{
    const unsigned int margin = 2;
    const unsigned int labelWidth = std::min(240U, width / 2);
    const unsigned int cellHeight = height / timings.size();

    const double totalTime = std::accumulate(
        timings.begin(),
        timings.end(),
        std::pair<std::string, double>("", 0.0),
        Utils::PairOp
        <std::string, double, Utils::Left<std::string>, std::plus<double> >())
                                 .second;

    cv::Mat mat(cv::Size(width, height), CV_8UC3, cv::Scalar(255, 255, 255));

    for (unsigned int i = 0, size = timings.size(); i < size; ++i) {
        const double relTime = timings[i].second / totalTime;

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            timings[i].first, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
        cv::putText(mat,
                    timings[i].first,
                    cv::Point(margin,
                              (i + 1) * cellHeight
                              - (cellHeight - textSize.height) / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cv::Scalar(0, 0, 0),
                    1,
                    CV_AA);
        cv::rectangle(mat,
                      cv::Point(labelWidth + margin, i * cellHeight + margin),
                      cv::Point(labelWidth + margin
                                + relTime * (width - labelWidth - 2.0 * margin),
                                (i + 1) * cellHeight - margin),
                      cv::Scalar(255, 255, 0),
                      CV_FILLED);

        std::stringstream valueStr;
        valueStr << std::fixed << std::setprecision(2) << (100.0 * relTime)
                 << "%";

        textSize = cv::getTextSize(
            valueStr.str(), cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
        cv::putText(mat,
                    valueStr.str(),
                    cv::Point(labelWidth + 2 * margin,
                              (i + 1) * cellHeight
                              - (cellHeight - textSize.height) / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    cv::Scalar(0, 0, 0),
                    1,
                    CV_AA);
    }

    return mat;
}

unsigned int process(unsigned int frameId,
                     const cv::Mat& imgOrg,
                     cv::Mat& img,
                     const std::shared_ptr<DeepNet>& deepNet,
                     const std::shared_ptr<DeepNet>& deepNetSub,
                     int target,
                     int targetCls,
                     const std::vector<std::string>& targetClsName)
{
    std::cout << "Processing frame #" << frameId << std::endl;

    std::shared_ptr<TargetROIs> targetROIs = deepNet->getTarget
                                             <TargetROIs>(target);
    const std::vector<TargetROIs::DetectedBB> detectedBB
        = targetROIs->getDetectedBB();
    std::vector<std::shared_ptr<ROI> > estimatedLabelsROIs;

    for (unsigned int id = 0, size = detectedBB.size(); id < size; ++id)
        estimatedLabelsROIs.push_back(detectedBB[id].bb->clone());

    Tensor2d<int> estimatedLabels;
    targetROIs->getStimuliProvider()->reverseLabels(
        img, Database::Test, estimatedLabels, estimatedLabelsROIs);

    const double fontScale = 1.5 * imgOrg.cols / 1920.0;

    for (unsigned int id = 0, size = detectedBB.size(); id < size; ++id) {
        // BB detection data
        const cv::Rect rect = estimatedLabelsROIs[id]->getBoundingRect();

        std::cout << "  Object detected #" << id << ": " << rect.width << "x"
                  << rect.height << "@" << rect.x << "," << rect.y << "("
                  << detectedBB[id].score << ")" << std::endl;

        if (targetCls >= 0) {
            // BB classification data
            std::shared_ptr<Target> targetROIsCls = deepNet->getTarget
                                                   <Target>(targetCls);
            int estimatedLabel;
            Float_T estimatedLabelValue;
            std::tie(estimatedLabel, estimatedLabelValue)
                = targetROIsCls->getEstimatedLabel(detectedBB[id].bb);

            const std::string estimatedLabelName
                = (estimatedLabel < (int)targetClsName.size())
                    ? targetClsName[estimatedLabel]
                    : deepNet->getDatabase()->getLabelName(estimatedLabel);

            std::stringstream labelStr;
            labelStr << estimatedLabelName << ": " << std::setprecision(3)
                           << (estimatedLabelValue * 100.0) << "%";

            cv::putText(img,
                        labelStr.str(),
                        cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        cv::Scalar(0, 0, 255),
                        1,
                        CV_AA);

            std::cout << "  |-Class: " << estimatedLabelName
                      << "(" << estimatedLabelValue << ")" << std::endl;
        }

        int baseline = 0;
        int textOffset = cv::getTextSize(
            "dummy", cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline).height;

        if (deepNetSub) {
            const double margin = 0.25; // percent
            const int left = std::max
                <int>(0, rect.x - rect.width * margin / 2.0);
            const int top = std::max
                <int>(0, rect.y - rect.height * margin / 2.0);
            const int width = rect.width + rect.width * margin;
            const int height = rect.height + rect.height * margin;
            const cv::Rect faceRect(left,
                                    top,
                                    std::min(imgOrg.cols - left, width),
                                    std::min(imgOrg.rows - top, height));

            deepNetSub->getStimuliProvider()->streamStimulus(imgOrg(faceRect),
                                                             Database::Test);
            deepNetSub->test(Database::Test);

            std::shared_ptr<TargetScore> targetSub = deepNetSub->getTarget
                                                         <TargetScore>();
            const Tensor3d<int> estimatedLabelSub
                = targetSub->getEstimatedLabels()[0];
            const Tensor3d<Float_T> estimatedLabelValueSub
                = targetSub->getEstimatedLabelsValue()[0];

            std::cout << "  |-Subclass(es):";

            for (unsigned int n = 0; n < 3; ++n) {
                std::stringstream nStr;
                nStr << (n + 1);

                const std::string estimatedLabelNameSub
                    = deepNetSub->getDatabase()->getLabelName(
                        estimatedLabelSub(n));

                std::stringstream labelSubStr;
                labelSubStr << estimatedLabelNameSub << ": "
                         << std::setprecision(3)
                         << (estimatedLabelValueSub(n) * 100.0) << "%";

                const cv::Size textSize
                    = cv::getTextSize(labelSubStr.str(),
                                      cv::FONT_HERSHEY_SIMPLEX,
                                      fontScale,
                                      1,
                                      &baseline);
                cv::putText(img,
                            labelSubStr.str(),
                            cv::Point(rect.x + 5, rect.y + textOffset + 5),
                            cv::FONT_HERSHEY_SIMPLEX,
                            fontScale,
                            cv::Scalar(0, 255, 0),
                            1,
                            CV_AA);
                textOffset += textSize.height + 5;

                std::cout << " " << estimatedLabelNameSub << "("
                          << estimatedLabelValueSub(n) << ")";
            }

            std::cout << std::endl;
        }

        estimatedLabelsROIs[id]->draw(img);
    }

    return detectedBB.size();
}

bool quit = false; // signal flag

void signalHandler(int)
{
    quit = true;
}

std::mutex viewLock;
cv::Mat imgView;
cv::Mat estimatedView;
cv::Mat timingsView;

void viewLoop()
{
    while (true) {
        viewLock.lock();

        if (imgView.data) {
            // const cv::Rect logoArea = cv::Rect(0, 0, N2D2_LIST_LOGO.cols,
            // N2D2_LIST_LOGO.rows);
            // cv::addWeighted(imgView(logoArea), 0.25, N2D2_LIST_LOGO, 0.75,
            // 0.0, imgView(logoArea));
            cv::imshow(frameWindow.c_str(), imgView);
        }

        if (estimatedView.data)
            cv::imshow(labelsWindow.c_str(), estimatedView);

        if (timingsView.data)
            cv::imshow(timingsWindow.c_str(), timingsView);

        viewLock.unlock();

        int k = cv::waitKey(1);

        if (k == KEY_ESC) {
            quit = true;
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    // Program command line options
    ProgramOptions opts(argc, argv);
#ifdef CUDA
    const int cudaDevice = opts.parse("-dev", 0, "CUDA device ID");
#endif
    const std::string videoFileName
        = opts.parse<std::string>("-video", "", "run on a video file");
    const std::string recordFileName = opts.parse<std::string>(
        "-record", "", "record the display to a video file");
    const int target = opts.parse("-target", 1, "network ROIs target");
    const int targetCls = opts.parse("-target-cls", -1,
        "network ROI classes target");
    const std::vector<std::string> targetClsName
        = opts.parse<std::vector<std::string> >("-target-cls-name",
                                                std::vector<std::string>(),
                                                "network ROI class names");
    const std::string iniConfig
        = opts.grab<std::string>("<net>", "network config file (INI)");
    const std::string iniConfigSub = opts.grab<std::string>(
        "", "<net-sub>", "sub network config file (INI)");
    opts.done();

#ifdef CUDA
    CudaContext::setDevice(cudaDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    timingsWindow += std::string(" on ") + prop.name;
#endif

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    cv::VideoCapture video;
    cv::Mat img;
    unsigned int capWidth, capHeight;

    if (!videoFileName.empty()) {
        video.open(videoFileName);

        if (!video.isOpened())
            throw std::runtime_error("Could not open video file: "
                                     + videoFileName);

        if (!video.grab() || !video.retrieve(img))
            throw std::runtime_error(
                "Unable to read first frame in video file: " + videoFileName);

        capWidth = img.cols;
        capHeight = img.rows;
    } else {
        video.open(CV_CAP_ANY);

        if (!video.isOpened())
            throw std::runtime_error("Could not open video stream.");

        video.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
        video.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

        capWidth = video.get(CV_CAP_PROP_FRAME_WIDTH);
        capHeight = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    }

    std::cout << "Capture resolution: " << capWidth << "x" << capHeight
              << std::endl;

    cv::namedWindow(frameWindow.c_str(), CV_WINDOW_NORMAL);
    cv::namedWindow(labelsWindow.c_str(), CV_WINDOW_AUTOSIZE);
    cv::namedWindow(timingsWindow.c_str(), CV_WINDOW_AUTOSIZE);
    cvMoveWindow(frameWindow.c_str(), 0, 0);
    cvResizeWindow(frameWindow.c_str(), DISPLAY_WIDTH, DISPLAY_HEIGHT);
    cvMoveWindow(labelsWindow.c_str(), DISPLAY_WIDTH + 50, 0);
    cvMoveWindow(timingsWindow.c_str(), DISPLAY_WIDTH + 50, 400 + 50);

    // Network topology construction
    Network net;
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, iniConfig);
    deepNet->initialize();
    deepNet->importNetworkFreeParameters("weights");

    std::shared_ptr<DeepNet> deepNetSub;

    if (!iniConfigSub.empty()) {
        deepNetSub = DeepNetGenerator::generate(net, iniConfigSub);
        deepNetSub->initialize();
        deepNetSub->importNetworkFreeParameters("weights-sub");
    }

    cv::Mat frame;
    std::thread captureThread;

    if (videoFileName.empty())
        captureThread = std::thread(capture, std::ref(video), std::ref(frame));

    std::vector<std::pair<std::string, double> > timings;

    cv::VideoWriter videoWriter;

    if (!recordFileName.empty()) {
        videoWriter.open(recordFileName,
                         CV_FOURCC('X', 'V', 'I', 'D'),
                         25.0,
                         cv::Size(capWidth, capHeight));

        if (!videoWriter.isOpened())
            std::cout << Utils::cnotice
                      << "Notice: Unable to write video file: "
                      << recordFileName << Utils::cdef << std::endl;
    }

#ifndef WIN32
    // The signalHandler is there to make sure that video.release() is called
    // when doing a CTRL+C
    // Otherwise, the webcam can be left in corrupted state and may not work
    // afterward,
    // requiring a reset of the USB ports (at least on OpenCV 2.0.0)
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signalHandler;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
#endif

    unsigned int frameId = 0;

    imgView = img.clone();
    std::thread viewLoopThread(viewLoop);
    viewLoopThread.detach();

    while (true) {
        std::chrono::high_resolution_clock::time_point startTime
            = std::chrono::high_resolution_clock::now();

        if (videoFileName.empty()) {
            captureLock.lock();
            img = frame.clone();
            captureLock.unlock();

            if (!img.data)
                continue;
        }

        deepNet->getStimuliProvider()->streamStimulus(img, Database::Test);
        deepNet->test(Database::Test, &timings);

        const cv::Mat imgOrg = img.clone();
        const cv::Rect logoArea
            = cv::Rect(0, 0, N2D2_LIST_LOGO.cols, N2D2_LIST_LOGO.rows);
        cv::Mat imgLogoArea = img(logoArea);
        cv::addWeighted(
            imgLogoArea, 0.25, N2D2_LIST_LOGO, 0.75, 0.0, imgLogoArea);

        const cv::Mat imgLogo = img.clone();
        process(frameId,
                imgOrg,
                img,
                deepNet,
                deepNetSub,
                target,
                targetCls,
                targetClsName);

        std::chrono::high_resolution_clock::time_point curTime
            = std::chrono::high_resolution_clock::now();
        const double timeElapsed
            = std::chrono::duration_cast
              <std::chrono::duration<double> >(curTime - startTime).count();

        cv::Mat estimated = deepNet->getTarget<TargetROIs>(target)
                                ->drawEstimatedLabels();

        std::stringstream fpsStr;
        fpsStr << std::fixed << std::setprecision(2) << (1.0 / timeElapsed)
               << " fps";

        cv::putText(estimated,
                    fpsStr.str(),
                    cv::Point(estimated.cols - 100, estimated.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1,
                    CV_AA);

        if (videoWriter.isOpened())
            videoWriter << img;

        viewLock.lock();
        imgView = img.clone();
        estimatedView = estimated.clone();
        timingsView = drawTimings(timings);
        viewLock.unlock();

        if (!videoFileName.empty()) {
            if (!video.grab() || !video.retrieve(img))
                break;
        }

        if (quit) {
            std::cout << "Terminating..." << std::endl;
            break;
        }

        ++frameId;
    }

    if (videoFileName.empty()) {
        captureLock.lock();
        captureFlag = false;
        captureLock.unlock();
        captureThread.join();
    }

    video.release();

    return 0;
}
