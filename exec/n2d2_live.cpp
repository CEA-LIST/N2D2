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

#ifdef MONGODB
#include "mongo/client/dbclient.h"
#endif

#include "n2d2_list_logo.hpp"

using namespace N2D2;

#define TARGET_FACE 0
#define TARGET_FACE_ROI 2
#define TARGET_GENDER 1

#define DISPLAY_WIDTH 1280
#define DISPLAY_HEIGHT 720

// Distance estimator calibration
#define DIST_ENABLE // display estimated distance information
#define DIST_CAL_SLOPE -0.017 // real distance vs ROIs height slope
#define DIST_CAL_INTERCEPT 3.55 // real distance vs ROIs height intercept
#define DIST_CAL_MIN 1.0 // min. distance to display (in m)

const std::string frameWindow = "FACE DEMO - LIST, Institute of CEA Tech";
const std::string labelsWindow = "Labels";
std::string timingsWindow = "Timings";

std::mutex captureLock;
bool captureFlag = true;
#ifdef MONGODB
mongo::DBClientConnection conn;
#endif

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
                     bool noDisplay,
                     bool saveRoiImg,
                     bool saveRoiHist)
{
    std::cout << "Processing frame #" << frameId << std::endl;

    std::shared_ptr<TargetROIs> targetFace = deepNet->getTarget
                                             <TargetROIs>(TARGET_FACE_ROI);
    const std::vector<TargetROIs::DetectedBB> detectedBB
        = targetFace->getDetectedBB();
    std::vector<std::shared_ptr<ROI> > estimatedLabelsROIs;

    for (unsigned int id = 0, size = detectedBB.size(); id < size; ++id)
        estimatedLabelsROIs.push_back(detectedBB[id].bb->clone());

    Tensor2d<int> estimatedLabels;
    targetFace->getStimuliProvider()->reverseLabels(
        img, Database::Test, estimatedLabels, estimatedLabelsROIs);

#ifdef MONGODB
    const mongo::Date_t now
        = mongo::Date_t(std::chrono::system_clock::now().time_since_epoch()
                        / std::chrono::milliseconds(1));
#endif

    const double fontScale = 1.5 * imgOrg.cols / 1920.0;

    for (unsigned int id = 0, size = detectedBB.size(); id < size; ++id) {
        // BB detection data
        const cv::Rect rect = estimatedLabelsROIs[id]->getBoundingRect();

#ifdef MONGODB
        mongo::BSONObjBuilder row;
        row.append("frame_id", frameId);
        row.append("id", id);
        row.append("time", now);
        row.append("x0", rect.x);
        row.append("y0", rect.y);
        row.append("width", rect.width);
        row.append("height", rect.height);
        row.append("score", detectedBB[id].score);

        std::cout << "  Face detected #" << id << ": " << rect.width << "x"
                  << rect.height << "@" << rect.x << "," << rect.y << "("
                  << detectedBB[id].score << ")" << std::endl;

        const cv::Mat data = (saveRoiImg || saveRoiHist)
                                 ? estimatedLabelsROIs[id]->extract(imgOrg)
                                 : cv::Mat();

        if (saveRoiImg) {
            assert(data.channels() == imgOrg.channels());
            assert(data.type() == CV_8UC3);

            std::vector<unsigned char> binaryData;

            if (!cv::imencode(".png", data, binaryData))
                throw std::runtime_error(
                    "cv::imencode(): unable to encode frame data.");

            row.appendBinData("image",
                              binaryData.size(),
                              mongo::BinDataGeneral,
                              &binaryData[0]);
            // row.appendBinData("image", data.elemSize()*data.rows*data.cols,
            // mongo::BinDataGeneral, data.data);
        }

        if (saveRoiHist) {
            // Histogram computation
            std::vector<cv::Mat> bgrPlanes;
            cv::split(data, bgrPlanes);

            int channels[] = {0};
            const int histSize = 256; // from 0 to 255
            /// Set the ranges ( for B,G,R) )
            float range[] = {0, 256}; // the upper boundary is exclusive
            const float* histRange = {range};
            const bool uniform = true;
            const bool accumulate = false;

            std::vector<cv::MatND> histPlanes(3);
            cv::calcHist(&bgrPlanes[0],
                         1,
                         channels,
                         cv::Mat(),
                         histPlanes[0],
                         1,
                         &histSize,
                         &histRange,
                         uniform,
                         accumulate);
            cv::calcHist(&bgrPlanes[1],
                         1,
                         channels,
                         cv::Mat(),
                         histPlanes[1],
                         1,
                         &histSize,
                         &histRange,
                         uniform,
                         accumulate);
            cv::calcHist(&bgrPlanes[2],
                         1,
                         channels,
                         cv::Mat(),
                         histPlanes[2],
                         1,
                         &histSize,
                         &histRange,
                         uniform,
                         accumulate);

            cv::MatND histND;
            cv::merge(histPlanes, histND);
            const cv::Mat hist(histND);

            assert(hist.cols == 1);
            assert(hist.rows == histSize);
            assert(hist.channels() == 3);
            assert(hist.elemSize() == 12);
            assert(hist.type() == CV_32FC3);

            cv::Mat hist16;
            hist.convertTo(hist16, CV_16U);
            assert(hist.channels() == 3);
            assert(hist.elemSize() == 6);
            assert(hist.type() == CV_16UC3);

            std::vector<unsigned char> binaryData;

            if (!cv::imencode(".png", hist16, binaryData))
                throw std::runtime_error(
                    "cv::imencode(): unable to encode frame data.");

            row.appendBinData("image_hist",
                              binaryData.size(),
                              mongo::BinDataGeneral,
                              &binaryData[0]);
            // row.appendBinData("image_hist",
            // hist.elemSize()*hist.rows*hist.cols, mongo::BinDataGeneral,
            // hist.data);
        }
#else
        // Dummy if (avoid warning)
        if (saveRoiImg) {
        }
        if (saveRoiHist) {
        }
#endif

// BB classification data
#ifdef TARGET_GENDER
        int gender;
        Float_T genderScore;

        std::shared_ptr<Target> targetGender = deepNet->getTarget
                                               <Target>(TARGET_GENDER);
        std::tie(gender, genderScore)
            = targetGender->getEstimatedLabel(detectedBB[id].bb);

        const std::string genderLabel = (gender == 0) ? "male" : "female";

        if (!noDisplay) {
            std::stringstream genderLabelStr;
            genderLabelStr << genderLabel << ": " << std::setprecision(3)
                           << (genderScore * 100.0) << "%";

            cv::putText(img,
                        genderLabelStr.str(),
                        cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        cv::Scalar(0, 0, 255),
                        1,
                        CV_AA);
        }

#ifdef MONGODB
        row.append("gender", genderLabel);
        row.append("gender_score", genderScore);
#endif

        std::cout << "  |-Gender: " << genderLabel << "(" << genderScore << ")"
                  << std::endl;
#endif

#ifdef MONGODB
        row.append("age_1", "");
        row.append("age_1_score", 0.0);
        row.append("age_2", "");
        row.append("age_2_score", 0.0);
        row.append("age_3", "");
        row.append("age_3_score", 0.0);
#endif

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

            // DEBUG
            // std::ostringstream fileName;
            // fileName << "subframe_" << id << ".dat";
            // StimuliProvider::logData(fileName.str(),
            // deepNetSub->getStimuliProvider()->getData()[0]);

            std::shared_ptr<TargetScore> targetEmotion = deepNetSub->getTarget
                                                         <TargetScore>();
            const Tensor3d<int> estimatedLabel
                = targetEmotion->getEstimatedLabels()[0];
            const Tensor3d<Float_T> estimatedLabelValue
                = targetEmotion->getEstimatedLabelsValue()[0];

            // cv::rectangle(img, faceRect.tl(), faceRect.br(), cv::Scalar(0, 0,
            // 255));

            std::cout << "  |-Emotion(s):";

            for (unsigned int n = 0; n < 3; ++n) {
                std::stringstream nStr;
                nStr << (n + 1);

                const std::string estimatedLabelName
                    = deepNetSub->getDatabase()->getLabelName(
                        estimatedLabel(n));

                if (!noDisplay) {
                    std::stringstream labelStr;
                    labelStr << estimatedLabelName << ": "
                             << std::setprecision(3)
                             << (estimatedLabelValue(n) * 100.0) << "%";

                    const cv::Size textSize
                        = cv::getTextSize(labelStr.str(),
                                          cv::FONT_HERSHEY_SIMPLEX,
                                          fontScale,
                                          1,
                                          &baseline);
                    cv::putText(img,
                                labelStr.str(),
                                cv::Point(rect.x + 5, rect.y + textOffset + 5),
                                cv::FONT_HERSHEY_SIMPLEX,
                                fontScale,
                                cv::Scalar(0, 255, 0),
                                1,
                                CV_AA);
                    textOffset += textSize.height + 5;
                }

#ifdef MONGODB
                row.append("emotion_" + nStr.str(), estimatedLabelName);
                row.append("emotion_" + nStr.str() + "_score",
                           estimatedLabelValue(n));
#endif

                std::cout << " " << estimatedLabelName << "("
                          << estimatedLabelValue(n) << ")";
            }

            std::cout << std::endl;
        }

        if (!noDisplay) {
#ifdef DIST_ENABLE
            const double estimatedDistance = DIST_CAL_SLOPE * rect.height
                                             + DIST_CAL_INTERCEPT;

            std::stringstream distanceStr;

            if (estimatedDistance < DIST_CAL_MIN)
                distanceStr << "dist.<" << DIST_CAL_MIN << "m";
            else
                distanceStr << "dist.~" << std::setprecision(3)
                            << estimatedDistance << "m";

            cv::putText(img,
                        distanceStr.str(),
                        cv::Point(rect.x + 5, rect.y + textOffset + 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        cv::Scalar(255, 255, 0),
                        1,
                        CV_AA);
#endif

            estimatedLabelsROIs[id]->draw(img);
        }

#ifdef MONGODB
        conn.insert("cea-ds.detections", row.obj());

        const std::string lastError = conn.getLastError();
        if (!lastError.empty())
            std::cout << Utils::cwarning
                      << "Warning: MongoDB insertion failed: " << lastError
                      << Utils::cdef << std::endl;
#endif
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
    const bool noDisplay
        = opts.parse("-no-display", "disable display and visual feedbacks");
#if defined(MONGODB)
    const bool saveFrames
        = opts.parse("-save-frames", "save captured frames in MongoDB");
#endif
    const bool saveRoiImg
        = opts.parse("-save-roi-img", "save detected ROIs image in MongoDB");
    const bool saveRoiHist = opts.parse(
        "-save-roi-hist", "save detected ROIs image histogram in MongoDB");
    const std::string videoFileName
        = opts.parse<std::string>("-video", "", "run on a video file");
    const std::string recordFileName = opts.parse<std::string>(
        "-record", "", "record the display to a video file");
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

    if (!noDisplay) {
        cv::namedWindow(frameWindow.c_str(), CV_WINDOW_NORMAL);
        cv::namedWindow(labelsWindow.c_str(), CV_WINDOW_AUTOSIZE);
        cv::namedWindow(timingsWindow.c_str(), CV_WINDOW_AUTOSIZE);
        cvMoveWindow(frameWindow.c_str(), 0, 0);
        cvResizeWindow(frameWindow.c_str(), DISPLAY_WIDTH, DISPLAY_HEIGHT);
        cvMoveWindow(labelsWindow.c_str(), DISPLAY_WIDTH + 50, 0);
        cvMoveWindow(timingsWindow.c_str(), DISPLAY_WIDTH + 50, 400 + 50);
    }

    // Network topology construction
    Network net;
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, iniConfig);
    deepNet->initialize();
    deepNet->importNetworkFreeParameters("weights");

    // DrawNet::draw(*deepNet, "deepnet.svg");
    // deepNet->logStats("stats");

    std::shared_ptr<DeepNet> deepNetSub;

    if (!iniConfigSub.empty()) {
        deepNetSub = DeepNetGenerator::generate(net, iniConfigSub);
        deepNetSub->initialize();
        deepNetSub->importNetworkFreeParameters("weights-sub");

        // DrawNet::draw(*deepNetSub, "deepnet-sub.svg");
        // deepNetSub->logStats("stats-sub");
    }

    cv::Mat frame;
    std::thread captureThread;

    if (videoFileName.empty())
        captureThread = std::thread(capture, std::ref(video), std::ref(frame));

    std::vector<std::pair<std::string, double> > timings;

#ifdef MONGODB
    try
    {
        const char* server = std::getenv("N2D2_MONGODB_SERVER");

        if (server != NULL) {
            conn.connect(server);
            std::cout << "Connected to MongoDB on " << server << std::endl;
        } else {
            conn.connect("localhost");
            std::cout << "Connected to MongoDB on localhost" << std::endl;
        }
    }
    catch (const mongo::DBException& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
#endif

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

    if (!noDisplay) {
        imgView = img.clone();
        std::thread viewLoopThread(viewLoop);
        viewLoopThread.detach();
    }

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
                noDisplay,
                saveRoiImg,
                saveRoiHist);

#if defined(MONGODB)
        if (saveFrames) {
            mongo::BSONObjBuilder row;
            row.append("frame_id", frameId);

            std::vector<unsigned char> binaryData;

            if (!cv::imencode(".jpg", imgLogo, binaryData))
                throw std::runtime_error(
                    "cv::imencode(): unable to encode frame data.");

            row.appendBinData("data",
                              binaryData.size(),
                              mongo::BinDataGeneral,
                              &binaryData[0]);
            conn.insert("cea-ds.framesQueue", row.obj());

            const std::string lastError = conn.getLastError();
            if (!lastError.empty())
                std::cout << Utils::cwarning
                          << "Warning: MongoDB insertion failed: " << lastError
                          << Utils::cdef << std::endl;
        }
#endif

        std::chrono::high_resolution_clock::time_point curTime
            = std::chrono::high_resolution_clock::now();
        const double timeElapsed
            = std::chrono::duration_cast
              <std::chrono::duration<double> >(curTime - startTime).count();

        if (!noDisplay) {
            cv::Mat estimated = deepNet->getTarget<TargetROIs>(TARGET_FACE_ROI)
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
            /*
                        cv::imshow(frameWindow.c_str(), img);
                        cv::imshow(labelsWindow.c_str(), estimated);
                        cv::imshow(timingsWindow.c_str(), drawTimings(timings));

                        int k = cv::waitKey(1);

                        if (k == KEY_ESC) {
                            std::cout << "Terminating..." << std::endl;
                            break;
                        }
            */
        }

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
