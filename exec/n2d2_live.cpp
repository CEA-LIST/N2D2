/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#define CAPTURE_WIDTH 1280
#define CAPTURE_HEIGHT 720
#define TOPN 1

bool quit = false;    // signal flag

void signalHandler(int) {
    quit = true;
}
std::mutex captureLock;
bool captureFlag = true;

void capture(cv::VideoCapture& video, cv::Mat& frame) {
    while (true) {
        captureLock.lock();
        video >> frame;
        const bool flag = captureFlag;
        captureLock.unlock();

        if (!flag)
            return;

#ifdef WIN32
        Sleep(10);    // ms
#else
        usleep(100000);
#endif
    }
}
std::mutex viewLock;
cv::Mat imgView;
const std::string frameWindow = "DEMO - LIST - , Institute of CEA Tech";

void viewLoop() {
    while (true) {
        viewLock.lock();

        if (imgView.data)
            cv::imshow(frameWindow.c_str(), imgView);

        viewLock.unlock();

        int k = cv::waitKey(1);

        if (k == KEY_ESC) {
            quit = true;
            break;
        }
    }
}

cv::Mat inputPicture;
int main(int argc, char *argv[]) {
    // Program command line options
    ProgramOptions opts(argc, argv);
#ifdef CUDA
    const int cudaDevice
        = opts.parse("-dev", 0,              "CUDA device ID");
#endif
    const bool noDisplay
        = opts.parse("-no-display",
                     "disable display and visual feedbacks");
    const std::string pathToSave
        = opts.parse<std::string>("-save",
                                  "",
                                  "save captured picture to a specific path");
    const std::string pictureFileName
        = opts.parse<std::string>("-picture",
                                  "",
                                  "test the network on a specific picture");
    const std::string importedWeights
        = opts.parse<std::string>("-w",
                                  "",
                                  "import specific weight");
    const std::string iniConfig
        = opts.grab<std::string>("<net>",
                                 "network config file (INI)");
    opts.done();

#ifdef CUDA
    CudaContext::setDevice(cudaDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    //timingsWindow+= std::string(" on ") + prop.name;
#endif
    //Network Initialisation
    Network net;
    std::shared_ptr<DeepNet> deepNet
        = DeepNetGenerator::generate(net, iniConfig);

    deepNet->initialize();

    if(!importedWeights.empty())
        deepNet->importNetworkFreeParameters(importedWeights);
    else
        deepNet->importNetworkFreeParameters("weights_validation");

    //Video Initialisation
    cv::VideoCapture video(0); // open the default camera
    if(!video.isOpened())  // check if we succeeded
        return -1;
    video.set(CV_CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH);
    video.set(CV_CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT);

    cv::Mat frame;

    std::thread captureThread;
    captureThread = std::thread(capture, std::ref(video), std::ref(frame));

    if(!noDisplay){
        cv::namedWindow(frameWindow.c_str(), CV_WINDOW_NORMAL);
        cvMoveWindow(frameWindow.c_str(), 0, 0);
        cvResizeWindow(frameWindow.c_str(), DISPLAY_WIDTH, DISPLAY_HEIGHT);

    }
    cv::Size display_size(DISPLAY_WIDTH,DISPLAY_HEIGHT);


    cv::Mat img;
    cv::Mat img_display;
    cv::Mat img_save;
    std::fstream labelsFileSaved;

    std::thread viewLoopThread(viewLoop);
    viewLoopThread.detach();


    for(;;)
    {
        std::chrono::high_resolution_clock::time_point startTime
            = std::chrono::high_resolution_clock::now();

        captureLock.lock();
        img = frame.clone();
        captureLock.unlock();

        if (!img.data)
            continue;

        deepNet
             ->getStimuliProvider()
                 ->streamStimulus(img, Database::Test);

        deepNet->test(Database::Test);

        std::shared_ptr<TargetScore> targetPicture
            = deepNet->getTarget<TargetScore>();

        const Tensor3d<int> estimatedLabel
            = targetPicture->getEstimatedLabels()[0];
        const Tensor3d<Float_T> estimatedLabelValue
            = targetPicture->getEstimatedLabelsValue()[0];


        std::chrono::high_resolution_clock::time_point curTime
            = std::chrono::high_resolution_clock::now();

        const double timeElapsed
            = std::chrono::duration_cast<std::chrono::duration<double> >
                (curTime - startTime).count();

        std::stringstream fpsStr;

        fpsStr << std::fixed << std::setprecision(2)
            << (1.0/timeElapsed) << " fps";

        if(!pathToSave.empty()) {
            labelsFileSaved.open(pathToSave +
                                 "output_labels.txt",
                                 std::fstream::out);

            cv::resize(img, img_save, display_size);
            std::vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            try {
                cv::imwrite(pathToSave +  "output_picture.png",
                            img_save,
                            compression_params);
            }
            catch (std::runtime_error& ex) {
                fprintf(stderr,
                        "Exception converting image to PNG format: %s\n",
                        ex.what());
            }
            std::stringstream labels_save;
            for (unsigned int i = 0, size = TOPN; i < size; ++i) {
                std::string displayEstimatedName
                = deepNet->getDatabase()->getLabelName(estimatedLabel(i));

                double displayEstimatedValue = estimatedLabelValue(i);
                labels_save << displayEstimatedName
                    << ": "
                    << displayEstimatedValue
                    << "\n";
            }
            labelsFileSaved << labels_save.str();
            labelsFileSaved.close();
        }

        if(!noDisplay) {
            cv::resize(img, img_display, display_size);

            cv::putText(img_display,
                        fpsStr.str(),
                        cv::Point(img_display.cols-100, img_display.rows-10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(255, 0, 0),
                        1,
                        CV_AA);

            unsigned int width = 240; unsigned int height = 180;
            const unsigned int margin = 2;

            const unsigned int labelWidth
                = std::min((unsigned int)120, width/2);

            const unsigned int cellHeight = height / 5;

            cv::Mat mat(cv::Size(width, height),
                        CV_8UC3,
                        cv::Scalar(255, 255, 255));

            for (unsigned int i = 0, size = TOPN; i < size; ++i) {

                std::string displayEstimatedName
                    = deepNet
                        ->getDatabase()
                            ->getLabelName(estimatedLabel(i));
                double displayEstimatedValue = estimatedLabelValue(i);

                int baseline = 0;
                cv::Size textSize = cv::getTextSize(displayEstimatedName,
                                                    cv::FONT_HERSHEY_SIMPLEX,
                                                    0.70,
                                                    1,
                                                    &baseline);

                cv::putText(img_display,
                            displayEstimatedName,
                            cv::Point(labelWidth + margin
                                      + displayEstimatedValue*(width -
                                                               labelWidth -
                                                               2.0*margin),
                            (i+1)*cellHeight
                                      - (cellHeight
                                         - textSize.height)/2.0),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.70,
                            cv::Scalar(0, 0, 255),
                            1,
                            CV_AA);

                cv::rectangle(img_display,
                              cv::Point(margin, i*cellHeight +
                                        margin +
                                        cellHeight/2),
                              cv::Point(labelWidth + margin +
                                        displayEstimatedValue*(width
                                                     - labelWidth
                                                     - 2.0*margin),
                              i*cellHeight - margin + cellHeight/2),
                              cv::Scalar(255, 255, 0), CV_FILLED);


                std::stringstream valueStr;
                valueStr << std::fixed << std::setprecision(2)
                    << (100.0*displayEstimatedValue) << "%";

                textSize = cv::getTextSize(valueStr.str(),
                                           cv::FONT_HERSHEY_SIMPLEX,
                                           0.35,
                                           1,
                                           &baseline);

                cv::putText(img_display,
                            valueStr.str(),
                            cv::Point( 2*margin, (i + 1)*cellHeight -
                                      (cellHeight - textSize.height)/2.0 +
                                      cellHeight/2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.70,
                    cv::Scalar(255, 255, 255),
                            1,
                            CV_AA);
            }
        }

        std::cout << "frame per second: " << fpsStr.str() << std::endl;

        if(!noDisplay) {
            viewLock.lock();
            imgView = img_display.clone();
            viewLock.unlock();
        }

        if (quit) {
            std::cout << "Terminating..." << std::endl;
            break;
        }

    }


    captureLock.lock();
    captureFlag = false;
    captureLock.unlock();
    captureThread.join();

    video.release();

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;

}

