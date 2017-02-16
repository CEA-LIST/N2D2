#!/usr/bin/python -u
# -*- coding: ISO-8859-1 -*-
################################################################################
#    (C) Copyright 2016 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

import glob
import cv2
import os, errno
import re
import numpy

from python import TargetViewer

class Viewer(TargetViewer.TargetViewer):
    def __init__(self):
        path = os.getcwd()

        if os.path.basename(__file__) != "target_viewer.py":
            path = os.path.join(path,
                os.path.splitext(os.path.basename(__file__))[0])

        super(Viewer, self).__init__(path,
            r'_estimated\.[^.]*(?<!log)$')

        self.estimatedWindow = ""
        self.targetWindow = ""
        self.imgEstimated = []
        self.imgTarget = []
        self.imgLegend = cv2.imread(os.path.join(os.getcwd(),
            "labels_legend.png"));
        self.imgEstimatedHsv = []
        self.imgTargetHsv = []

        if self.imgLegend is not None:
            imgLegendHsv = cv2.cvtColor(self.imgLegend, cv2.COLOR_BGR2HSV)
            self.labelsHueOffset = imgLegendHsv[self.cellHeight/2,
                self.cellHeight/2][0]

    # PRIVATE
    def _run(self):
        cv2.destroyWindow(self.estimatedWindow)
        cv2.destroyWindow(self.targetWindow)

        estimatedName = self.files[self.index]
        targetName = self._replace_last_of(estimatedName,
            "_estimated.", "_target.")
        self.estimatedWindow = os.path.basename(estimatedName)
        self.targetWindow = os.path.basename(targetName)
        cv2.namedWindow(self.estimatedWindow, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.estimatedWindow, self._mouseCallback, True)
        cv2.namedWindow(self.targetWindow, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.targetWindow, self._mouseCallback, False)

        self.imgTarget = cv2.imread(targetName);
        self.imgTargetHsv = cv2.cvtColor(self.imgTarget, cv2.COLOR_BGR2HSV)
        cv2.imshow(self.targetWindow, self.imgTarget)
        cv2.resizeWindow(self.targetWindow, 1024, 512)
        cv2.moveWindow(self.targetWindow, 0, 0)

        # A target image is skippable if it doesn't contain any annotation
        # We check that the saturation is below a threshold for every pixels,
        # as annotations have a 100% saturation (hue is not reliable)
        skippable = False
        if numpy.all([v < 20 for v in self.imgTargetHsv[:,:,1]]):
            skippable = True

        self.imgEstimated = cv2.imread(estimatedName);
        self.imgEstimatedHsv = cv2.cvtColor(self.imgEstimated,
            cv2.COLOR_BGR2HSV)
        cv2.imshow(self.estimatedWindow, self.imgEstimated)
        cv2.resizeWindow(self.estimatedWindow, 1024, 512)
        cv2.moveWindow(self.estimatedWindow, 0, 512 + 10)

        return (os.path.basename(self._replace_last_of(estimatedName,
            "_estimated", "")), skippable)

    def _mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            newImgEstimated = self.imgEstimated.copy()
            newImgTarget = self.imgTarget.copy()
            height, width = newImgEstimated.shape[:2]

            cv2.line(newImgEstimated, (x, 0), (x, height), (255, 255, 255))
            cv2.line(newImgEstimated, (0, y), (width, y), (255, 255, 255))
            cv2.imshow(self.estimatedWindow, newImgEstimated)

            cv2.line(newImgTarget, (x, 0), (x, height), (255, 255, 255))
            cv2.line(newImgTarget, (0, y), (width, y), (255, 255, 255))
            cv2.imshow(self.targetWindow, newImgTarget)
        elif event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            legendHeight, legendWidth = self.imgLegend.shape[:2]
            nbTargets = legendHeight/self.cellHeight

            target = int(round(((self.imgTargetHsv[y,x][0]
                + 180 - self.labelsHueOffset) % 180)*nbTargets/180.0))
            estimated = int(round(((self.imgEstimatedHsv[y,x][0]
                + 180 - self.labelsHueOffset) % 180)*nbTargets/180.0))
            score = self.imgEstimatedHsv[y,x][2]/255.0

            print "  (%d,%d): target = %d / estimated = %d (score = %.02f)" % (
                x, y, target, estimated, score)

            if self.imgLegend is not None:
                newImgLegend = self.imgLegend.copy()

                if target == estimated:
                    cv2.rectangle(newImgLegend,
                        (0, target*self.cellHeight),
                        (legendWidth, (target+1)*self.cellHeight),
                        (0, 255, 0), 5)
                else:
                    cv2.rectangle(newImgLegend,
                        (0, target*self.cellHeight),
                        (legendWidth, (target+1)*self.cellHeight),
                        (255, 0, 0), 5)
                    cv2.rectangle(newImgLegend,
                        (0, estimated*self.cellHeight),
                        (legendWidth, (estimated+1)*self.cellHeight),
                        (0, 0, 255), 5)

                cv2.imshow("legend", newImgLegend)

            if event == cv2.EVENT_RBUTTONDOWN:
                try:
                    os.makedirs("capture")
                except OSError, exc:
                    if exc.errno == errno.EEXIST:
                        pass
                    else: raise

                newImgEstimated = self.imgEstimated.copy()
                height, width = newImgEstimated.shape[:2]

                if nbTargets > 2 and self.imgLegend is not None:
                    labelWidth = min(100, width)
                    labelHeight = self.cellHeight*labelWidth/legendWidth
                    imgLabel = self.imgLegend[
                        estimated*self.cellHeight:(estimated+1)*self.cellHeight,
                        0:legendWidth]
                    imgLabel = cv2.resize(imgLabel, (labelWidth, labelHeight),
                        interpolation=cv2.INTER_AREA)

                    newImgEstimated[height-labelHeight:height,
                        width-labelWidth:width] = imgLabel

                cv2.line(newImgEstimated,
                    (x, max(0, y-5)),
                    (x, min(height, y+5)), (255, 255, 255))
                cv2.line(newImgEstimated,
                    (max(0, x-5), y),
                    (min(width, x+5), y), (255, 255, 255))
                cv2.imwrite(os.path.join("capture",
                    self.estimatedWindow + ".png"), newImgEstimated)

viewer = Viewer()
viewer.run()
