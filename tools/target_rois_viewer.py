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
import os
import re

from python import TargetViewer

class RoisViewer(TargetViewer.TargetViewer):
    def __init__(self):
        path = os.getcwd()

        if os.path.basename(__file__) != "target_rois_viewer.py":
            path = os.path.join(path,
                os.path.splitext(os.path.basename(__file__))[0])

        super(RoisViewer, self).__init__(os.path.join(path,
            "*[!_estimated].[!log]*"))

        self.roiWindow = ""
        self.estimatedWindow = ""
        self.imgRoi = []
        self.imgEstimated = []
        self.imgEstimatedHsv = []
        self.estimatedPath = None

        if os.path.isfile("target_rois_label.dat"):
            ROIsLabelTarget = open("target_rois_label.dat", 'r').read().strip()

            self.estimatedPath = os.path.dirname(os.path.dirname(path))
            self.estimatedPath = os.path.join(self.estimatedPath,
                ROIsLabelTarget)
            self.estimatedPath = os.path.join(self.estimatedPath,
                os.path.basename(path))
        else:
            self.estimatedPath = path

        if self.estimatedPath is not None:
            self.imgLegend = cv2.imread(os.path.join(
                os.path.dirname(self.estimatedPath), "labels_legend.png"));

            if self.imgLegend is not None:
                imgLegendHsv = cv2.cvtColor(self.imgLegend, cv2.COLOR_BGR2HSV)
                self.labelsHueOffset = imgLegendHsv[self.cellHeight/2,
                    self.cellHeight/2][0]

    # PRIVATE
    def _run(self):
        roiName = self.files[self.index]
        self.roiWindow = os.path.basename(roiName)
        cv2.namedWindow(self.roiWindow, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.roiWindow, self._mouseCallback, True)

        self.imgRoi = cv2.imread(roiName);
        cv2.imshow(self.roiWindow, self.imgRoi)
        cv2.resizeWindow(self.roiWindow, 1024, 512)
        cv2.moveWindow(self.roiWindow, 0, 0)

        frameName = os.path.basename(self.files[self.index])

        if self.estimatedPath is not None:
            estimatedName = os.path.join(self.estimatedPath,
                frameName.replace(".", "_estimated."))
            self.estimatedWindow = os.path.basename(estimatedName)
            cv2.namedWindow(self.estimatedWindow, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.estimatedWindow,
                self._mouseCallback, False)

            self.imgEstimated = cv2.imread(estimatedName);
            self.imgEstimatedHsv = cv2.cvtColor(self.imgEstimated,
                cv2.COLOR_BGR2HSV)
            cv2.imshow(self.estimatedWindow, self.imgEstimated)
            cv2.resizeWindow(self.estimatedWindow, 1024, 512)
            cv2.moveWindow(self.estimatedWindow, 0, 512 + 50)

        return frameName

    def _mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            newImgRoi = self.imgRoi.copy()
            height, width = self.imgRoi.shape[:2]

            cv2.line(newImgRoi, (x, 0), (x, height), (255, 255, 255))
            cv2.line(newImgRoi, (0, y), (width, y), (255, 255, 255))
            cv2.imshow(self.roiWindow, newImgRoi)

            if self.estimatedPath is not None:
                newImgEstimated = self.imgEstimated.copy()
                cv2.line(newImgEstimated, (x, 0), (x, height), (255, 255, 255))
                cv2.line(newImgEstimated, (0, y), (width, y), (255, 255, 255))
                cv2.imshow(self.estimatedWindow, newImgEstimated)
        elif event == cv2.EVENT_LBUTTONDOWN and self.estimatedPath is not None:
            legendHeight, legendWidth = self.imgLegend.shape[:2]
            nbTargets = legendHeight/self.cellHeight

            estimated = int(round(((self.imgEstimatedHsv[y,x][0]
                + 180 - self.labelsHueOffset) % 180)*nbTargets/180.0))
            score = self.imgEstimatedHsv[y,x][2]/255.0

            print "  (%d,%d): estimated = %d (score = %.02f)" % (
                x, y, estimated, score)

            if self.imgLegend is not None:
                newImgLegend = self.imgLegend.copy()
                cv2.rectangle(newImgLegend,
                    (0, estimated*self.cellHeight),
                    (legendWidth, (estimated+1)*self.cellHeight),
                    (255, 255, 255), 5)
                cv2.imshow("legend", newImgLegend)

viewer = RoisViewer()
viewer.run()
