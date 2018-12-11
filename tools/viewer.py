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

        if os.path.basename(__file__) != "viewer.py":
            path = os.path.join(path,
                os.path.splitext(os.path.basename(__file__))[0])

        super(Viewer, self).__init__(path, r'\.[^.]*(?<!log)$')

        self.targetWindow = ""
        self.imgLegend = None
        self.imgTarget = []

    # PRIVATE
    def _run(self):
        if self.targetWindow != "":
            cv2.destroyWindow(self.targetWindow)

        targetName = self.files[self.index]
        self.targetWindow = os.path.basename(targetName)
        cv2.namedWindow(self.targetWindow, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.targetWindow, self._mouseCallback, False)

        self.imgTarget = cv2.imread(targetName);
        cv2.imshow(self.targetWindow, self.imgTarget)
        cv2.resizeWindow(self.targetWindow, 1024, 512)
        cv2.moveWindow(self.targetWindow, 0, 0)

        return (os.path.basename(targetName), False)

    def _mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            newImgTarget = self.imgTarget.copy()
            height, width = newImgTarget.shape[:2]

            cv2.line(newImgTarget, (x, 0), (x, height), (255, 255, 255))
            cv2.line(newImgTarget, (0, y), (width, y), (255, 255, 255))
            cv2.imshow(self.targetWindow, newImgTarget)
        elif event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            print "  (%d,%d): %s" % (x, y, self.imgTarget[y,x])

viewer = Viewer()
viewer.run()
