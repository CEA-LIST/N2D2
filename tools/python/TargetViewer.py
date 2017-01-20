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
import re
import numpy

class TargetViewer(object):
    def __init__(self, path):
        self.files = glob.glob(path)
        self.files.sort(key=self._naturalKeys)
        self.index = -1

        # Same as in N2D2::Target::logLabelsLegend()
        self.cellHeight = 50
        self.imgLegend = []

    def run(self):
        newIndex = 0

        while (True):
            if newIndex != self.index:
                self.index = newIndex
                cv2.destroyAllWindows()

                # Determine slice position --
                # Note that this code only works if slices are correctly
                # ordered, using "sort" with "natural_keys"
                gridX = 1
                gridY = 1
                offsetX = 0
                offsetY = 0
                i = self.index
                regexSlice = r'\[([0-9]+),([0-9]+)\]$'
                stimulusName = re.sub(regexSlice, '', self.files[i])

                # |- Rewind to the first slice index for the current original
                # image file
                while i > 0:
                    if re.sub(regexSlice, '', self.files[i-1]) == stimulusName:
                        i-= 1
                    else:
                        break

                # |- Find grid size and max offsets
                while i < len(self.files):
                    if re.sub(regexSlice, '', self.files[i]) != stimulusName:
                        break

                    m = re.search(regexSlice, self.files[i])

                    if m:
                        if int(m.group(2)) > offsetX:
                            offsetX = int(m.group(2))
                            gridX+= 1

                        if int(m.group(1)) > offsetY:
                            offsetY = int(m.group(1))
                            gridY+= 1

                        i+= 1
                    else:
                        break

                # |- Find and display current slice position
                m = re.search(regexSlice, self.files[self.index])

                if m:
                    sliceX = offsetX/max(gridX-1, 1)
                    sliceY = offsetY/max(gridY-1, 1)
                    posX = int(m.group(2))/max(sliceX, 1)
                    posY = int(m.group(1))/max(sliceY, 1)

                    imgSlicing = numpy.zeros((50*gridY, 50*gridX, 1),
                        dtype = "uint8")
                    imgSlicing[50*posY:50*(posY+1), 50*posX:50*(posX+1)] = 255

                    for x in xrange(0,gridX):
                        for y in xrange(0,gridY):
                            color = (255)
                            if x == posX and y == posY:
                                color = (0)
                            cv2.putText(imgSlicing,
                                "%d,%d"% (y*sliceY, x*sliceX),
                                (50*x + 5, 50*y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, color)

                    cv2.namedWindow("slicing", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("slicing", 50*gridX, 50*gridY)
                    cv2.moveWindow("slicing", 1024 + 256 + 20, 0)
                    cv2.imshow("slicing", imgSlicing)
                # -- Determine slice position

                if self.imgLegend is not None:
                    cv2.namedWindow("legend", cv2.WINDOW_NORMAL)
                    cv2.imshow("legend", self.imgLegend)
                    cv2.resizeWindow("legend", 256, 1024)
                    cv2.moveWindow("legend", 1024 + 10, 0)

                frameName = self._run()
                print "Frame #%d/%d: %s"
                    % (self.index+1, len(self.files), frameName)

            newIndex = self.index
            key = cv2.waitKey(0)

            if key == (0x10FF00 | 80):
                # KEY_HOME
                newIndex = 0
            elif key == (0x10FF00 | 87):
                # KEY_END
                newIndex = len(self.files)-1
            elif key == (0x10FF00 | 81):
                # KEY_LEFT
                if newIndex > 0:
                    newIndex-= 1
            elif key == (0x10FF00 | 83):
                # KEY_RIGHT
                if newIndex < len(self.files)-1:
                    newIndex+= 1
            elif key == (0x10FF00 | 82):
                # KEY_UP
                if newIndex > 10:
                    newIndex-= 10
                else:
                    newIndex = 0
            elif key == (0x10FF00 | 84):
                # KEY_DOWN
                if newIndex < (len(self.files)-1)-10:
                    newIndex+= 10
                else:
                    newIndex = len(self.files)-1
            elif key == (0x10FF00 | 85):
                # KEY_PAGEUP
                if newIndex > 100:
                    newIndex-= 100
                else:
                    newIndex = 0
            elif key == (0x10FF00 | 86):
                # KEY_PAGEDOWN
                if newIndex < (len(self.files)-1)-100:
                    newIndex+= 100
                else:
                    newIndex = len(self.files)-1
            elif key == (0x100000 | 9):
                # KEY_TAB
                if newIndex < (len(self.files)-1)-1000:
                    newIndex+= 1000
                else:
                    newIndex = len(self.files)-1
            elif key == (0x100000 | 27):
                # KEY_ESC
                break

    # PRIVATE
    def _atoi(self, text):
        return int(text) if text.isdigit() else text

    def _naturalKeys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self._atoi(c) for c in re.split('(\d+)', text) ]
