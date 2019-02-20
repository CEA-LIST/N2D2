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

import os
import cv2
import re
import numpy

class TargetViewer(object):
    def __init__(self, path, regexp):
        self.files = [os.path.join(path, f) for f in os.listdir(path)
            if re.search(regexp, f)]
        self.files.sort(key=self._naturalKeys)
        self.index = -1

        # Same as in N2D2::Target::logLabelsLegend()
        self.cellHeight = 50
        self.imgLegend = []
        self.labelsHueOffset = 0

        self.initialDir = "~"
        self.gridSize = 50
        self.regexSlice = r'\[([0-9]+),([0-9]+)\]$'

    def run(self):
        self.newIndex = 0
        skipUp = False
        skipDown = False

        cv2.namedWindow("index", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("index", numpy.array([0.0]))
        cv2.createTrackbar("#", "index", 1, # start at 1 to force update to 0
            max(1, len(self.files)-1), self._onTrackbarChange)
        cv2.moveWindow("index", 1024 + 256 + 20, 0)
        cv2.resizeWindow("index", 512, 1)

        if self.imgLegend is not None:
            cv2.namedWindow("legend", cv2.WINDOW_NORMAL)
            cv2.imshow("legend", self.imgLegend)
            cv2.resizeWindow("legend", 256, 1024)
            cv2.moveWindow("legend", 1024 + 10, 0)

        while (True):
            self.skippable = False

            if self.newIndex != self.index:
                cv2.setTrackbarPos("#", "index", self.newIndex)

            if self.skippable:
                if skipUp and self.newIndex < len(self.files)-1:
                    self.newIndex = self.newIndex + 1
                    continue

                if skipDown and self.newIndex > 0:
                    self.newIndex = self.newIndex - 1
                    continue

            skipUp = False
            skipDown = False

            key = cv2.waitKey(0) & 0xFF

            print key

            if key == 80:
                # KEY_HOME
                self.newIndex = 0
            elif key == 87:
                # KEY_END
                self.newIndex = len(self.files)-1
            elif key == 81:
                # KEY_LEFT
                if self.newIndex > 0:
                    self.newIndex-= 1
            elif key == 83:
                # KEY_RIGHT
                if self.newIndex < len(self.files)-1:
                    self.newIndex+= 1
            elif key == 82:
                # KEY_UP
                if self.newIndex > 0:
                    self.newIndex-= 1
                    skipDown = True
                #if self.newIndex > 10:
                #    self.newIndex-= 10
                #else:
                #    self.newIndex = 0
            elif key == 84:
                # KEY_DOWN
                if self.newIndex < len(self.files)-1:
                    self.newIndex+= 1
                    skipUp = True
                #if self.newIndex < (len(self.files)-1)-10:
                #    self.newIndex+= 10
                #else:
                #    self.newIndex = len(self.files)-1
            elif key == 85:
                # KEY_PAGEUP
                if self.newIndex > 100:
                    self.newIndex-= 100
                else:
                    self.newIndex = 0
            elif key == 86:
                # KEY_PAGEDOWN
                if self.newIndex < (len(self.files)-1)-100:
                    self.newIndex+= 100
                else:
                    self.newIndex = len(self.files)-1
            elif key == 9:
                # KEY_TAB
                if self.newIndex < (len(self.files)-1)-1000:
                    self.newIndex+= 1000
                else:
                    self.newIndex = len(self.files)-1
            elif key == ord('f'):
                subString = raw_input("Find image: ")

                try:
                    self.newIndex = next(idx for idx, string in
                        enumerate(self.files) if subString in string)
                except StopIteration:
                    print "No match found!"
            elif key == ord('s'):
                import Tkinter
                import tkFileDialog

                root = Tkinter.Tk()
                root.withdraw()
                saveName = tkFileDialog.asksaveasfilename(
                  initialdir=self.initialDir,
                  initialfile=os.path.basename(self.files[self.index]) + ".jpg")
                root.destroy()

                self.initialDir = os.path.dirname(saveName)

                img = cv2.imread(self.files[self.index])
                cv2.imwrite(saveName, img)
            elif key == 27:
                # KEY_ESC
                break

    # PRIVATE
    def _onTrackbarChange(self, value):
        self.newIndex = value
        self._display()

    def _display(self):
        self.index = self.newIndex

        slicing = self._getSlicing()

        if slicing is not None:
            (gridX, gridY, sliceX, sliceY, posX, posY) = slicing
            imgSlicing = numpy.zeros((self.gridSize*gridY,
                                      self.gridSize*gridX, 1),
                dtype = "uint8")
            imgSlicing[self.gridSize*posY:self.gridSize*(posY+1),
                       self.gridSize*posX:self.gridSize*(posX+1)] = 255

            for x in xrange(0,gridX):
                for y in xrange(0,gridY):
                    color = (255)
                    if x == posX and y == posY:
                        color = (0)
                    cv2.putText(imgSlicing,
                        "%d,%d"% (x*sliceX, y*sliceY),
                        (self.gridSize*x + 5, self.gridSize*y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color)

            cv2.namedWindow("slicing", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("slicing", self.gridSize*gridX,
                                        self.gridSize*gridY)
            cv2.moveWindow("slicing", 1024 + 256 + 20, 128)
            cv2.imshow("slicing", imgSlicing)
            cv2.setMouseCallback("slicing", self._slicingMouseCallback, False)
        # -- Determine slice position

        frameName, self.skippable = self._run()

        print "Frame #%d/%d: %s" % (
            self.index+1, len(self.files), frameName)

    def _getSlicing(self):
        # Determine slice position --
        # Note that this code only works if slices are correctly
        # ordered, using "sort" with "natural_keys"
        gridX = 1
        gridY = 1
        offsetX = 0
        offsetY = 0
        i = self.index
        stimulusName = re.sub(self.regexSlice, '', self.files[i])

        # |- Rewind to the first slice index for the current original
        # image file
        while i > 0:
            if re.sub(self.regexSlice, '', self.files[i-1]) == stimulusName:
                i-= 1
            else:
                break

        # |- Find grid size and max offsets
        while i < len(self.files):
            if re.sub(self.regexSlice, '', self.files[i]) != stimulusName:
                break

            m = re.search(self.regexSlice, self.files[i])

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
        m = re.search(self.regexSlice, self.files[self.index])

        if m:
            sliceX = offsetX/max(gridX-1, 1)
            sliceY = offsetY/max(gridY-1, 1)
            posX = int(m.group(2))/max(sliceX, 1)
            posY = int(m.group(1))/max(sliceY, 1)
            return (gridX, gridY, sliceX, sliceY, posX, posY)
        else:
            return None

    def _slicingMouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            newPosX = x / self.gridSize
            newPosY = y / self.gridSize

            (gridX, gridY, sliceX, sliceY, posX, posY) = self._getSlicing()

            self.newIndex = self.index - (posY * gridX + posX) \
                                       + (newPosY * gridX + newPosX)
            self._display()

    def _atoi(self, text):
        return int(text) if text.isdigit() else text

    def _naturalKeys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self._atoi(c) for c in re.split('(\d+)', text) ]

    def _replace_last_of(self, text, pattern, new, occurrence=1):
        chunks = text.rsplit(pattern, occurrence)
        return new.join(chunks)
