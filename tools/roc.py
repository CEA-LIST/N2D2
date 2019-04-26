#!/usr/bin/python -u
# -*- coding: ISO-8859-1 -*-
################################################################################
# Author: Olivier BICHLER (olivier.bichler@cea.fr)
# (C) Copyright 2019 CEA LIST
################################################################################

import os, sys
import numpy, pylab
import optparse
import textwrap
import csv

parser = optparse.OptionParser(usage="""%prog <output> [options]

Display ROC curve for a given classifier output.""")
parser.add_option('-s', action="store", dest="step", type="float", default=0.05,
    help="threshold steps size for the ROC curve [%default]")
options, args = parser.parse_args()

if len(args) == 0:
    outputNum = int(input('Classifier output to consider: '))
else:
    outputNum = int(args[0])

################################################################################

prefix = os.path.basename(__file__).split("_")
dirPath = "."

if len(prefix) > 1:
    dirPath = prefix[0]

with open(os.path.join(dirPath, "classif.log"), 'rb') as f:
    # Support for N2D2 Utils::quote()
    classif = csv.reader(f, delimiter=' ', doublequote=False, escapechar='\\',
                         strict=True)
    classif = [row for row in classif]

classif = numpy.array(classif).transpose()
targets = numpy.array(classif[2]).astype(int)
outputs = numpy.array(classif[5:]).astype(float)

# Check range of either Softmax outputs or Logistic outputs
if outputNum < 0 or (len(outputs) > 1 and outputNum >= len(outputs)) \
  or (outputNum > len(outputs)):
    raise Exception("Specified output (%d) is out of range [0, %d]"
        % (outputNum, len(outputs) - (len(outputs) > 1)))

if os.path.isfile("labels_mapping.log.dat"):
    legend = numpy.loadtxt("labels_mapping.log.dat", dtype='S')
    legend = legend[1:].transpose()

    legendMask = legend[2].astype(int)
    legendMask = (legendMask == outputNum)
    legendLabels = legend[1]
    legendLabels = [legendLabels[i] for i in xrange(len(legendLabels))
                                                            if legendMask[i]]

    wrapper = textwrap.TextWrapper(width=50)
    legendWrapped = wrapper.wrap(text=("%s" % legendLabels))
    legendWrapped = "Labels for output #%d:\n" % (outputNum) \
                        + "\n".join(legendWrapped[0:10])
else:
    legendWrapped = "labels_mapping.log.dat not found"

thresholds = numpy.arange(options.step, 1.0, options.step)
totalHit = numpy.sum(targets == outputNum)
totalFp = numpy.sum(targets != outputNum)

hitRate = []
fpRate = []

if len(outputs) == 1 and outputNum == 0:
    outputs = 1.0 - outputs

for thresh in thresholds:
    if len(outputs) > 1:
        # Softmax outputs
        hits = numpy.logical_and(targets == outputNum,
                                 outputs[outputNum] > thresh)
        fps = numpy.logical_and(targets != outputNum,
                                outputs[outputNum] > thresh)
    else:
        # Logistic output
        hits = numpy.logical_and(targets == outputNum, outputs[0] > thresh)
        fps = numpy.logical_and(targets != outputNum, outputs[0] > thresh)

    hitRate.append(numpy.sum(hits) / float(totalHit))
    fpRate.append(numpy.sum(fps) / float(totalFp))


fig = pylab.figure()
ax1 = fig.add_subplot(111)
pylab.title("ROC curve for classifier output #%d" % (outputNum), y=1.08)
pylab.plot(fpRate, hitRate, drawstyle='steps')
pylab.plot(fpRate, hitRate, '+', label=legendWrapped)
pylab.legend(loc='lower right', markerscale=0.0, handlelength=0.0,
             handletextpad=0.0)
pylab.grid()
pylab.minorticks_on()
pylab.tick_params(which='both', direction='in')

prevXY = None
thres = []
for i, xy in enumerate(zip(fpRate, hitRate)):
    if prevXY is None:
        prevXY = xy

    if xy != prevXY:
        if len(thres) > 2:
            thres = [thres[0], thres[-1]]
        ax1.annotate("-".join(thres), xy=prevXY, xycoords='data',
            textcoords='offset pixels', xytext=(3,-12), fontsize=8)

        prevXY = xy
        thres = []

    thres.append("%s" % thresholds[i])

ax1.annotate(" ".join(thres), xy=prevXY, xycoords='data',
    textcoords='offset pixels', xytext=(3,-12), fontsize=8)

pylab.xlabel("False positive rate")
pylab.ylabel("True positive (hit) rate")

# Plot stimuli number corresponding to fpRate
ax2y = ax1.twiny()
ax1Xs = ax1.get_xticks()

ax2Xs = []
for X in ax1Xs:
    ax2Xs.append(int(round(X * float(totalFp))))

ax2y.set_xticks(ax1Xs)
ax2y.set_xbound(ax1.get_xbound())
ax2y.set_xticklabels(ax2Xs, fontsize=8, color='gray')
ax2y.tick_params(which='both', direction='in')

# Plot stimuli number corresponding to hitRate
ax2x = ax1.twinx()
ax1Ys = ax1.get_yticks()

ax2Ys = []
for Y in ax1Ys:
    ax2Ys.append(int(round(Y * float(totalHit))))

ax2x.set_yticks(ax1Ys)
ax2x.set_ybound(ax1.get_ybound())
ax2x.set_yticklabels(ax2Ys, fontsize=8, color='gray')
ax2x.tick_params(which='both', direction='in')

pylab.show()
