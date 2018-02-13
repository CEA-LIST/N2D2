#!/usr/bin/python
# -*- coding: ISO-8859-1 -*-
################################################################################
#    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

import os, sys
import optparse
import numpy

def exportFreeParameters(sourceDir, targetDir):
    numpy.set_printoptions(threshold=numpy.inf, precision=12)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    for fileName in os.listdir(sourceDir):
        if fileName.endswith(".npy"):
            paramsName = os.path.splitext(fileName)[0]
            targetFile = os.path.join(targetDir, paramsName + ".syntxt")
            print "Exporting %s" % (targetFile)

            params = numpy.load(os.path.join(sourceDir, fileName))
            numpy.savetxt(targetFile, params.flatten())

parser = optparse.OptionParser(usage="""%prog <ckpt model> <output dir>

Convert NumPy saved weights (*.npy) to N2D2 .syntxt files.""")
options, args = parser.parse_args()

if len(args) != 2:
    parser.print_help()
    sys.exit(-1)

if not os.path.exists(args[0]):
    raise Exception("Directory containing NumPy weights does not exist: "
        + args[0])

exportFreeParameters(args[0], args[1])
