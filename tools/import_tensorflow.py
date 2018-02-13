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
import re
from tensorflow.python import pywrap_tensorflow

def exportFreeParameters(modelName, targetDir):
    numpy.set_printoptions(threshold=numpy.inf, precision=12)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    try:
        reader = pywrap_tensorflow.NewCheckpointReader(modelName)
        varToShapeMap = reader.get_variable_to_shape_map()

        for key in varToShapeMap:
            name = str(key).split("/")

            offset = -1
            if name[-1] == "Momentum":
                offset = -2

            # Adapting batch norm naming convention
            if name[offset] == "gamma":
                name[offset] = "scales"
            elif name[offset] == "beta":
                name[offset] = "biases"
            elif name[offset] == "moving_mean":
                name[offset] = "means"
            elif name[offset] == "moving_variance":
                name[offset] = "variances"

            targetFile = os.path.join(targetDir, "_".join(name) + ".syntxt")
            print "Exporting %s" % (targetFile)

            f = open(targetFile, 'w')
            f.write(re.sub('[\[\]]', '',
                numpy.array_str(reader.get_tensor(key))))
            f.close()

    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in modelName for e in [".index", ".meta", ".data"]]))):
            proposedFile = ".".join(modelName.split(".")[0:-1])
            v2_file_error_template = """
                It's likely that this is a V2 checkpoint and you need to provide
                the filename *prefix*.  Try removing the '.' and extension. Try:
                inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposedFile))

parser = optparse.OptionParser(usage="""%prog <ckpt model> <output dir>

Convert a Tensorflow CKPT model to N2D2 .syntxt files.""")
options, args = parser.parse_args()

if len(args) != 2:
    parser.print_help()
    sys.exit(-1)

if not os.path.exists(args[0] + ".index"):
    raise Exception("CKPT model file does not exist: " + args[0])

exportFreeParameters(args[0], args[1])
