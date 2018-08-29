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
import subprocess

import urllib
from google.protobuf import text_format

def getChilds(graph, node):
    childs = []
    for n, parents in graph.iteritems():
        if node in parents:
            childs.append(n)
    return childs

def getParents(graph, node):
    return graph[node]

def graphMerge(graph, srcNode, dstNode):
    # srcNode childs become dstNode childs
    for child in getChilds(graph, srcNode):
        if child != dstNode and child not in getChilds(graph, dstNode):
            graph[child].append(dstNode)

    # srcNode parents become dstNode parents
    for parent in getParents(graph, srcNode):
        if parent != dstNode and parent not in getParents(graph, dstNode):
            graph[dstNode].append(parent)

    # Remove srcNode from graph
    for n, parents in graph.iteritems():
        if srcNode in parents:
            graph[n].remove(srcNode)

    del graph[srcNode]

def caffeToN2D2(netProtoFileName, solverProtoFileName = "", iniFileName = ""):
    urllib.urlretrieve("https://github.com/BVLC/caffe/raw/master/"
        + "src/caffe/proto/caffe.proto", "caffe.proto")

    if iniFileName == "":
        iniFileName = os.path.splitext(netProtoFileName)[0] + ".ini"

    (stdoutData, stderrData) = subprocess.Popen(["protoc",
        "--python_out=./", "caffe.proto"],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    print (stderrData)

    import caffe_pb2

    caffeNet = caffe_pb2.NetParameter()
    text_format.Merge(open(netProtoFileName).read(), caffeNet)

    caffeSolver = caffe_pb2.SolverParameter()

    if solverProtoFileName != "":
        text_format.Merge(open(solverProtoFileName).read(), caffeSolver)

    # Construct graph
    graph = {}  # graph stores the parent nodes as values
    attrs = {}

    for i, layer in enumerate(caffeNet.layer):
        # Parent nodes
        graph[layer.name] = layer.bottom

        # Child nodes
        inPlace = set(layer.top) & set(layer.bottom)

        if len(layer.top) == 1 and len(inPlace) == 1:
            # Unfold in-place layers
            for next_layer in caffeNet.layer[i+1:]:
                for b, val in enumerate(next_layer.bottom):
                    if val == layer.top[0]:
                        next_layer.bottom[b] = layer.name
                for t, val in enumerate(next_layer.top):
                    if val == layer.top[0]:
                        next_layer.top[t] = layer.name

    # Merge nodes
    for layer in caffeNet.layer:
        attrs[layer.name] = []
        parents = getParents(graph, layer.name)

        if layer.type == "ReLU":
            # Merge with parents
            for parent in parents:
                graphMerge(graph, layer.name, parent)
                attrs[parent].append("ReLU")

        elif layer.type == "Concat":
            # Merge with childs
            for child in getChilds(graph, layer.name):
                graphMerge(graph, layer.name, child)

        elif layer.type == "Dropout" and len(parents) > 1:
            # Split Dropout
            for parent in parents:
                graph[layer.name + "_" + parent] = [parent]

            for n, parent_nodes in graph.iteritems():
                if layer.name in parent_nodes:
                    idx = list(graph[n]).index(layer.name)
                    graph[n][idx:idx+1] \
                        = [layer.name + "_" + parent for parent in parents]

        elif layer.type == "Scale":
            # TODO: not supported yet
            # For now, merge with parents
            for parent in parents:
                graphMerge(graph, layer.name, parent)

    # Generate INI file
    iniTxt = ""

    iniTxt += "; Learning parameters\n"
    iniTxt += "$LR=" + str(caffeSolver.base_lr) + "\n"
    iniTxt += "$WD=" + str(caffeSolver.weight_decay) + "\n"
    iniTxt += "$MOMENTUM=" + str(caffeSolver.momentum) + "\n"
    iniTxt += "\n"

    for layer in caffeNet.layer:
        if layer.name not in graph:
            continue

        config = ""
        commonConfig = False

        if layer.type == "Data":
            inc = (len(layer.include) == 0)
            for include in layer.include:
                if caffe_pb2.Phase.Name(include.phase) == "TRAIN":
                    inc = True
            if not inc:
                continue

            iniTxt += "[sp]\n"
            iniTxt += "SizeX= ; TODO\n"
            iniTxt += "SizeY= ; TODO\n"
            iniTxt += "NbChannels= ; TODO\n"
            iniTxt += "BatchSize=" + str(layer.data_param.batch_size) + "\n"

            if layer.transform_param.crop_size > 0:
                iniTxt += "[sp.Transformation-crop]\n"
                iniTxt += "Type=PadCropTransformation\n"
                iniTxt += "Width=" + str(layer.transform_param.crop_size) + "\n"
                iniTxt += "Height=" + str(layer.transform_param.crop_size) + "\n"

            if len(layer.transform_param.mean_value) > 0:
                scale = layer.transform_param.scale \
                    if layer.transform_param.scale != 1 else 1.0/255.0

                iniTxt += "[sp.Transformation-scale-mean]\n"
                iniTxt += "Type=RangeAffineTransformation\n"
                iniTxt += "FirstOperator=Minus\n"
                iniTxt += "FirstValue=" + " ".join([str(x)
                    for x in layer.transform_param.mean_value]) + "\n"
                iniTxt += "SecondOperator=Multiplies\n"
                iniTxt += "SecondValue=" + str(scale) + "\n"

        elif layer.type == "Convolution":
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + ",".join(graph[layer.name]) + "\n"
            iniTxt += "Type=Conv\n"

            if len(layer.convolution_param.kernel_size) == 1:
                iniTxt += "KernelSize=" \
                    + str(layer.convolution_param.kernel_size[0]) + "\n"
            else:
                iniTxt += "KernelDims=" + " ".join([str(dim) \
                    for dim in layer.convolution_param.kernel_size]) + "\n"

            if len(layer.convolution_param.pad) == 1:
                iniTxt += "Padding=" + str(layer.convolution_param.pad[0]) \
                    + "\n"
            elif len(layer.convolution_param.pad) > 1:
                iniTxt += "Padding=" + " ".join([str(pad) \
                    for pad in layer.convolution_param.pad]) + " ; TODO\n"

            if len(layer.convolution_param.stride) == 1:
                iniTxt += "Stride=" + str(layer.convolution_param.stride[0]) \
                    + "\n"
            elif len(layer.convolution_param.stride) > 1:
                iniTxt += "StrideDims=" + " ".join([str(dim) \
                    for dim in layer.convolution_param.stride]) + "\n"

            if layer.convolution_param.HasField('group'):
                iniTxt += "NbGroups=" + str(layer.convolution_param.group) \
                    + "\n"

            # Weights filler
            if layer.convolution_param.weight_filler.type == "msra":
                iniTxt += "WeightsFiller=HeFiller\n"
            elif layer.convolution_param.weight_filler.type == "xavier":
                iniTxt += "WeightsFiller=XavierFiller\n"
            elif layer.convolution_param.weight_filler.type == "gaussian":
                iniTxt += "WeightsFiller=NormalFiller\n"
                iniTxt += "WeightsFiller.Mean=" \
                    + str(layer.convolution_param.weight_filler.mean) + "\n"
                iniTxt += "WeightsFiller.StdDev=" \
                    + str(layer.convolution_param.weight_filler.std) + "\n"

            # Bias
            if not layer.convolution_param.bias_term:
                config += "NoBias=1\n"
            elif layer.convolution_param.HasField('bias_filler'):
                if layer.convolution_param.bias_filler.type == "constant":
                    iniTxt += "BiasFiller=ConstantFiller\n"
                    iniTxt += "BiasFiller.Value=" \
                        + str(layer.convolution_param.bias_filler.value) + "\n"

            if len(layer.param) > 0:
                if layer.param[0].lr_mult != 1:
                    config += "WeightsSolver.LearningRate=$(" \
                        + str(layer.param[0].lr_mult) + " * ${LR})\n"
                if layer.param[0].decay_mult != 1:
                    config += "WeightsSolver.Decay=$(" \
                        + str(layer.param[0].decay_mult) + " * ${LR})\n"

            if len(layer.param) > 1:
                if layer.param[1].lr_mult != 1:
                    config += "BiasSolver.LearningRate=$(" \
                        + str(layer.param[1].lr_mult) + " * ${LR})\n"
                if layer.param[1].decay_mult != 1:
                    config += "BiasSolver.Decay=$(" \
                        + str(layer.param[1].decay_mult) + " * ${LR})\n"

            iniTxt += "NbOutputs=" + str(layer.convolution_param.num_output) \
                + "\n"

            commonConfig = True

        elif layer.type == "Pooling":
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + ",".join(graph[layer.name]) + "\n"
            iniTxt += "Type=Pool\n"

            pool = caffe_pb2.PoolingParameter.PoolMethod.Name(\
                    layer.pooling_param.pool)

            if pool == "AVE":
                iniTxt += "Pooling=Average\n"
            elif pool == "MAX":
                iniTxt += "Pooling=Max\n"
            else:
                iniTxt += "Pooling= ; TODO: unsupported: " + pool + "\n"

            if layer.pooling_param.global_pooling:
                iniTxt += "PoolDims=[" + graph[layer.name][0] \
                    + "]_OutputsWidth [" + graph[layer.name][0] \
                    + "]_OutputsHeight\n"
            else:
                iniTxt += "PoolSize=" + str(layer.pooling_param.kernel_size) \
                    + "\n"

            if layer.pooling_param.pad != 0:
                iniTxt += "Padding=" + str(layer.pooling_param.pad) + "\n"

            if layer.pooling_param.stride != 1:
                iniTxt += "Stride=" + str(layer.pooling_param.stride) + "\n"

            iniTxt += "NbOutputs=[" + graph[layer.name][0] + "]NbOutputs\n"
            iniTxt += "Mapping.ChannelsPerGroup=1\n"

        elif layer.type == "BatchNorm":
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + ",".join(layer.bottom) + "\n"
            iniTxt += "Type=BatchNorm\n"
            iniTxt += "NbOutputs=[" + graph[layer.name][0] + "]NbOutputs\n"

            config += "Epsilon=" + str(layer.batch_norm_param.eps) + "\n"

        elif layer.type == "Eltwise":
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + ",".join(graph[layer.name]) + "\n"
            iniTxt += "Type=ElemWise\n"

            if layer.eltwise_param.operation == "PROD":
                iniTxt += "Operation=Prod\n"
            elif layer.eltwise_param.operation == "MAX":
                iniTxt += "Operation=Max\n"
            else:
                iniTxt += "Operation=Sum\n"

            iniTxt += "NbOutputs=[" + graph[layer.name][0] + "]NbOutputs\n"

        elif layer.type == "Softmax" or layer.type == "SoftmaxWithLoss":
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + graph[layer.name][0] + "\n"
            iniTxt += "Type=Softmax\n"

            if layer.type == "SoftmaxWithLoss":
                iniTxt += "WithLoss=1\n"

            iniTxt += "NbOutputs=[" + graph[layer.name][0] + "]NbOutputs\n"

            # TODO: support with Caffe Accuracy layer
            iniTxt += "[" + layer.name + ".Target]\n"

        elif layer.type == "Dropout":
            if len(graph[layer.name]) > 1:
                for k, in_layer in enumerate(graph[layer.name]):
                    iniTxt += "[" + layer.name + "_" + in_layer + "]\n"
                    iniTxt += "Input=" + graph[layer.name][k] + "\n"
                    iniTxt += "Type=Dropout\n"
                    iniTxt += "NbOutputs=[" + graph[layer.name][k] \
                        + "]NbOutputs\n"

                    if k != len(graph[layer.name]) - 1:
                        iniTxt += "ConfigSection=" + layer.name + ".cfg\n"
            else:
                iniTxt += "[" + layer.name + "_" +  + "]\n"
                iniTxt += "Input=" + ",".join(graph[layer.name]) + "\n"
                iniTxt += "Type=Dropout\n"
                iniTxt += "NbOutputs=[" + graph[layer.name][0] + "]NbOutputs\n"

            config += "Dropout=" + str(layer.dropout_param.dropout_ratio) + "\n"

        elif layer.type == "Accuracy":
            iniTxt += "; Accuracy layer was ignored\n\n"
            continue

        else:
            iniTxt += "; TODO: not supported:\n"
            iniTxt += "[" + layer.name + "]\n"
            iniTxt += "Input=" + ",".join(graph[layer.name]) + "\n"
            iniTxt += "Type=" + layer.type + "\n"


        # Attributes
        if "ReLU" in attrs[layer.name]:
            iniTxt += "ActivationFunction=Rectifier\n"

        # Config section
        if commonConfig or config != "":
            iniTxt += "ConfigSection="

        if commonConfig:
            iniTxt += "common.cfg"

            if config != "":
                iniTxt += ","
            else:
                iniTxt += "\n"

        if config != "":
            iniTxt += layer.name + ".cfg\n"
            iniTxt += "[" + layer.name + ".cfg]\n"
            iniTxt += config

        iniTxt += "\n"

    iniTxt += "[common.cfg]\n"
    iniTxt += "Solvers.LearningRate=${LR}\n"
    iniTxt += "Solvers.Decay=${WD}\n"
    iniTxt += "Solvers.Momentum=${MOMENTUM}\n"

    if caffeSolver.lr_policy == "poly":
        iniTxt += "Solvers.LearningRatePolicy=PolyDecay\n"
    else:
        iniTxt += "Solvers.LearningRatePolicy= ; TODO: unsupported: " \
            + caffeSolver.lr_policy + "\n"

    iniTxt += "Solvers.Power=" + str(caffeSolver.power) + "\n"
    iniTxt += "Solvers.IterationSize=" + str(caffeSolver.iter_size) + "\n"
    iniTxt += "Solvers.MaxIterations=" + str(caffeSolver.max_iter) + "\n"
    iniTxt += "\n"

    iniFile = open(iniFileName, "w")
    iniFile.write(iniTxt)
    iniFile.close()

parser = optparse.OptionParser(usage="""%prog <Net .prototxt> <Solver .prototxt>

Convert Caffe .prototxt files to N2D2 INI file.""")
options, args = parser.parse_args()

if len(args) < 1:
    parser.print_help()
    sys.exit(-1)

netProto = args[0]
solverProto = args[1] if len(args) > 1 else ""

if not os.path.exists(netProto):
    raise Exception("Caffe Net .prototxt file not found: " + netProto)

if solverProto != ""  and not os.path.exists(solverProto):
    raise Exception("Caffe Solver .prototxt file not found: " + solverProto)

caffeToN2D2(netProto, solverProto)
