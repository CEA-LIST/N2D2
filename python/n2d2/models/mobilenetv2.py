"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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
"""


import n2d2.deepnet
import n2d2.global_variables
from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, Composite


"""
    For now, only support for ONNX model
"""


def load_from_ONNX(inputs, dims=None, batch_size=1, path=None, download=False):
    print("Loading MobileNet_v2 from ONNX with dims " + str(dims) + " and batch size " + str(batch_size))
    if path is None and not download:
        raise RuntimeError("No path specified")
    elif not path is None and download:
        raise RuntimeError("Specified at same time path and download=True")
    elif path and not download:
        path = n2d2.global_variables.model_cache + "/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    else:
        n2d2.utils.download_model("https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
            n2d2.global_variables.model_cache+"/ONNX/",
            "mobilenetv2")
        path = n2d2.global_variables.model_cache+"/ONNX/mobilenetv2/mobilenetv2-1.0.onnx"
    model = n2d2.cells.DeepNetCell.load_from_ONNX(inputs, path)
    return model


def ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin),
        PadCrop(width=size, height=size),
        RangeAffine(first_operator='Divides', first_value=[255.0]),
        ColorSpace(color_space='RGB'),
        RangeAffine(first_operator='Minus', first_value=[0.485, 0.456, 0.406], second_operator='Divides', second_value=[0.229, 0.224, 0.225]),
    ])

    return trans



