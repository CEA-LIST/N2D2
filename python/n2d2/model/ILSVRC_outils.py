"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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


from n2d2.transform import Rescale, PadCrop, ColorSpace, RangeAffine, SliceExtraction, Flip, Composite

def ILSVRC_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin, keepAspectRatio=True, resizeToFit=False),
        PadCrop(width=size+margin, height=size+margin),
        ColorSpace(colorSpace='BGR'),
        RangeAffine(firstOperator='Minus', firstValue=[103.94, 116.78, 123.68], secondOperator='Multiplies', secondValue=[0.017]),
        SliceExtraction(width=size, height=size, offsetX=margin//2, offsetY=margin//2, applyTo='NoLearn')
    ])

    otf_trans = Composite([
        SliceExtraction(width=size, height=size, randomOffsetX=True, randomOffsetY=True, applyTo='LearnOnly'),
        Flip(randomHorizontalFlip=True, applyTo='LearnOnly')
    ])

    return trans, otf_trans



def MobileNet_v2_ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin),
        PadCrop(width=size, height=size),
        RangeAffine(firstOperator='Divides', firstValue=[255.0]),
        ColorSpace(colorSpace='RGB'),
        RangeAffine(firstOperator='Minus', firstValue=[0.485, 0.456, 0.406], secondOperator='Divides', secondValue=[0.229, 0.224, 0.225]),
    ])

    return trans




def ResNet_ONNX_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin, keepAspectRatio=False, ResizeToFit=False),
        PadCrop(width=size, height=size),
        RangeAffine(firstOperator='Divides', firstValue=[255.0]),
        ColorSpace(colorSpace='RGB'),
        RangeAffine(firstOperator='Minus', firstValue=[0.485, 0.456, 0.406], secondOperator='Divides', secondValue=[0.229, 0.224, 0.225]),
    ])

    return trans
