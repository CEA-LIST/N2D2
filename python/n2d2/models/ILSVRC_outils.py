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

# TODO: Is this correct, in particular the PadCrop of dimension size+margin?
def ILSVRC_preprocessing(size=224):
    margin = 32

    trans = Composite([
        Rescale(width=size+margin, height=size+margin, keep_aspect_ratio=True, resize_to_fit=False),
        PadCrop(width=size+margin, height=size+margin),
        ColorSpace(color_space='BGR'),
        RangeAffine(first_operator='Minus', first_value=[103.94, 116.78, 123.68], second_operator='Multiplies', second_value=[0.017]),
        SliceExtraction(width=size, height=size, offset_x=margin//2, offset_y=margin//2, apply_to='NoLearn')
    ])

    otf_trans = Composite([
        SliceExtraction(width=size, height=size, random_offset_x=True, random_offset_y=True, apply_to='LearnOnly'),
        Flip(random_horizontal_flip=True, apply_to='LearnOnly')
    ])

    return trans, otf_trans
