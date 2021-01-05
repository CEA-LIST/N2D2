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

import n2d2

def fc_base():
    net = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=300),
            n2d2.cell.Fc(NbOutputs=10)
        ]),
        n2d2.cell.Softmax(NbOutputs=10)
    ])
    return net



def fc_base_named():
    net = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=300, Name='fc1'),
            n2d2.cell.Fc(NbOutputs=10, Name='fc2')
        ], Name='block1'),
        n2d2.cell.Softmax(NbOutputs=10, Name='softmax')
    ])
    return net


def conv_base():
    net = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Conv(NbOutputs=4, KernelDims=[5, 5]),
            n2d2.cell.Fc(NbOutputs=10)
        ]),
        n2d2.cell.Softmax(NbOutputs=10)
    ])
    return net



def fc_one_layer():

    first_block = n2d2.cell.Fc(NbOutputs=10)
    second_block = n2d2.cell.Softmax(NbOutputs=10)

    net = n2d2.cell.Block([
        first_block,
        second_block,
    ])
    return net


def fc_nested_named():
    first_block = n2d2.cell.Fc(NbOutputs=100, Name='fc1')
    second_block = n2d2.cell.Block([
        n2d2.cell.Fc(NbOutputs=100, Name='fc2'),
    ], Name='block1')
    third_block = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=100, Name='fc3'),
            n2d2.cell.Fc(NbOutputs=100, Name='fc4'),
        ], Name='block21'),
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=100, Name='fc5'),
            n2d2.cell.Fc(NbOutputs=100, Name='fc6'),
            n2d2.cell.Fc(NbOutputs=100, Name='fc7'),
        ], Name='block22'),
    ], Name='block2')
    fourth_block = n2d2.cell.Fc(NbOutputs=10, Name='fc8')
    top_block = n2d2.cell.Softmax(NbOutputs=10)

    net = n2d2.cell.Block([
        first_block,
        second_block,
        third_block,
        fourth_block,
        top_block,
    ])
    return net





def fc_nested():
    first_block = n2d2.cell.Fc(NbOutputs=100)
    second_block = n2d2.cell.Block([
        n2d2.cell.Fc(NbOutputs=100),
    ])
    third_block = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=100),
            n2d2.cell.Fc(NbOutputs=100),
        ]),
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=100),
            n2d2.cell.Fc(NbOutputs=100),
            n2d2.cell.Fc(NbOutputs=100),
        ]),
    ])
    fourth_block = n2d2.cell.Fc(NbOutputs=10)
    top_block = n2d2.cell.Softmax(NbOutputs=10)

    net = n2d2.cell.Block([
        first_block,
        second_block,
        third_block,
        fourth_block,
        top_block,
    ])
    return net