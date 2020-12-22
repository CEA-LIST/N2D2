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
import N2D2

def test_graph():
    first_block = n2d2.cell.Fc(NbOutputs=10)
    second_block = n2d2.cell.Block([
        n2d2.cell.Fc(NbOutputs=10),
    ])
    third_block = n2d2.cell.Block([
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=10),
            n2d2.cell.Fc(NbOutputs=10),
        ]),
        n2d2.cell.Block([
            n2d2.cell.Fc(NbOutputs=10),
            n2d2.cell.Fc(NbOutputs=10),
            n2d2.cell.Fc(NbOutputs=10),
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


def deepnet_sequential_constructor_test(verbose=False):

    neuron_checks = ['Fc_0', 'Fc_1.0', 'Fc_2.0.0', 'Fc_2.0.1', 'Fc_2.1.0', 'Fc_2.1.1', 'Fc_2.1.2', 'Fc_3', 'Softmax_4']

    net = N2D2.Network()
    deepnet = N2D2.DeepNet(net)

    model = n2d2.deepnet.Sequential(deepnet, test_graph(), Model='Frame')

    if verbose:
        print(model)

    seq = model.get_cells()

    for idx, name in enumerate(neuron_checks):
        # Test access by name
        cell = model.get_cell(name)
        if not cell.get_name() == name:
            if verbose:
                print(cell.get_name() + " vs. " + name)
            return False
        # Test access by sequence index
        cell = seq[idx]
        if not cell.get_name() == name:
            if verbose:
                print(cell.get_name() + " vs. " + name)
            return False

    return True

test_functions = [
    deepnet_sequential_constructor_test,
]

def main():
    for test in test_functions:
        if not test():
            print("Test \'" + test.__name__ + "\' failed!")
            print("### Re-run in verbose mode ###")
            test(verbose=True)
            print("### End re-run ###\n")

        else:
            print("Test \'" + test.__name__ + "\' succeeded!")


if __name__ == "__main__":
    main()
