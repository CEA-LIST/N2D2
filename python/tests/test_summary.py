"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 

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


import unittest
import n2d2


class test_summary(unittest.TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_summary_after_remove_cell(self):
        database = n2d2.database.Database()
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=1)

        fc1 = n2d2.cells.Fc(28*28, 50, activation=n2d2.activation.Rectifier())
        fc2 = n2d2.cells.Fc(50, 10)
        softmax = n2d2.cells.Softmax(with_loss=True)

        seq = n2d2.cells.Sequence([fc1, fc2, softmax])

        dn_cell = seq.to_deepnet_cell(provider)
        dn_cell.remove(dn_cell[1].get_name(), reconnect=True)
        dn_cell.summary()


if __name__ == '__main__':
    """
    You need to add this line for the tests to be run.
    """
    unittest.main()



