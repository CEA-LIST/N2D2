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

'''
Guide line for testing

The file test file should be placed in the test folder and should be named following the convention : "test_*.py".

The command to test the library is (you have to be in N2D2/python) :

python -m unittest discover -s test -v

The option discovery check for test_*.py files, so for example this file will note be caught !


If you need more information please check : https://docs.python.org/3/library/unittest.html
'''

import unittest

class test_name(unittest.TestCase):
    """
    The class needs to inherite unittest.TestCase, the name doesn't matter and the class doesn't need to be instanciated.
    """

    def setUp(self):
        """
        Method called before each test
        """
        pass

    def tearDown(self):
        """
        Method called after a test even if it failed.
        Can be used to clean variables
        """
        pass

    def test_X(self):
        """
        Method called to test a functionnality. It needs to be named test_* to be called.
        """

        """
        To test the functions you can use one of the following method :
        - self.assertEqual(a, b)
        - self.assertNotEqual(a, b)
        - self.assertTrue(a)
        - self.assertFalse(a)
        - self.assertIs(a, b)
        - self.assertIsNot(a, b)
        - self.assertIsNotNone(a)
        - self.assertIn(a, b)
        - self.assertNotIn(a, b)
        - self.assertIsInstance(a, b)
        """

        """
        You can use the following decorator to test 
        - @unittest.skip(display_text)
        - @unittest.skipIf(cond, display_text)
        - @unittest.skipUnless(cond, display_text)
        - @unittest.expectedFailure()
        """
        """
        You can test that a function raises error by putting it in a block :
        with self.assertRaises(TypeError): 
        You can replace TypeError by the type of message you are expecting
        """
        pass

if __name__ == '__main__':
    """
    You need to had this line for the tests to be run.
    """
    unittest.main()