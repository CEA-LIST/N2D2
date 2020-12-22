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

import N2D2
import n2d2

"""
At the moment, this class is rather superfluous, and servers mainly for hiding
the raw N2D2 binding class. However, in the long term it could serve as a 
canvas for defining datasets without the need to write a N2D2 database driver.
Alternatively, this could simply be done by the corresponding Pytorch functions
since there is no GPU model involved.
"""

class Database():

    StimuliSets = {
        'Learn': N2D2.Database.Learn,
        'Test': N2D2.Database.Test,
        'Validation': N2D2.Database.Validation,
        'Unpartitioned': N2D2.Database.Unpartitioned
    }



    def __init__(self, database):
        self._database = database

    def get_nb_stimuli(self, partition):
        return self._database.getNbStimuli(self.StimuliSets[partition])


    def N2D2(self):
        if self._database is None:
            raise n2d2.UndefinedModelError("N2D2 database member has not been created")
        return self._database

    def load(self, dataPath, **kwargs):
        self._database.load(dataPath=dataPath, **kwargs)


class MNIST(Database):

    _type = 'MNIST_IDX_Database'

    def __init__(self, datapath, Validation):

        self._datapath = datapath
        self._Validation = Validation

        super().__init__(database=N2D2.MNIST_IDX_Database(validation=self._Validation))

        # Necessary to initialize random number generator; TODO: Replace
        #net = N2D2.Network()
        #deepNet = N2D2.DeepNet(net)  # Proposition : overload the constructor to avoid passing a Network object

        self._database.load(self._datapath)

    

    def convert_to_INI_section(self):
        output = "[database]\n"
        output += "Type=" + self._type + "\n"
        output += "Validation=" + str(self._Validation) + "\n"
        return output

