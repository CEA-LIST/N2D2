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


class DataProvider():

    _name = 'sp'

    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, Database, Size, **provider_parameters):

        self._modified_keys = []

        self._constructor_parameters = {
            'Database': Database,
            'Size': Size
        }

        self._provider_parameters = {
            'BatchSize': 1,
            'CompositeStimuli': False,
        }

        for key, value in provider_parameters.items():
            if key in self._provider_parameters:
                self._provider_parameters[key] = value
                self._modified_keys.append(key)
            else:
                raise n2d2.UndefinedParameterError(key, self)

        self._provider = N2D2.StimuliProvider(database=self._constructor_parameters['Database'].N2D2(),
                                              size=self._constructor_parameters['Size'],
                                              batchSize=self._provider_parameters['BatchSize'],
                                              compositeStimuli=self._provider_parameters['CompositeStimuli'])

        # Dictionary of transformation objects
        #self.transformations = None
        
    """def addTransformations(self, transformations)
        self.transformations = transformations
    """

    def get_name(self):
        return self._name

    def get_database(self):
        return self._constructor_parameters['Database']

    def read_random_batch(self, partition):
        return self._provider.readRandomBatch(set=self._constructor_parameters['Database'].StimuliSets[partition])

    def read_batch(self, partition, idx):
        return self._provider.readBatch(set=self._constructor_parameters['Database'].StimuliSets[partition], startIndex=idx)

    def addTransformation(self, transformation):
        self._provider.addTransformation(transformation.N2D2(), self.get_database().N2D2().StimuliSetMask(0))

    def addOnTheFlyTransformation(self, transformation):
        self._provider.addOnTheFlyTransformation(transformation.N2D2(), self.get_database().N2D2().StimuliSetMask(0))

    def N2D2(self):
        if self._provider is None:
            raise n2d2.UndefinedModelError("N2D2 solver member has not been created")
        return self._provider

    def convert_to_INI_section(self):
        output = "[" + self._name + "]\n"
        output += "Size="
        for idx, dim in enumerate(self._constructor_parameters['Size']):
            if idx > 0:
                output += " "
            output += str(dim)
        output += "\n"
        for key, value in self._provider_parameters.items():
            if key in self._modified_keys:
                if isinstance(value, bool):
                    output += key + "=" + str(int(value)) + "\n"
                else:
                    output += key + "=" + str(value) + "\n"
        return output
