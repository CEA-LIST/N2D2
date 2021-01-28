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
from n2d2.n2d2_interface import N2D2_Interface


class DataProvider(N2D2_Interface):
    _INI_type = 'sp'
    _type = "DataProvider"

    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, Database, Size, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'Database': Database,
            'Size': Size
        })

        self._parse_optional_arguments(['BatchSize', 'CompositeStimuli'])

        self._N2D2_object = N2D2.StimuliProvider(database=self._constructor_arguments['Database'].N2D2(),
                                                 size=self._constructor_arguments['Size'],
                                                 **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

        # Dictionary of transformation objects
        self._transformations = []
        self._otf_transformations = []


    def get_database(self):
        return self._constructor_arguments['Database']

    def read_random_batch(self, partition):
        return self._N2D2_object.readRandomBatch(set=N2D2.Database.StimuliSet.__members__[partition])

    def read_batch(self, partition, idx):
        return self._N2D2_object.readBatch(set=N2D2.Database.StimuliSet.__members__[partition],
                                           startIndex=idx)

    def add_transformation(self, transformation):
        self._N2D2_object.addTransformation(transformation.N2D2(), transformation.get_apply_set())
        self._transformations.append(transformation)

    def add_on_the_fly_transformation(self, transformation):
        self._N2D2_object.addOnTheFlyTransformation(transformation.N2D2(), transformation.get_apply_set())
        self._otf_transformations.append(transformation)

    def __str__(self):
        return self._type + N2D2_Interface.__str__(self)

    def convert_to_INI_section(self):
        output = "[" + self._INI_type + "]\n"
        output += "Size="
        for idx, dim in enumerate(self._constructor_arguments['Size']):
            if idx > 0:
                output += " "
            output += str(dim)
        output += "\n"
        for key, value in self._optional_constructor_arguments.items():
            if key in self._modified_keys:
                if isinstance(value, bool):
                    output += key + "=" + str(int(value)) + "\n"
                else:
                    output += key + "=" + str(value) + "\n"
        return output
