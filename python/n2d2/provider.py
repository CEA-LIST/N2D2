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


class Provider(N2D2_Interface):
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

    def dims(self):
        return self._N2D2_object.getSize()


class DataProvider(Provider):
    _type = "DataProvider"

    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, database, size, **config_parameters):
        Provider.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'database': database,
            'size': size
        })

        if 'name' in config_parameters:
            self._name = config_parameters.pop['name']
        else:
            self._name = "provider_" + str(n2d2.global_variables.provider_counter)
        n2d2.global_variables.provider_counter += 1

        self._parse_optional_arguments(['batchSize', 'compositeStimuli'])

        self._N2D2_object = N2D2.StimuliProvider(database=self._constructor_arguments['database'].N2D2(),
                                                 size=self._constructor_arguments['size'],
                                                 **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

        # Dictionary of transformation objects
        self._transformations = []
        self._otf_transformations = []


    def get_name(self):
        return self._name

    def get_database(self):
        return self._constructor_arguments['database']

    def read_random_batch(self, partition):
        return self._N2D2_object.readRandomBatch(set=N2D2.Database.StimuliSet.__members__[partition])

    def read_batch(self, partition, idx):
        return self._N2D2_object.readBatch(set=N2D2.Database.StimuliSet.__members__[partition], startIndex=idx)

    def add_transformation(self, transformation):
        if isinstance(transformation, n2d2.transform.Composite):
            for trans in transformation.get_transformations():
                self._N2D2_object.addTransformation(trans.N2D2(), trans.get_apply_set())
                self._transformations.append(trans)
        else:
            self._N2D2_object.addTransformation(transformation.N2D2(), transformation.get_apply_set())
            self._transformations.append(transformation)

    def add_on_the_fly_transformation(self, transformation):
        if isinstance(transformation, n2d2.transform.Composite):
            for trans in transformation.get_transformations():
                self._N2D2_object.addOnTheFlyTransformation(trans.N2D2(), trans.get_apply_set())
                self._transformations.append(transformation)
        else:
            self._N2D2_object.addOnTheFlyTransformation(transformation.N2D2(), transformation.get_apply_set())
            self._transformations.append(transformation)

    def __str__(self):
        output = "'" + self.get_name() + "' " + self._type + N2D2_Interface.__str__(self)
        if len(self._transformations) > 0:
            output += "[Transformations="
            for trans in self._transformations:
                output += trans.__str__() + ", "
            output = output[:-2]
            output += "]"
        return output


class TensorPlaceholder(Provider):
    def __init__(self, inputs, name=None):
        #Provider.__init__(self)

        if name is not None:
            self._name = name
        else:
            self._name = "provider_" + str(n2d2.global_variables.provider_counter)
        n2d2.global_variables.provider_counter += 1

        if isinstance(inputs, list):
            self._N2D2_object = N2D2.StimuliProvider(database=n2d2.database.Database().N2D2(),
                                                     size=inputs[0:2],
                                                     batchSize=inputs[3])
        elif isinstance(inputs, n2d2.tensor.Tensor) or isinstance(inputs, N2D2.BaseTensor):
            dims = [inputs.dimX(), inputs.dimY(), inputs.dimZ()]
            self._N2D2_object = N2D2.StimuliProvider(database=n2d2.database.Database().N2D2(),
                                                     size=dims,
                                                     batchSize=inputs.dimB())
            self._N2D2_object.setStreamedTensor(inputs)
        else:
            n2d2.error_handler.wrong_input_type("inputs", type(inputs), [type(list), 'n2d2.tensor.Tensor', 'N2D2.BaseTensor'])
        self._set_N2D2_parameter('StreamTensor', True)

    def set_streamed_tensor(self, tensor):
        self._N2D2_object.setStreamedTensor(tensor)

    def get_name(self):
        return self._name

    def __str__(self):
        return "'" + self.get_name() + "' TensorPlaceholder"

