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

class Target:

    """Provider is not a parameter in the INI file in the case of Target class,
    but usually inferred from the deepnet in N2D2. Name and Cell are parts of the section name"""
    def __init__(self, Name, Cell, Provider):

        self._modified_keys = []

        self._constructor_parameters = {
            'Name': Name,
            'Cell': Cell,
            'Provider': Provider
        }


    def get_name(self):
        return self._constructor_parameters['Name']


    def N2D2(self):
        if self._target is None:
            raise n2d2.UndefinedModelError("N2D2 target member has not been created")
        return self._target


class Score(Target):

    _type = 'TargetScore'

    def __init__(self, Name, Cell, Provider, **target_parameters):

        super().__init__(Name, Cell, Provider)

        self._target_parameters = {
            'TargetValue': 1.0,
            'DefaultValue': 0.0,
            'TopN': 1,
            'LabelsMapping': "",
            'CreateMissingLabels': False
        }

        for key, value in target_parameters.items():
            if key in self._target_parameters:
                self._target_parameters[key] = value
                self._modified_keys.append(key)
            else:
                raise n2d2.UndefinedParameterError(key, self)

        self._target = N2D2.TargetScore(
            name=self._constructor_parameters['Name'],
            cell=self._constructor_parameters['Cell'].N2D2(),
            sp=self._constructor_parameters['Provider'].N2D2(),
            targetValue=self._target_parameters['TargetValue'],
            defaultValue=self._target_parameters['DefaultValue'],
            targetTopN=self._target_parameters['TopN'],
            labelsMapping=self._target_parameters['LabelsMapping'],
            createMissingLabels=self._target_parameters['CreateMissingLabels'])

        # TODO: Add post generation parameters

    def provide_targets(self, partition):
        self._target.provideTargets(self._constructor_parameters['Provider'].get_database().StimuliSets[partition])

    def process(self, partition):
        self._target.process(self._constructor_parameters['Provider'].get_database().StimuliSets[partition])

    def get_average_success(self, partition, window):
        return self._target.getAverageSuccess(self._constructor_parameters['Provider'].get_database().StimuliSets[partition], window)

    def convert_to_INI_section(self):
        output = "[" + self._constructor_parameters['Name'] + "]\n"
        #output += "Type=" + self._type + "\n"
        for key, value in self._target_parameters.items():
            if key in self._modified_keys:
                if isinstance(value, bool):
                    output += key + "=" + str(int(value)) + "\n"
                else:
                    output += key + "=" + str(value) + "\n"
        return output
