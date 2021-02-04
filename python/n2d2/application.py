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

class Application:
    _target = None
    _provider = None
    _mode = None
    def __init__(self, model):
        self._model = model

    def convert_to_INI(self, filename):
        n2d2.utils.convert_to_INI(filename, self._provider.get_database(), self._provider, self._model, self._target)

    def getLoss(self, saveFile=""):
        """
        Return the list of loss.
        """

        if self._target:
            loss = self._target.N2D2().getLoss()
            if saveFile:
                with open(saveFile, 'w') as file:
                    for l in loss:
                        file.write(l)
            return loss
        else:
            raise RuntimeError("Target not initialized !")

    def recognitionRate(self):
        """
        return the recognition rate
        """
        set = N2D2.Database.StimuliSet.__members__[self._mode]
        return self._target.N2D2().getAverageSuccess(set)

    def logConfusionMatrix(self, path):
        assert self._provider and self._mode and self._target
        set = N2D2.Database.StimuliSet.__members__[self._mode]
        self._target.N2D2().logConfusionMatrix(path, set)

    def logSuccess(self, path):
        """
        Save a graph of the loss and the validation score as a function of the step number.
        """
        assert self._provider and self._mode and self._target
        set = N2D2.Database.StimuliSet.__members__[self._mode]
        self._target.N2D2().logSuccess(path, set)

    # TODO : doesn't work for frame_CUDA and Spike
    # TODO : also doesn't work with the current structure of layers and sequences !
    # def show_outputs(self):
    #     string = "Cells outputs :\n###############\n"
    #     for cell in self._model.get_cells():
    #         string += cell.getName() + ": "
    #         for output in cell.getOutputs():
    #             string += str(output) + " "
    #         string += "\n"
    #     print(string)

class Classifier(Application):
    def __init__(self, provider, model, **target_config_parameters):
        Application.__init__(self, model)

        self._provider = provider

        #print("Add provider")
        #self._model.add_input(self._provider)

        print("Create target")
        self._target = n2d2.target.Score(self._model.get_last(), self._provider, **target_config_parameters)

        #print("Initialize")
        #self._model.initialize()

        self._mode = 'Test'

    def set_mode(self, mode):
        if mode not in N2D2.Database.StimuliSet.__members__:
            raise ValueError("Mode " + mode + " not compatible with database stimuli sets")
        self._mode = mode

    def process(self):
        # Set target of cell
        self._target.provide_targets(partition=self._mode)

        # Propagate
        self._model.propagate(inference=(self._mode == 'Test' or self._mode == 'Validation'))

        # Calculate loss and error
        self._target.process(partition=self._mode)

    def optimize(self):
        # Backpropagate
        self._model.back_propagate()

        # Update parameters
        self._model.update()

    def read_random_batch(self):
        self._provider.read_random_batch(partition=self._mode)

    def read_batch(self, idx):
        self._provider.read_batch(partition=self._mode, idx=idx)

    def get_average_success(self, window=0):
        return self._target.get_average_success(partition=self._mode, window=window)

    def get_average_score(self, metric):
        return self._target.get_average_score(partition=self._mode, metric=metric)

