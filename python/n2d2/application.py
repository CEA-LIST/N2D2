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

# TODO: Make abstract
class Application:
    def __init__(self, provider, model):
        self._deepnet = model
        self._provider = provider
        self._mode = 'Test'


   

class Classifier(Application):
    def __init__(self, provider, cell, **target_config_parameters):
        deepnet_endpoint = cell.get_last()
        if not isinstance(deepnet_endpoint, n2d2.cell.Cell):
            raise RuntimeError("The deepnet endpoint is not a single cell")
        Application.__init__(self, provider, deepnet_endpoint.get_deepnet())

        self._target = n2d2.target.Score(deepnet_endpoint,
                                         self._provider,
                                         **target_config_parameters)


    def get_average_success(self, window=0):
        return self._target.get_average_success(partition=self._mode, window=window)

    def clear_success(self):
        self._target.clear_success(partition=self._mode)

    def get_average_score(self, metric):
        return self._target.get_average_score(partition=self._mode, metric=metric)

    def set_mode(self, mode):
        if mode not in N2D2.Database.StimuliSet.__members__:
            raise ValueError("Mode " + mode + " not compatible with database stimuli sets")
        self._mode = mode

    def read_random_batch(self):
        self._provider.read_random_batch(partition=self._mode)

    def read_batch(self, idx):
        self._provider.read_batch(partition=self._mode, idx=idx)

    def process(self):
        # Set target of cell
        self._target.provide_targets(partition=self._mode)

        # Propagate
        self._deepnet.propagate(inference=(self._mode == 'Test' or self._mode == 'Validation'))

        # Calculate loss and error
        self._target.process(partition=self._mode)

    def optimize(self):
        # Backpropagate
        self._deepnet.back_propagate()

        # Update parameters
        self._deepnet.update()

    def get_loss(self, saveFile=""):
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

    def get_current_loss(self):
        return self._target.N2D2().getLoss()[-1]

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

    def log_estimated_labels(self, path):
        self._target.log_estimated_labels(path)

    def log_estimated_labels_json(self, dir_name, **kwargs):
        self._target.log_estimated_labels_json(dir_name, **kwargs)

    # TODO : doesn't work for frame_CUDA and Spike
    # TODO : also doesn't work with the current structure of layers and sequences !
    # def show_outputs(self):
    #     string = "Cells outputs :\n###############\n"
    #     for cell in self._deepnet.get_cells():
    #         string += cell.getName() + ": "
    #         for output in cell.getOutputs():
    #             string += str(output) + " "
    #         string += "\n"
    #     print(string)



class Extractor(Application):
    def __init__(self, provider, cell):
        deepnet_endpoint = cell.get_last()
        if not isinstance(deepnet_endpoint, n2d2.cell.Cell):
            raise RuntimeError("The deepnet endpoint is not a single cell")
        Application.__init__(self, provider, deepnet_endpoint.get_deepnet())

    def read_random_batch(self, partition):
        self._provider.read_random_batch(partition=partition)

    def read_batch(self, partition, idx):
        self._provider.read_batch(partition=partition, idx=idx)

    def process(self):
        # Propagate
        self._deepnet.propagate(inference=True)

