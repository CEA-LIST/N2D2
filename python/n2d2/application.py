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
    def __init__(self, model):
        self._model = model

    def convert_to_INI(self, filename):
        n2d2.utils.convert_to_INI(filename, self._provider.get_database(), self._provider, self._model, self._target)


class Classifier(Application):
    def __init__(self, provider, model):
        super().__init__(model)

        self._provider = provider

        print("Add provider")
        self._model.add_provider(self._provider)

        print("Create target")
        self._target = n2d2.target.Score('softmax.Target', self._model.get_output(), self._provider)

        print("Initialize model")
        self._model.initialize()

        self._mode = 'Test'

    def set_mode(self, mode):
        if mode not in self._provider.get_database().StimuliSets:
            raise RuntimeError("Mode " + mode + " not compatible with database stimuli sets")
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

