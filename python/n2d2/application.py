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


class CrossEntropyClassifier:
    def __init__(self, provider, **target_config_parameters):
        self._softmax = n2d2.cells.nn.Softmax(with_loss=True)
        self._target = n2d2.target.Score(provider, **target_config_parameters)

    def __call__(self, inputs):
        x = self._softmax(inputs)
        self._target(x)
        #self._target.provide_targets()
        #self._target.process()
        loss = n2d2.Tensor(dims=[1], value=self.get_current_loss(), cell=self)
        loss._leaf = True
        return loss

    def get_deepnet(self):
        return self._softmax.get_deepnet()

    def get_average_success(self, window=0):
        return self._target.get_average_success(window=window)

    def clear_success(self):
        self._target.clear_success()

    def get_average_score(self, metric):
        return self._target.get_average_score(metric=metric)

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

    def recognition_rate(self):
        """
        return the recognition rate
        """
        return self._target.get_average_success()


    def log_success(self, path):
        """
        Save a graph of the loss and the validation score as a function of the step number.
        """
        self._target.log_success(path)

    def log_stats(self, path):
        """
        Save computational stats on the cells.
        """
        self.get_deepnet().N2D2().logStats(path)

    def log_estimated_labels(self, path):
        self._target.log_estimated_labels(path)

    def log_estimated_labels_json(self, dir_name, **kwargs):
        self._target.log_estimated_labels_json(dir_name, **kwargs)

    def log_confusion_matrix(self, path):
        self._target.log_confusion_matrix(path)

    # TODO : doesn't work for frame_CUDA and Spike
    # TODO : also doesn't work with the current structure of layers and sequences !
    # def show_outputs(self):
    #     string = "Cells outputs :\n###############\n"
    #     for cells in self._deepnet.get_cells():
    #         string += cells.getName() + ": "
    #         for output in cells.getOutputs():
    #             string += str(output) + " "
    #         string += "\n"
    #     print(string)

