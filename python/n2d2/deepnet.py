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
import n2d2.global_variables
import n2d2.cells.nn
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.utils import generate_name

class DeepNet(N2D2_Interface):

    _convention_converter= n2d2.ConventionConverter({
        "name": "Name",
    })

    def __init__(self, **config_parameters):
        self._provider = None
        N2D2_Interface.__init__(self, **config_parameters)

        self._config_parameters['name'] = generate_name(self)

        self._set_N2D2_object(N2D2.DeepNet(n2d2.global_variables.default_net))
        self._set_N2D2_parameters(self._config_parameters)

        self.load_N2D2_parameters(self.N2D2())

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object, **_):
        deepnet = super().create_from_N2D2_object(N2D2_object)

        deepnet._config_parameters['name'] = "DeepNet(id=" + str(id(deepnet)) + ")"

        deepnet._N2D2_object = N2D2_object
        deepnet.load_N2D2_parameters(deepnet.N2D2())

        cells = deepnet.N2D2().getCells()
        layers = deepnet.N2D2().getLayers()
        if not layers[0][0] == "env":
            print("Is env:" + layers[0][0])
            raise RuntimeError("First layer of N2D2 deepnet is not a StimuliProvider. You may be skipping cells")

        deepnet._cells = {}

        for layer in layers[1:]:
            for cell in layer:
                N2D2_cell = cells[cell]
                n2d2_cell = n2d2.converter.from_N2D2_object(N2D2_cell, n2d2_deepnet=deepnet)
                deepnet._cells[n2d2_cell.get_name()] = n2d2_cell

        return deepnet

    def back_propagate(self):
        self._N2D2_object.backPropagate()

    def update(self):
        self._N2D2_object.update()

    def set_provider(self, provider):
        self._provider = provider
        self._N2D2_object.setStimuliProvider(provider.N2D2())

    def get_provider(self):
        return self._provider

    def get_cells(self):
        return self._cells

    def remove(self, cell_name:str, reconnect:bool=False)->None:
        """Remove a cell from the encapsulated deepnet.

        :param name: Name of cell that shall be removed.
        :type name: str
        :param reconnect: If `True`, reconnects the parents with the child of the removed cell, default=True
        :type reconnect: bool, optional
        """
        cell = self.N2D2().getCells()[cell_name]
        self.N2D2().removeCell(cell, reconnect)
        self._cells.pop(cell_name)

    def get_input_cells(self):
        """
        Return the first N2D2 cell in the deepNet
        """
        output = []
        for cell in self.N2D2().getLayers()[1]:
            output.append(self._cells[cell])
        return output

    def get_output_cells(self):
        """
        Return the last N2D2 cell in the deepNet
        """
        output = []
        for cell in self.N2D2().getLayers()[-1]:
            output.append(self._cells[cell])
        return output

    def draw(self, filename):
        N2D2.DrawNet.draw(self._N2D2_object, filename)

    def draw_graph(self, filename):
        """Plot a graphic representation of the neural network.
        :param filename: path where you want to save the graph
        :type filename: str
        """
        N2D2.DrawNet.drawGraph(self._N2D2_object, filename)

    def export_network_free_parameters(self, dirname):
        """Export free parameters and create a plot of the distributions of these parameters
        :param dirname: path to the directory where you want to save the data.
        :type dirname: str
        """
        self.N2D2().exportNetworkFreeParameters(dirname)

    def log_stats(self, dirname):
        """Export statistics of the deepnet graph
        :param dirname: path to the directory where you want to save the data.
        :type dirname: str
        """
        self.N2D2().logStats(dirname)
