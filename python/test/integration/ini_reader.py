import N2D2
import n2d2


class IniReader():
    """
    Quick class to create a deepNet from an ini file and to use it.
    """
    def __init__(self, path):
        net = N2D2.Network(1)
        self.deepNet = N2D2.DeepNetGenerator.generate(net, path)
        self.deepNet.initialize() 
        self.cells = self.deepNet.getCells()
        self.first_cell = self.cells[self.deepNet.getLayers()[1][0]]
        self.last_cell = self.cells[self.deepNet.getLayers()[-1][-1]]

    def forward(self, input_tensor):
        self.first_cell.clearInputs()

        shape = [i for i in reversed(input_tensor.dims())]
        diffOutputs = n2d2.Tensor(shape, value=0)
        self.first_cell.addInputBis(input_tensor.N2D2(), diffOutputs.N2D2())

        N2D2_inputs = self.deepNet.getCell_Frame_Top(self.first_cell.getName()).getInputs(0)
        N2D2_inputs.op_assign(input_tensor.N2D2())
        N2D2_inputs.synchronizeHToD()

        self.deepNet.propagate(N2D2.Database.Learn, False, [])

        outputs = self.deepNet.getCell_Frame_Top(self.last_cell.getName()).getOutputs() 
        outputs.synchronizeDToH()
        return n2d2.Tensor.from_N2D2(outputs)
    
    def backward(self):
        # TODO 
        pass