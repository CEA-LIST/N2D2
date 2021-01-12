import n2d2 
import N2D2

net = N2D2.Network()
deepNet = N2D2.DeepNet(net)

elemWiseCell= N2D2.ElemWiseCell_Frame(deepNet, "name", 10)

n_elemWiseCell = n2d2.converter.cell_converter(elemWiseCell)

print(n_elemWiseCell)
