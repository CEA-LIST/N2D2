import N2D2
import n2d2

database = n2d2.database.MNIST(DataPath="/nvme0/DATABASE/MNIST/raw/", Validation=0.2)
provider = n2d2.provider.DataProvider(Database=database, Size=[28, 28, 1], BatchSize=1)


net = N2D2.Network()
deepNet = N2D2.DeepNet(net)
database = N2D2.MNIST_IDX_Database()
database.load("/nvme0/DATABASE/MNIST/raw/")
stimuli = N2D2.StimuliProvider(database, [28, 28, 1], 1, False)

parent = N2D2.ConvCell_Frame_CUDA_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1])
pool = 	N2D2.PoolCell_Frame_CUDA_float(deepNet, "pool1", [3, 3], 16, strideDims=[2, 2])
# TODO : update, curently I use converter because we don't use n2d2.DeepNet ...
n_parent = n2d2.converter.cell_converter(parent)

t_map = n2d2.mapping.get_mapping(n_parent, 16, 32)

parent.addInput(stimuli)
pool.addInput(parent, t_map.N2D2())

parent.initialize()
pool.initialize()

tar = N2D2.TargetScore('target', pool, stimuli)

stimuli.readRandomBatch(set=N2D2.Database.Learn)
tar.provideTargets(N2D2.Database.Learn)

# Propagate
parent.propagate()
pool.propagate()

# Process
tar.process(N2D2.Database.Learn)

# Backpropagate
parent.backPropagate()
pool.backPropagate()

# Update parameters by calling solver on gradients
parent.update()
pool.update()

