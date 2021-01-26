import N2D2


net = N2D2.Network()
deepNet = N2D2.DeepNet(net)
database = N2D2.MNIST_IDX_Database()
database.load("/nvme0/DATABASE/MNIST/raw/")
stimuli = N2D2.StimuliProvider(database, [28, 28, 1], 1, False)

parent = N2D2.ConvCell_Frame_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1])
pool = 	N2D2.PoolCell_Frame_float(deepNet, "pool1", [3, 3], 16, strideDims=[2, 2])


default_mapping = N2D2.MappingGenerator.Mapping(1,1,1,1,1,1,1)

iniParser = N2D2.IniParser()

t_map = N2D2.MappingGenerator.generate(stimuli, parent, 16, iniParser, "", "", default_mapping)

parent.addInput(stimuli)
pool.addInput(parent, t_map)

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


print(t_map)