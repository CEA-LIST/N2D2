import N2D2
import numpy
if __name__ == "__main__":
    # number of steps between reportings
    report = 50
    # number of static backprop learning steps
    learn = 500
    # perform static testing
    test = False
    # average window to compute success rate during learning
    avgWindow = 5

    batchSize = 10

    # Need to initialise the network before the database to generate a seed.
    # Otherwise, you can use this function to set a seed.
    # N2D2.mtSeed(0)
    net = N2D2.Network()
    deepNet = N2D2.DeepNet(net) # Proposition : overload the constructor to avoid passing a Network object
    # Import MNIST database
    database = N2D2.MNIST_IDX_Database()

    database.load("/local/is154584/cm264821/mnist")

    # env = N2D2.Environment(net, database, [24, 24, 1])
    
    stimuli = N2D2.StimuliProvider(database, [24, 24, 1], batchSize, False)
    
    # Network topology ===========================================================================================================

    conv1 = N2D2.ConvCell_Frame_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    conv2 = N2D2.ConvCell_Frame_float(deepNet, "conv2", [5, 5], 24, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    fc1 = N2D2.FcCell_Frame_float(deepNet, "fc1", 150, N2D2.TanhActivation_Frame_float())
    fc2 = N2D2.FcCell_Frame_float(deepNet, "fc2", 10, N2D2.TanhActivation_Frame_float())


    # Add transformation to the data ====================================================================
    trans = N2D2.DistortionTransformation()
    trans.setParameter("ElasticGaussianSize", "21")
    trans.setParameter("ElasticSigma", "6.0")
    trans.setParameter("ElasticScaling", "36.0")
    trans.setParameter("Scaling", "10.0")
    trans.setParameter("Rotation", "10.0")
    
    # Ne semble pas fonctionner car CompositeTransformation != DistortionTransformation ...
    # ou pas ... le problème semble similaire à l'erreur lors de l'initialistion de ConvCell

    # stimuli.addTransformation(N2D2.PadCropTransformation(24, 24), database.StimuliSetMask)
    # stimuli.addOnTheFlyTransformation(trans, database.StimuliSetMask)

    # Connect cells ======================================================================================


    # conv2mapping = [
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
    # conv2mapping = [
    #     [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True],
    #     [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True],
    #     [False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True],
    #     [False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True],
    #     [False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True],
    #     [False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True],
    #     [False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True],
    #     [False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True],
    #     [False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True],
    #     [False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, False, True, True],
    #     [False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, True, True],
    #     [False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, False, True, True],
    #     [False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, True, True],
    #     [False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True, True],
    #     [False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True],
    #     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True, True]]

    conv2mapping = [
        True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True, True]

    t_con2mapping = N2D2.Tensor_bool(conv2mapping)

    conv1.addInput(stimuli)
    conv2.addInput(conv1, t_con2mapping)
    fc1.addInput(conv2)
    fc2.addInput(fc1)

    # print("Learning database size ", database.getNbStimuli(Database.Learn), ' images')

    conv1.initialize()
    conv2.initialize()
    fc1.initialize()
    fc2.initialize()

    # learning process ===================================================================================

    