import N2D2
import numpy
from var import data_path # Hide variables.

if __name__ == "__main__":
    # number of steps between reportings
    report = 50
    # number of static backprop learning steps
    learn = 500
    # perform static testing
    test = False
    # average window to compute success rate during learning
    avgWindow = 5

    batchSize = 1

    # Need to initialise the network before the database to generate a seed.
    # Otherwise, you can use this function to set a seed.
    # N2D2.mtSeed(0)
    net = N2D2.Network()
    deepNet = N2D2.DeepNet(net) # Proposition : overload the constructor to avoid passing a Network object
    # Import MNIST database
    database = N2D2.MNIST_IDX_Database()
    database.load(data_path)
    print(database)    
    
    # Network topology ===========================================================================================================

    conv1 = N2D2.ConvCell_Frame_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    conv2 = N2D2.ConvCell_Frame_float(deepNet, "conv2", [5, 5], 24, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    fc1 = N2D2.FcCell_Frame_float(deepNet, "fc1", 150, N2D2.TanhActivation_Frame_float())
    fc2 = N2D2.FcCell_Frame_float(deepNet, "fc2", 10, N2D2.TanhActivation_Frame_float())


    # Add transformation to the data ====================================================================
    trans = N2D2.DistortionTransformation()
    # Why can't we set those parameters in the constructor ?
    trans.setParameter("ElasticGaussianSize", "21")
    trans.setParameter("ElasticSigma", "6.0")
    trans.setParameter("ElasticScaling", "36.0")
    trans.setParameter("Scaling", "10.0")
    trans.setParameter("Rotation", "10.0")
    


    stimuli = N2D2.StimuliProvider(database, [24, 24, 1], batchSize, False)
    # need an implicit transformation => CompositeTransformation
    stimuli.addTransformation(N2D2.CompositeTransformation(N2D2.PadCropTransformation(24, 24)), database.StimuliSetMask(0))
    stimuli.addOnTheFlyTransformation(N2D2.CompositeTransformation(trans), database.StimuliSetMask(0))

    # Connect cells ======================================================================================

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

    t_con2mapping = N2D2.Tensor_bool(numpy.array(conv2mapping))

    conv1.addInput(stimuli)
    conv2.addInput(conv1, t_con2mapping)
    fc1.addInput(conv2)
    fc2.addInput(fc1)

    print("Learning database size ", database.getNbStimuli(N2D2.Database.Learn), ' images')
    print("Testing database size ", database.getNbStimuli(N2D2.Database.Test), ' images')

    conv1.initialize()
    conv2.initialize()
    fc1.initialize()
    fc2.initialize()

    # learning process ===================================================================================
    success = []
    if learn > 0 :
        for i in range(learn):
            outputtensor = N2D2.Tensor_int([1,1,batchSize]) # initialize a tensor with 1 dim of size batchSize
            for iteration in range(batchSize):
                id = stimuli.readRandomStimulus(N2D2.Database.Learn)
                outputTarget = database.getStimulusLabel(id)
                conv1.propagate()
                conv2.propagate()
                fc1.propagate()
                fc2.propagate()
                outputEstimated = fc2.getMaxOutput()

                success.append(outputEstimated == outputTarget)

                outputtensor[iteration] = outputTarget

            fc2.setOutputTarget(outputtensor)
            fc2.backPropagate()
            fc1.backPropagate()
            conv2.backPropagate()
            conv1.backPropagate()

            conv1.update()
            conv2.update()
            fc1.update()
            fc2.update()
            if (i + 1) % report == 0 or i == learn - 1:
                if avgWindow > 0 and len(success) > avgWindow:
                    max_index = len(success) - 1
                    avgSuccess = sum(success[max_index-avgWindow:max_index]) / avgWindow

                    # TODO : Traduire ceci en python
                    # const double avgSuccess
                    # = (avgWindow > 0 && success.size() > avgWindow)
                    #       ? std::accumulate(success.end() - avgWindow,
                    #                         success.end(),
                    #                         0.0) / (double)avgWindow
                    #       : std::accumulate(success.begin(), success.end(), 0.0)
                    #         / (double)success.size();
                else:
                    avgSuccess = sum(success) / len(success)
                print("Learning #", i, "  ", 100.0 * avgSuccess, " %")
            if i == learn - 1:
                conv1.exportFreeParameters("conv1.syntxt")
                conv2.exportFreeParameters("conv2.syntxt")
                fc1.exportFreeParameters("fc1.syntxt")
                fc2.exportFreeParameters("fc2.syntxt")
                print('THE END')