import n2d2
import math

# Path to MNIST dataset (search for "mnist")
data_path_mnist = "path_to_mnist"

# Change default model
n2d2.global_variables.default_model = "Frame"

# Create dataloader
database = n2d2.database.MNIST(data_path=data_path_mnist, validation=0.2)
database.get_partition_summary()

# Create provider
provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=128)
provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))

# Load model
model = n2d2.models.lenet.LeNet(10)

batch_size = 128
softmax = n2d2.cells.nn.Softmax(with_loss=True)
target = n2d2.target.Score(provider)

solver = n2d2.solver.SGD(
    learning_rate=0.05, 
    momentum=0.9,
    decay=0.0005,
    learning_rate_policy="StepDecay",
    learning_rate_step_size=48000,
    learning_rate_decay=0.993
    )
model.set_solver(solver)

# Convert model to deepnet_cell for training and export
model = model.to_deepnet_cell(provider)

print("\n### Training ###")
for epoch in range(5):

    provider.set_partition("Learn")
    model.learn()

    print("\n# Train Epoch: " + str(epoch) + " #")

    for i in range(math.ceil(database.get_nb_stimuli('Learn')/batch_size)):

        x = provider.read_random_batch()
        x = model(x)
        x = softmax(x)
        x = target(x)

        x.back_propagate()
        x.update()

        print("Example: " + str(i * batch_size) + ", loss: " 
              + "{0:.3f}\r".format(x[0]), end="\r")


    print("\n### Validation ###")

    target.clear_success()
    
    provider.set_partition('Validation')
    model.test()

    for i in range(math.ceil(database.get_nb_stimuli('Validation') / batch_size)):
        batch_idx = i * batch_size

        x = provider.read_batch(batch_idx)
        x = model(x)
        x = softmax(x)
        x = target(x)

        print("Validate example: " + str(i * batch_size) + ", val success: " 
              + "{0:.2f}".format(100 * target.get_average_success()) + "%", end="\r")


print("\n\n### Testing ###")

provider.set_partition('Test')
model.test()
target.clear_success()
for i in range(math.ceil(provider.get_database().get_nb_stimuli('Test')/batch_size)):
    batch_idx = i*batch_size

    x = provider.read_batch(batch_idx)
    x = model(x)
    x = softmax(x)
    x = target(x)

    print("Example: " + str(i * batch_size) + ", test success: "
          + "{0:.2f}".format(100 * target.get_average_success()) + "%", end="\r")

print("\n")

# Export model
n2d2.export.export_cpp_stm32(
    model,
    provider,
    nb_bits=8, 
    calibration=-1,
    export_nb_stimuli_max=10
    )
