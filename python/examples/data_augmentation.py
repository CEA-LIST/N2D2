import n2d2
import matplotlib.pyplot as plt
import argparse


# ARGUMENTS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help='Path to the MNIST Dataset')
args = parser.parse_args()

def plot_tensor(tensor, path):
    plt.imshow(tensor[0,0,:], cmap='gray', vmin=0, vmax=255)
    plt.savefig(path)

database = n2d2.database.MNIST(data_path=args.data_path, validation=0.1)
provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=1)

database.get_partition_summary()

image = provider.read_batch(idx=0).to_numpy() * 255
# next function work too !
# image = next(provider).to_numpy() * 255

plot_tensor(image, "first_stimuli.png")

# Note : add_transformation would not have change the image as it has already been loaded 
provider.add_on_the_fly_transformation(n2d2.transform.Flip(vertical_flip=True))

image = provider.read_batch(idx=0).to_numpy() * 255
plot_tensor(image, "first_stimuli_fliped.png")

# negating the first transformation with another one
provider.add_transformation(n2d2.transform.Flip(vertical_flip=True))
image = provider.read_batch(idx=1).to_numpy() * 255
plot_tensor(image, "second_stimuli.png")
print("Second stimuli label :", provider.get_labels()[0])