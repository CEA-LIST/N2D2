"""
Example of exporting a ResNet18 model to any board which can support C++

Download the onnx from 
https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx
"""

import n2d2

# Path to ImageNet dataset (search for "ILSVRC2012")
data_path_imagenet = "path_to_imagenet"

# Change default model
n2d2.global_variables.default_model = "Frame_CUDA"

# Create dataloader
database = n2d2.database.ILSVRC2012(learn=1.0, random_partitioning=True)
database.load(data_path_imagenet, label_path=f"{data_path_imagenet}/synsets.txt")

# Create provider
provider = n2d2.provider.DataProvider(database, [224, 224, 3], batch_size=32)

# Apply transformations
transformations = n2d2.transform.Composite([
    n2d2.transform.Rescale(256, 256, keep_aspect_ratio=False, resize_to_fit=False), 
    n2d2.transform.PadCrop(224, 224),
    n2d2.transform.RangeAffine("Divides", 255.0),
    n2d2.transform.ColorSpace("RGB"),
    n2d2.transform.RangeAffine("Minus", [0.485, 0.456, 0.406], second_operator="Divides", second_value=[0.229,0.224,0.225])
])

provider.add_transformation(transformations)

# Load model
model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, "resnet18-v1-7.onnx")

# Export model
n2d2.export.export_cpp(
    model, 
    provider,
    nb_bits=8, 
    calibration=-1, 
    export_no_unsigned=True, 
    export_nb_stimuli_max=10
    )
