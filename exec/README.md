# N2D2 executables and application examples

Executables
----------

### `n2d2`

The main N2D2 binary, which allow to run DNN learning, testing, benchmarking and
export.

Just run `./n2d2 -h` to get the full list of program options.

#### `n2d2.sh`

A shell helper for launching `n2d2` processes in sub-directory.

### `n2d2_live`

This binary allows you to run a classification DNN live from a webcam.
It works with 1D output layer (generally softmax or fully-connected) networks
equiped with a `Target` object. See the application examples for a
use-case.

### `n2d2_live_fcnn`

This binary allows you to run a segmentation and classification DNN of type
 "fully-CNN" live from a webcam. It works with 2D output layer networks equiped
 with a `TargetROIs` object. See the application examples for a use-case.


Application examples
--------------------

The following application examples are provided:

### `AppFaceDetection/`

A live face detection application, with gender recognition. You will need a
webcam supported by OpenCV to run this application live, or you can run it on
a video file.

