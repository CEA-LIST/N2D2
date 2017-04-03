# N2D2 executables and application examples

Main executables
----------------

### `n2d2`

The main N2D2 binary, which allows to run DNN learning, testing, benchmarking
and export.

Just run `./n2d2 -h` to get the full list of program options.

#### `n2d2.sh`

A shell helper for launching `n2d2` processes in sub-directory.

### `n2d2_live`

This binary allows you to run a classification DNN live from a webcam or a
video.
It works with 1D output layer (generally softmax or fully-connected) networks
equiped with a `Target` object. See the application examples for a
use-case.

### `n2d2_live_fcnn`

This binary allows you to run a segmentation and classification DNN of type
 "fully-CNN" live from a webcam or a video. It works with 2D output layer
 networks equiped with a `TargetROIs` object. See the application examples for
 a use-case.


Application examples
--------------------

The following application examples are provided:

### `AppFaceDetection/`

A live face detection application, with gender recognition, based on the
[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset. You
will need a webcam supported by OpenCV to run this application live, or you can
run it on a video file.


### `AppObjectRecognition/`

A live object recognition application, based on 
[ILSVRC2012 (ImageNet)](http://www.image-net.org/challenges/LSVRC/2012/) dataset.
You will need a webcam supported by OpenCV to run this application live, or you 
can run it on a video file.


### `AppRoadDetection/`

A road segmentation application, based on the
[KITTI Road](http://www.cvlibs.net/datasets/kitti/eval_road.php) dataset.


Spike-based simulations
-----------------------

### `aer_cars`

This simulation implements STDP unsupervised learning on a recorded AER
sequence of cars running on a highway, using the event-based simulator embedded
into N2D2.

This binary reproduces some of the results published in 
[@Bichler2011](#Bichler2011).


### `aer_viewer`

This binary is a simple DVS128 format AER viewer.


References
----------

[@Bichler2011]: <a id="Bichler2011"></a>O. Bichler, D. Querlioz, S. Thorpe, J.
Bourgoin, and C. Gamrat. Extraction of temporally correlated features from
dynamic vision sensors with spike-timing-dependent plasticity. Neural Networks,
 **32**:339-348, 2012. doi:[10.1016/j.neunet.2012.02.022]
 (http://dx.doi.org/10.1016/j.neunet.2012.02.022).
