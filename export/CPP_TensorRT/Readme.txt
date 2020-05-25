How to install TensorRT export

1: 
	Go to export directory
2:
	make run WRAPPER_PYTHON=3.5m BOOST_NATIVE=1
	Now the tensorRT model file has been created as "n2d2_tensorRT_model.dat"
3:
	cd /usr/lib/x86_64-linux-gnu
4:
	ln -s /path/to/export/bin/n2d2_tensorRT_inference.so .

Test install:
	python3
>import n2d2_tensorRT_inference
>

Delete useless files:
sudo rm -rf bin.obj/ include/ Makefile  src/ *.py stimuli/
sudo rm -rf dnn/weights/ dnn/src/ dnn/Makefile dnn/include/ dnn/bin_dnn.obj/
