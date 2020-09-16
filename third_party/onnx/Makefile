all:
	wget https://raw.githubusercontent.com/onnx/onnx/master/onnx/onnx.proto3 -O onnx.proto3.wget
	cmp -s onnx.proto3 onnx.proto3.wget || (cp onnx.proto3.wget onnx.proto3 && $(MAKE) proto)

proto: onnx.proto3
	protoc -I=. --cpp_out=. onnx.proto3
	mv onnx.proto3.pb.h onnx.pb.h
	# Patch bug in protobuf (Unused parameter 'deterministic' #2032)
	sed -i 's/bool deterministic, ::google::protobuf::uint8\* target) const {/bool \/*deterministic*\/, ::google::protobuf::uint8* target) const {/g' onnx.proto3.pb.cc
	# Correct include in generated file and move it to the correct location
	sed -i 's/#include "onnx.proto3.pb.h"/#include <onnx.pb.h>/g' onnx.proto3.pb.cc
	# Add ONNX guard
	sed -i '1i#ifdef ONNX' onnx.proto3.pb.cc
	sed -e "\$$a#endif" onnx.proto3.pb.cc > onnx.pb.cpp
	rm -f onnx.proto3.pb.cc
