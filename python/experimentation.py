import N2D2
import n2d2

# Need to reproduce : n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -export CPP -fuse -nbbits 8 -calib -1 -db-export 100 -test

c = n2d2.cells.Fc(10,10, datatype="float")

t = n2d2.Tensor([1,1,10,10], datatype="float", cuda=True)

outputs = c(t)

N2D2.DeepNetExport.generate(outputs.get_deepnet().N2D2(), "./test_C_export", "C")
