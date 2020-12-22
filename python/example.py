# import pylab
import N2D2

a = N2D2.Tensor_float([1, 2, 3])
print(a)
b = a.dimX()
print(b)

a.resize([4, 5, 6])
print(a.size())
print(a.dimX())

print(len(a))

a[0:4] = 1.0

[print(x) for x in reversed(a)]

print(numpy.array(a))


print(a.empty())


b = N2D2.CudaTensor_float(numpy.array([[1.0, 2.0], [3.0, 4.0]]))
print(b.dims())
[print(x) for x in b]
print(numpy.array(b))

net = N2D2.Network(1)
deepNet = N2D2.DeepNetGenerator.generate(net, "../model/mnist24_16c4s2_24c5s2_150_10.ini")
deepNet.initialize()
sp = deepNet.getStimuliProvider()
sp.readBatch(N2D2.Database.Test, 0)

#pylab.imshow(numpy.array(sp.getData(0)))
#pylab.show()

print(numpy.array(sp.getLabelsData(0)))

deepNet.test(N2D2.Database.Test, [])
target = deepNet.getTargets()[0]

print(numpy.array(target.getEstimatedLabels()))
print(numpy.array(target.getEstimatedLabelsValue()))


