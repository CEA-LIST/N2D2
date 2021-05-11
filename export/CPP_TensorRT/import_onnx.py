from PIL import Image
from resizeimage import resizeimage
import numpy as np
import N2D2 as n2d2
import os, sys
import optparse
import matplotlib.pyplot as plt
import cv2 as cv

ImageInputPath = "/home/db246302/Documents/SuperResolution/Input_images/toto.jpeg"
ONNXModel = "/home/db246302/Documents/SuperResolution/ONNX_model/super-resolution-10.onnx"
ImageOutputPath = " "
profiling = True

def preprocess(img_path):
    input_shape = (1, 3, 960, 320)
    img = Image.open(img_path)
    img = img.resize((960, 320), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data



def main():
    # loading input and resize if needed
    #image = Image.open(ImageInputPath)
    #size_reduction_factor = 1
    #image = image.resize((int(image.size[0] / size_reduction_factor), int(image.size[1] / size_reduction_factor)), Image.ANTIALIAS)

    # Preprocess image
    #x = np.array(image).astype('float32')
    #x = np.transpose(x, [2, 0, 1])
    img = preprocess(ImageInputPath)

    orig_img = Image.open(ImageInputPath)
    img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    #img_5_1D = 

    img_1D = np.expand_dims(img_5, axis=0)

    print("---------------N2D2-TensorRT DNN Initialization----------------------------")
    network = n2d2.N2D2_Network()
    network.setPrecision(-32)
    network.setDeviceID(0)
    network.setMaxBatchSize(1)
    network.setInputEngine("")
    network.setOutputEngine("ssdm.dat")
    network.setCalibCache("")
    network.setCalibFolder("")
    network.setInputDims(960, 320, 3)
    network.setOutputNbTargets(3)
    network.setOutputTarget(1, 1 , 672, 672, 0)

    network.setONNXModel("ssdm.onnx")
    network.initialize()

    dimX = network.getInputDimX()
    dimY = network.getInputDimY()
    outDimX = []
    outDimY = []
    outDimZ = []
    outTarget = []
    
    for d in range(0, network.getOutputNbTargets()):
        outDimX.append(network.getOutputDimX(d))
        outDimY.append(network.getOutputDimY(d))
        outDimZ.append(network.getOutputDimZ(d))
        outTarget.append(network.getOutputTarget(d))
    print([outDimX])
    print([outDimY])
    print([outDimZ])
    print([outTarget])
    print(dimX, ", ", dimY)

    outputImage = np.zeros(network.getOutputDimX(0) * network.getOutputDimY(0) * network.getOutputDimZ(0), dtype=np.float32)

    if profiling == True:
        network.setProfiling()

    print("Process Image...")
    network.syncExe(img_1D, 1)
    network.cpyOutput(outputImage, 0)
    print("Process Image Done! ")
    if profiling == True:
        network.reportProfiling(1)
    #print([outputImage])
    #outputImage = np.reshape(outputImage, (1, 1, 672,672))
    #print(outputImage.shape)


    key = cv.waitKey(0)

if __name__ == "__main__":
    sys.exit(main())