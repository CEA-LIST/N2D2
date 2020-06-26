#!/usr/bin/env python
import os, sys
import optparse
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
import timeit
import n2d2_tensorRT_inference
np.set_printoptions(threshold=1000000000)

total_time = 0.0
successRate = 0.0
stimulus = "/nvme0/DATABASE/KAIST_multispectral/images/set03/V001/lwir/"

def main():

    device = 0
    iterBuild = 1
    batchSize = 1
    profiling = True
    bitPrecision = -32

    print("Initialization of the N2D2 tensorRT network...")
    deepmanta_nn = n2d2_tensorRT_inference.n2d2_tensorRT_inference(batchSize, device, iterBuild, bitPrecision, profiling, "", "n2d2_tensorRT_model.dat", False)
    print("Initialization of the N2D2 tensorRT network done")

    inputSize = deepmanta_nn.inputDimX()*deepmanta_nn.inputDimY()*deepmanta_nn.inputDimZ()

    dimX = deepmanta_nn.inputDimX()
    dimY = deepmanta_nn.inputDimY()

    #automatic convert from vector to python list
    outDimX = deepmanta_nn.outputDimX()
    outDimY = deepmanta_nn.outputDimY()
    outDimZ = deepmanta_nn.outputDimZ()
    outTarget = deepmanta_nn.outputTarget()


    #if outDimX[0] > 1 or outDimY[0] > 1:

    ROIsDim = outDimX[0]*outDimY[0]*outDimZ[0]

    ROIs = np.zeros(ROIsDim*batchSize, dtype=np.float32)
    NbProposal=int(outDimX[0]*outDimY[0]*outDimZ[0]/6)

    print('execute for ', NbProposal, ' proposals')
    idx = 1
    rate_elapsed = 0.0
    for file in sorted(os.listdir(stimulus)):
        image = Image.open(stimulus + file)
        image_width, image_height = image.size
        print('[ ', image_width, ', ', image_height, ']')
        image_i8 = []

        image_i8.append(np.array(image.crop((0,0,dimX,dimY))))


        image_b0_f32 = np.array(image.crop((0,0,dimX,dimY)), dtype=np.float32)

        image_b0_f32 =  ((image_b0_f32 / 150.0) - 0.5)*1.0

        image_b0_1D = np.concatenate((image_b0_f32[:,:,2], image_b0_f32[:,:,1], image_b0_f32[:,:,0]), axis=None)
        #print(image_b0_1D)
        im0 = []
     
        im0.append(image_b0_1D)

        #image_compute = np.concatenate((im0, im1, im2, im3), axis=None)

        image_compute = np.concatenate((im0), axis=None)

        tic = timeit.default_timer()
        #execute neural network
        deepmanta_nn.execute(image_compute)
        deepmanta_nn.getOutput(ROIs, 0)
       # deepmanta_nn.logOutput(0)
       # deepmanta_nn.logOutput(1)
        toc = timeit.default_timer()
        
        print('detection done in ', toc - tic, ' sec')
        rate_elapsed = 1 /(toc - tic)
        print('Current frame rate: ', rate_elapsed, ' fps')

        final_boxes_list = []
        final_label_list = []
        final_scores_list = []

        for b in range(0, batchSize):
            batchOffset = b*ROIsDim
            boxes = np.zeros((NbProposal,4))
            print(ROIs)
            boxes[:,0]=ROIs[1 + batchOffset:ROIsDim + batchOffset: 6] #y0
            boxes[:,1]=ROIs[0 + batchOffset:ROIsDim + batchOffset: 6] #x0
            boxes[:,2]=ROIs[1 + batchOffset:ROIsDim + batchOffset: 6] + ROIs[3 + batchOffset:ROIsDim + batchOffset: 6] #h0
            boxes[:,3]=ROIs[0 + batchOffset:ROIsDim + batchOffset: 6] + ROIs[2 + batchOffset:ROIsDim + batchOffset: 6] #w0
            scores=ROIs[batchOffset + 4:ROIsDim + batchOffset: 6] #labels
            labels=ROIs[batchOffset + 5:ROIsDim + batchOffset: 6] #labels
            indexInput = 0
            indexOutput = 0
            boxes_final = []
            scores_final = []
            labels_final = []

            for r_1 in boxes:
                if (r_1[1] + r_1[3]) and (r_1[0] + r_1[1]) > 0.0 :
                    boxes_final.append(r_1)
                    scores_final.append(scores[indexInput].astype(np.float32))
                    labels_final.append(labels[indexInput].astype(int) + 1)
                    indexOutput += 1
                indexInput += 1
            boxes_final = np.reshape(boxes_final, (indexOutput, 4))
            scores_final = np.reshape(scores_final, (indexOutput))
            labels_final = np.reshape(labels_final, (indexOutput))

            final_label_list.append(labels_final)
            final_boxes_list.append(boxes_final)
            final_scores_list.append(scores_final)

        for b in range(0, batchSize):
            for n in range (0, final_boxes_list[b].shape[0]):
                r_1 = final_boxes_list[b][n]
                if(final_scores_list[b][n] > 0.99):
                    cv.rectangle(image_i8[b], (int(r_1[1]), int(r_1[0])), (int(r_1[3]), int(r_1[2])), [255, 0, 0], 2)
                print(r_1)
                print(final_scores_list[b][n])
        display_img_0 = cv.cvtColor(image_i8[0], cv.COLOR_RGB2BGR)

        resize_image_0 = cv.resize(display_img_0, (0, 0), fx=1.0, fy=1.0)

        cv.namedWindow('ssDM - GPU')

        cv.imshow('ssDM - GPU - b0 - GPU', resize_image_0)


        cv.moveWindow('DeepManta - v3 on N2D2 - GPU', 0, 100)

        key = cv.waitKey(0)
        if key == 27:  # exit on ESC
            if profiling :
                deepmanta_nn.getProfiling(idx)
            break
        idx += 1

if __name__ == "__main__":
    sys.exit(main())