from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from os.path import splitext, basename, join

class Colorizer:
    def __init__(self, height=480, width=600):
        (self.height, self.width) = height, width

        self.colorModel = cv2.dnn.readNetFromCaffe('./video-colorizer/models/colorization_deploy_v2.prototxt', caffeModel = "./video-colorizer/models/colorization_release_v2.caffemodel")
        

        clusterCenters = np.load("./video-colorizer/models/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2,313,1,1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1,313], 2.606, np.float32)]

    # colorize image
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width, self.height))

        self.processFrame()
        cv2.imwrite(join("output", basename(imgName)), self.imgFinal)

        cv2.imshow("output", self.imgFinal)

    # colorize video
    def processVideo(self, videoName):
        count = 1
        cap = cv2.VideoCapture(videoName)

        if(cap.isOpened() == False):
            print('Error opeing video')
            return

        (success, self.img) = cap.read()

        prevFrameTime = time.time()
        nextFrameTime = 0

        out = cv2.VideoWriter(join("./video-colorizer/output", splitext(basename(videoName))[0] + '.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), (self.width * 2, self.height), True)

        while success:
            print("colorizing frame" , count)
            count += 1
            
            self.img = cv2.resize(self.img, (self.width, self.height))

            ## testing each input frame
            # cv2.imshow("input", self.img)

            self.processFrame()  

            out.write(self.imgFinal)
            
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break;
            (success, self.img) = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # colorize a single image or a frame
    def processFrame(self):
        imgNormalized = (self.img[:, :, [2,1,0]] * 1.0 / 255).astype(np.float32)

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:,:,0]

        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224,224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -= 50
 
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))

        resultResized = cv2.resize(result, (self.width, self.height))

        self.imgOut = np.concatenate((channelL[:,:,np.newaxis], resultResized), axis=2)
        self.imgOut = np.clip(cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0 , 1)

        self.imgOut = np.array((self.imgOut)*255, dtype=np.uint8)

        self.imgFinal = np.hstack((self.img, self.imgOut))

