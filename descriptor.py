import cv2
class descriptorVC():

    def creacionHOGDescriptor():
        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (10, 10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True
        descrip = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                       histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
        return descrip
