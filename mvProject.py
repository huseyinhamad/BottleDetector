from os import listdir
import numpy as np
import cv2

def preprocessImage(image, sigma=0.33):
    bilateral = cv2.bilateralFilter(image, 256, 85, 85, cv2.BORDER_CONSTANT)
    blurredImage = cv2.GaussianBlur(bilateral, (5, 5), 0)
    cannyImage = cv2.Canny(blurredImage, int(max(0, (1 - sigma) * np.median(image))),
                           int(min(255, (1 - sigma) * np.median(image))))
    return cannyImage


def findCnt(cannyImage, drawnImage):
    contours, hierarchy = cv2.findContours(cannyImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for cnt in contours:
        if (cv2.contourArea(cnt) < 150):
            continue
        _, _, w, h = cv2.boundingRect(cnt)
        if (len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) > 5) and (0.9 < w / float(h) < 1.1):
            count += 1
            cv2.drawContours(drawnImage, [cnt], 0, 255, -1)

    cv2.putText(drawnImage, "Bottles found in image : " + str(int(count)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 1)
    cv2.imshow("Result", drawnImage)
    cv2.waitKey(0)
    return drawnImage


if "__main__" == __name__:
    for i, image in enumerate(listdir("images/")):
        image = cv2.imread("images/" + listdir("images/")[i], 0)

        cannyImage = preprocessImage(image)
        drawnImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        findCnt(cannyImage, drawnImage)