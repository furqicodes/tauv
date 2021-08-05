import numpy as np
import cv2
cap = cv2.VideoCapture(0)

minarea = 1000
while True:
    _, frame = cap.read()
    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,127,255,1)

    contours,h = cv2.findContours(thresh,1,2)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        x,y = approx[0][0]
        #print(len(approx))
        area = cv2.contourArea(cnt)


        if len(approx)==4 and area>=minarea and area<width*height*0.8:
            M = cv2.moments(cnt)
            #print( M )

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print("rectangle")
            print("x: ",cx,"y: ",cy)
            print("**")
            cart = [0, 0, 0, 0]
            cart[0] = width / 2 - 1  # x_ax_pos
            cart[1] = height / 2 - 1  # y_ax_pos
            cart[2] = (cx ) - cart[0]  # x
            cart[3] = - (cy) + cart[1]  # y
            print(cart)

            cv2.putText(frame, "X: " + str(cart[2]) + " Y: " + str(cart[3]), (cx + 20, cy + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)

            cv2.drawContours(frame,[cnt],0,255,-1)
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), 3)
            #cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        else:
            #dikdortgen bulunana kadar ne yapilacak onu bul
            pass
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()