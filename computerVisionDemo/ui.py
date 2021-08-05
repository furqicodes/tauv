import cv2 as cv
import time

minarea = 1000


def coordinate_conversion(capture, bounding_box):
    height, width, channels = capture.shape
    x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
    cart = [0, 0, 0, 0]
    cart[0] = width / 2 - 1  # x_ax_pos
    cart[1] = height / 2 - 1  # y_ax_pos
    cart[2] = (x + w / 2) - cart[0]  # x
    cart[3] = - (y + h / 2) + cart[1]  # y
    print(cart)


def find_center(array):
    x, y, w, h = array
    x_center = (x + w / 2) - 1
    y_center = -(y + h / 2) - 1
    return x_center, y_center


def draw_overlay(screen, status, bounding_box):
    bbar_height = 48
    font = cv.FONT_HERSHEY_SIMPLEX
    height, width, channels = screen.shape
    center_point = [int(width / 2), int(height / 2)]
    cv.rectangle(screen, (0, 0), (width, bbar_height), (0, 0, 0), -1)

    # write processing durations
    cam = round(exec_time[0] * 1000, 0)
    inference = round(exec_time[1] * 1000, 0)
    other = round(exec_time[2] * 1000, 0)
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(cam, inference, other)
    cv.putText(screen, text_dur, (int(width / 4) - 30, int(bbar_height / 2 + 8)), font, 1, (255, 255, 255), 1)
    # text yerleşimi genelleştirilecek !

    # write FPS
    total_duration = cam + inference + other
    fps = round(1000 / total_duration, 1)
    text1 = 'FPS: {}'.format(fps)
    cv.putText(screen, text1, (10, 32), font, 1, (150, 150, 255), 2)

    # draw black rectangle at bottom
    cv.rectangle(screen, (0, height - bbar_height), (width, height), (0, 0, 0), -1)

    # write deviations and tolerance
    str_tol = 'Status : {}'.format(status)  # change placeholder with tolerance
    cv.putText(screen, str_tol, (10, height - 8), font, 1, (150, 150, 255), 2)

    # draw center cross lines
    cv.rectangle(screen, (0, int(height / 2) - 1), (width, int(height / 2) + 1), (255, 0, 0), -1)
    cv.rectangle(screen, (int(width / 2) - 1, 0 + bbar_height), (int(width / 2) + 1, height - bbar_height), (255, 0, 0), -1)

    ## draw the center red dot on the object
    #cv.circle(screen, (int(arr_track_data[0] * width), int(arr_track_data[1] * height)), 7, (0, 0, 255), -1)

    # draw the tolerance box
    tolerance = 0.1
    cv.rectangle(screen, (int(width / 2 - tolerance * width), int(height / 2 - tolerance * height)),
                           (int(width / 2 + tolerance * width), int(height / 2 + tolerance * height)), (0, 255, 0), 2)

    x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
    if status == "Tracking":
        cv.rectangle(screen, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
        cv.arrowedLine(screen, (center_point[0], center_point[1]), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 3)

    # cart = [0, 0, 0, 0]
    # cart[0] = width / 2 - 1  # x_ax_pos
    # cart[1] = height / 2 - 1  # y_ax_pos
    # cart[2] = (x + w / 2) - cart[0]  # x
    # cart[3] = - (y + h / 2) + cart[1]  # y
    # print(cart)


cap = cv.VideoCapture(0)
tracker = cv.TrackerKCF_create()
success, frame = cap.read()
bbox = cv.selectROI("Tracking", frame, True)
tracker.init(frame, bbox)

#processing time için geçen süre arrayi
exec_time = [0, 0, 0]

while True:
    start_time = time.time()

    # ----------------Capture Camera Frame-----------------
    start_t0 = time.time()
    success, frame = cap.read()
    if not success:
        break

    # cv2_im = frame
    # cv2_im = cv.flip(cv2_im, 0)
    # cv2_im = cv.flip(cv2_im, 1)
    # cv.cvtColor(cv2_im, cv.COLOR_BGR2RGB)

    height, width, channels = frame.shape
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    exec_time[0] = time.time() - start_t0
    # ----------------------------------------------------

    # -------------------Inference---------------------------------
    start_t1 = time.time()
    #buraya robot kontrolcülerinin initializationları gelecek
    exec_time[1] = time.time() - start_t1
    # ----------------------------------------------------

    # -----------------other------------------------------------
    start_t2 = time.time()
    #tracking here
    # success, frame = cap.read()
    # success, bbox = tracker.update(frame)
    #
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
    #
    # draw_overlay(frame, "Tracking" if success else "Lost", bbox)
    # coordinate_conversion(frame, bbox)
    # cv.imshow('TAUV', frame)


    ret, thresh = cv.threshold(gray, 127, 255, 1)

    contours, h = cv.findContours(thresh, 1, 2)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        x, y = approx[0][0]
        # print(len(approx))
        area = cv.contourArea(cnt)

        if len(approx) == 4 and minarea <= area < width * height * 0.8:
            M = cv.moments(cnt)
            # print( M )

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print("rectangle")
            print("x: ", cx, "y: ", cy)
            print("**")
            cart = [0, 0, 0, 0]
            cart[0] = width / 2 - 1  # x_ax_pos
            cart[1] = height / 2 - 1  # y_ax_pos
            cart[2] = cx - cart[0]  # x
            cart[3] = - cy + cart[1]  # y
            print(cart)

            cv.putText(frame, "X: " + str(cart[2]) + " Y: " + str(cart[3]), (cx + 20, cy + 20),
                        cv.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)

            cv.drawContours(frame, [cnt], 0, 255, -1)
            cv.circle(frame, (cx, cy), 1, (0, 0, 255), 3)
            # cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        else:
            # dikdortgen bulunana kadar ne yapilacak onu bul
            pass
    cv.imshow("frame", frame)

    exec_time[2] = time.time() - start_t2
    fps = round(1.0 / (time.time() - start_time), 1)

cap.release()
cv.destroyAllWindows()