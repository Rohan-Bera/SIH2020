import cv2
import numpy as np
import math
import heapq

kernel3 = np.ones((3,3), dtype=np.uint8)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def show(win_name, img):
    if __name__ == "__main__":
        cv2.imshow(win_name, img)

def get_hsv_mask(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array([0, 20, 0], dtype=np.uint8)
    upper_range = np.array([40, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_range, upper_range)
    return mask

def get_ycrcb_mask(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    lower_range = np.array([0, 134, 67], dtype=np.uint8)
    upper_range = np.array([255, 160, 123], dtype=np.uint8)    
    mask = cv2.inRange(ycrcb_img, lower_range, upper_range)
    return mask

def get_lab_mask(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lower_range = np.array([0, 130, 130], dtype=np.uint8)
    upper_range = np.array([255, 150, 188], dtype=np.uint8)
    mask = cv2.inRange(lab_img, lower_range, upper_range)
    return mask

	
def get_skin_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    light_level = cv2.mean(gray)[0]
    # print("light level:", light_level)
    if light_level < 50:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:    
        mask = get_hsv_mask(img)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    # remove face from mask
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.putText(img, "light: %.3f"%(light_level,), (30, img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1)
    # show("face_light", img)

    # mask cleanup
    mask = cv2.erode(mask, kernel3, iterations=2)    
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    
    # if light_level >= 50:
    #     # subtract edges    
    #     edges = cv2.Canny(img[:,:,2], 50, 180)    
    #     edges = cv2.dilate(edges, kernel3, iterations=1)
    #     # cv2.imshow("edges", edges)
    #     mask[edges==255] = 0

    return mask


def find_palm(contour, img_shape):
    contour_img = np.zeros(img_shape, np.uint8)
    cv2.drawContours(contour_img, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour) 
    h = min(int(w*3/2), h)
    contour_img = contour_img[y:y + h, x:x + w]

    dist_img = cv2.distanceTransform(contour_img, cv2.DIST_C, 3)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_img, mask=contour_img)
    palm_center = tuple([x+max_loc[0], y+max_loc[1]])
    palm_radius = int(max_val)

    return palm_center, palm_radius


def detectFingerCount(img):
    img_h, img_w, _ = img.shape
    # smoothing
    # img = cv2.medianBlur(img, 7)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # skin mask
    skin_mask = get_skin_mask(img)

    # show("skin_mask", skin_mask)

    # find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # keep only the top 3 largest contours
    contours = heapq.nlargest(3, contours, key=cv2.contourArea)
    
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    fingercount = -1
    for contour in contours:
        min_size = max(img_w/5, img_h/5)
        x, y, w, h = cv2.boundingRect(contour)

        #========================[checkpoint]=======================================
        # check if contour is too small
        if w<min_size or h<min_size:
            continue
        
        #========================[checkpoint]=======================================
        # check if center of contour is inside the contour
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cv2.pointPolygonTest(contour, (cx, cy), False) <= 0:
            continue
        
        # find palm
        palm_center, palm_radius = find_palm(contour, skin_mask.shape)

        #========================[checkpoint]=======================================
        # check if top of contour is too far away from palm center 
        if ((palm_center[1] - y) > palm_radius*4):
            continue

        # ------remove_below_wrist-----------
        wrist_y = palm_center[1] + int(palm_radius*1.6)
        res = np.empty([1,1,2], dtype=int)
        for p in contour:
            if p[0][1]<=wrist_y:
                res = np.insert(res, 0, p, 0)
        contour = res[:-1]

        # find the new bounding box
        x, y, w, h = cv2.boundingRect(contour)

        #========================[checkpoint]=======================================
        # check aspect ratio range
        aspect = float(w)/h
        if aspect > 1.5:
            continue

        # calculate some statistics
        rect_area = w*h
        hull = cv2.convexHull(contour)        
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(contour_area)/hull_area
        extent = float(contour_area)/rect_area
        palm_area = 3.14159*palm_radius*palm_radius
        palmbyhull = palm_area/hull_area

        #========================[checkpoint]=======================================
        # check if palm area is too small
        if palmbyhull < 0.18:
            continue

        # approcimate the contour
        epsilon = 0.01* cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        # cv2.putText(img, "%.3f"%(len(contour),), (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        #========================[checkpoint]=======================================
        if len(contour) >= 20:
            continue

        # extent and solidity will have higher values for a closed wrist
        # in that case we do not process anything further and return finger count as 0
        if extent > 0.7 or solidity > 0.85:
            if 0.8 < aspect < 1.1:
                fingercount = 0
            else:
                continue
        else:
            
            hullIndices = cv2.convexHull(contour, returnPoints=False)
            
            try:
                defects = cv2.convexityDefects(contour, hullIndices)
            except cv2.error as e:
                fingercount = -1
                continue

            if defects is None:
                fingercount = 0
            else:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    rsqr = (palm_radius*2/3)**2
                    if ((far[1] > start[1]) and (far[1] > end[1])):
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                        # check angle
                        if angle <= 90:
                            count_defects += 1
                            cv2.circle(img, far, 5, [0, 255, 0], -1)
                fingercount = count_defects + 1

        # draw on image for visualization
        cv2.drawContours(img, [hull], -1, (0,255,0), 2)
        cv2.drawContours(img, [contour], -1, (0,255,255), 2)
        
        
        # draw circle for testing
        cv2.circle(img, palm_center, palm_radius, (255, 255, 0), 1)
        cv2.circle(img, palm_center, 2, (255, 255, 0), -1)
        cv2.putText(img, "%d"%(fingercount,), palm_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, "%.3f"%(aspect,), (palm_center[0]+10, palm_center[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if fingercount != -1:
            break

    return fingercount, img


# a main function for testing purposes
if __name__ == "__main__":
    
    # get the camera video
    cap = cv2.VideoCapture(0)

    # camera loop
    while cap.isOpened():
        ret, frame = cap.read()
	
        fingercount, img = detectFingerCount(frame)

        cv2.imshow("img", img)

        # stop camera loop when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
