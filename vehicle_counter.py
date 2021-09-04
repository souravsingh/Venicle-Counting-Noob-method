import numpy as np

import cv2

# Function to calculate the centre of the bounding box
def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1= int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy


# Load a video here
vid = cv2.VideoCapture('highway.mp4')

# Variables for lines and bounding boxes
count_line_pos = 550
min_width= 80
min_height= 80


# add a background subtractor
sub = cv2.bgsegm.createBackgroundSubtractorGSOC()


detect = []
offset = 6
counter = 0


while vid.isOpened():

    # read video
    ret, image = vid.read()

    # convert to grayscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add Gaussian Blur
    blur = cv2.GaussianBlur(grey_img, (3, 3), 5)

    img_sub = sub.apply(blur)

    # add dilation to the background subtracted image

    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    # Conduct morphological transforms

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # find contours
    
    contourshape = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


    # Draw a line for counting
    cv2.line(image, (20, count_line_pos), (1000, count_line_pos), (255, 127, 0), 3)


    # Draw bounding boxes around contours
    for i,c in enumerate(contourshape):
        (x, y, w, h) = cv2.boundingRect(c)
        
        validate_counter = (w>=min_width) and(h>=min_height)
        if not validate_counter:
            continue
        
        cv2.rectangle(image, (x, y), (x+h, y+w), (0, 0, 255), 2)
        
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(image, center, 4, (0, 0, 255), -1)


        # Count the vehicles here
        for (x,y) in detect:
            if x<(count_line_pos+offset) and y<(count_line_pos+offset):
                counter +=1
            cv2.line(image, (15, count_line_pos), (1200, count_line_pos), (255, 127, 0), 3)
            detect.remove((x, y))
            print("Car count:"+ str(counter))

    #Show output here
    cv2.putText(image, "Vehicle count:" + str(counter), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
    
    cv2.imshow('', image)
    
    if cv2.waitKey(1)==13:
      break

cv2.destroyAllWindows()
vid.release()
