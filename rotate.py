import cv2 as cv
import numpy as np
colors = {"o": [3,190,162,20,255,255], "b": [110,162,36,130,255,255],"r": [0,213,95,41,255,140], "y": [30,162,45,59,255,255], "g": [64,45,73,87,255,255],"w": [0,0,81,179,53,255]}
color_box = {"o": (0,127,255), "b":(255,0,0) ,"r":(0,0,255) , "y":(0,255,255) ,"g":(0,255,0),"w":(255,0,255)}             
mode = 0
list_box = []
row0=[]
row1=[]
row2=[]


def remove_small_contours(image): # loại bỏ countour nhỏ nhất
    try:
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        mask = cv.drawContours(image_binary, [max(contours, key=cv.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv.bitwise_and(image, image, mask=mask)
        return image_remove
    except:
        return image
    
def find_rubik(image): # detect cục rubik
    img=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    b_img_rubik = cv.GaussianBlur(img, (5,5), 0)
    b_img_rubik =  cv.inRange(b_img_rubik,(0,0,0),(255,152,74))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    b_img_rubik = cv.morphologyEx(b_img_rubik, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_rubik = remove_small_contours(b_img_rubik)
    try:
        contours, hierachy = cv.findContours(b_img_rubik,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            Area = cv.contourArea(cnt)
            x,y,w,h = cv.boundingRect(cnt)
            # cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)  
            # cv.putText(image, str(int(Area)),(x,y), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        return b_img_rubik,x,y,w,h
    except:
        return b_img_rubik,1,1,1,1

def find_color(rubik_shape, color_str):
    # global list_box
    img_pro = cv.GaussianBlur(rubik_shape.copy(), (5,5), 0)
    hsv = cv.cvtColor(img_pro, cv.COLOR_BGR2HSV)
    color = colors[str(color_str)]
    color_box_show = color_box[str(color_str)]
    b_img = cv.inRange(hsv, (color[0],color[1],color[2]), (color[3],color[4],color[5]))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    b_img = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations=2)
    b_img = cv.dilate(b_img,kernel,iterations = 2)
    d_img = cv.distanceTransform(b_img, cv.DIST_L1,3)
    cv.normalize(d_img,d_img,0, 1.0, cv.NORM_MINMAX)
    d_img_show = d_img.copy()
    d_img = cv.threshold(d_img, 0.1, 1.0, cv.THRESH_BINARY)[1]
    d_img = d_img*255
    d_img = d_img.astype(np.uint8)
    #find contours
    contours, hierachy = cv.findContours(d_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        Area = cv.contourArea(cnt)
        if Area > 2000 and Area<7000:
            x,y,w,h = cv.boundingRect(cnt)
            if w/h >= 1.5 or w/h <= 0.5:
                pass
            else:
                cv.rectangle(rubik_shape,(x,y),(x+w,y+h),color_box_show,2)  
                cv.circle(rubik_shape, (x+int(w/2),y+int(h/2)),3,color_box_show,3)
                # cv.putText(rubik_shape, str(Area),(x,y), cv.FONT_HERSHEY_SIMPLEX,0.5,color_box_show,1)
                list_box.append([(x+int(w/2)),(y+int(h/2)),color_str])
    
    return b_img, d_img, d_img_show,list_box
def test(cap):
    list_box = []
    sort_list_box = []
    plane = np.zeros((3,3), dtype=str)
    ret,frame = cap.read()   
    b_img,x,y,w,h = find_rubik(frame)
    rubik_shape = frame[y:y+h, x:x+w]
    for i in colors:
        b_img_color, d_img, d_img_show,list_box = find_color(rubik_shape,str(i))
    row0=[]
    row1=[]
    row2=[]
    hang = []
    if len(list_box) > 9: list_box.clear()
    if (len(list_box)==9):
        try:
            for box in list_box:
                if box[1] >= 0 and box[1] <= h/3:
                    row0.append(box)
                if box[1] >= h/3 and box[1] <= 2*h/3:
                    row1.append(box)
                if box[1] >= 2*h/3 and box[1] <= h:
                    row2.append(box)
            hang = [row0,row1,row2]
            hang = np.asarray(hang)
            for j,row  in enumerate(hang):
                min = np.argmin(np.array(row[:,0], dtype=int))
                max = np.argmax(np.array(row[:,0], dtype=int))
                plane[j,0] = row[min,2]
                plane[j,2] = row[max,2]
                plane[j,1] = row[3-(min+max),2]
                
                sort_list_box.append([(int(row[min,0])+x),(int(row[min,1])+y)])
                sort_list_box.append([(int(row[3-(min+max),0])+x),(int(row[3-(min+max),1])+y)])
                sort_list_box.append([(int(row[max,0])+x),(int(row[max,1])+y)])
                
        except:
            pass
    return frame,plane,sort_list_box


def rotate_cw(face):
    final = np.copy(face)
    final[0, 0] = face[0, 6]
    final[0, 1] = face[0, 3]
    final[0, 2] = face[0, 0]
    final[0, 3] = face[0, 7]
    final[0, 4] = face[0, 4]
    final[0, 5] = face[0, 1]
    final[0, 6] = face[0, 8]
    final[0, 7] = face[0, 5]
    final[0, 8] = face[0, 2]
    return final

def rotate_ccw(face):
    final = np.copy(face)
    final[0, 8] = face[0, 6]
    final[0, 7] = face[0, 3]
    final[0, 6] = face[0, 0]
    final[0, 5] = face[0, 7]
    final[0, 4] = face[0, 4]
    final[0, 3] = face[0, 1]
    final[0, 2] = face[0, 8]
    final[0, 1] = face[0, 5]
    final[0, 0] = face[0, 2]
    return final

def right_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 2] = down_face[0, 2]
    front_face[0, 5] = down_face[0, 5]
    front_face[0, 8] = down_face[0, 8]
    down_face[0, 2] = back_face[0, 6]
    down_face[0, 5] = back_face[0, 3]
    down_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = up_face[0, 8]
    back_face[0, 3] = up_face[0, 5]
    back_face[0, 6] = up_face[0, 2]
    up_face[0, 2] = temp[0, 2]
    up_face[0, 5] = temp[0, 5]
    up_face[0, 8] = temp[0, 8]
    right_face = rotate_cw(right_face)
    #front_face = temp


    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4] == temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[8][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break   
      
def right_ccw(cap ,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 2] = up_face[0, 2]
    front_face[0, 5] = up_face[0, 5]
    front_face[0, 8] = up_face[0, 8]
    up_face[0, 2] = back_face[0, 6]
    up_face[0, 5] = back_face[0, 3]
    up_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = down_face[0, 8]
    back_face[0, 3] = down_face[0, 5]
    back_face[0, 6] = down_face[0, 2]
    down_face[0, 2] = temp[0, 2]
    down_face[0, 5] = temp[0, 5]
    down_face[0, 8] = temp[0, 8]
    right_face = rotate_ccw(right_face)
    
    faces = []
    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[2][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    
    # cap.release()  
        #destroys all window
    # cv.destroyAllWindows()


def left_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 0] = up_face[0, 0]
    front_face[0, 3] = up_face[0, 3]
    front_face[0, 6] = up_face[0, 6]
    up_face[0, 0] = back_face[0, 8]
    up_face[0, 3] = back_face[0, 5]
    up_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = down_face[0, 6]
    back_face[0, 5] = down_face[0, 3]
    back_face[0, 8] = down_face[0, 0]
    down_face[0, 0] = temp[0, 0]
    down_face[0, 3] = temp[0, 3]
    down_face[0, 6] = temp[0, 6]
    left_face = rotate_cw(left_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[6][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    
    # cap.release()  
        #destroys all window
    # cv.destroyAllWindows()
        
def left_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 0] = down_face[0, 0]
    front_face[0, 3] = down_face[0, 3]
    front_face[0, 6] = down_face[0, 6]
    down_face[0, 0] = back_face[0, 8]
    down_face[0, 3] = back_face[0, 5]
    down_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = up_face[0, 6]
    back_face[0, 5] = up_face[0, 3]
    back_face[0, 8] = up_face[0, 0]
    up_face[0, 0] = temp[0, 0]
    up_face[0, 3] = temp[0, 3]
    up_face[0, 6] = temp[0, 6]
    left_face = rotate_ccw(left_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[0][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break     
    # cap.release()  
        #destroys all window
    # cv.destroyAllWindows()

def front_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp1 = np.copy(front_face)
    temp = np.copy(up_face)
    front_face = rotate_cw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2, temp1) == True:
        up_face, right_face, front_face, down_face, left_face, back_face = turn_to_right(cap,  up_face, right_face, front_face, down_face, left_face, back_face)
        up_face, right_face, front_face, down_face, left_face, back_face = left_cw(cap,  up_face, right_face, front_face, down_face, left_face, back_face)
        up_face, right_face, front_face, down_face, left_face, back_face = turn_to_left(cap,  up_face, right_face, front_face, down_face, left_face, back_face)
        return up_face, right_face, front_face, down_face, left_face, back_face
    up_face[0, 8] = left_face[0, 2]
    up_face[0, 7] = left_face[0, 5]
    up_face[0, 6] = left_face[0, 8]
    left_face[0, 2] = down_face[0, 0]
    left_face[0, 5] = down_face[0, 1]
    left_face[0, 8] = down_face[0, 2]
    down_face[0, 2] = right_face[0, 0]
    down_face[0, 1] = right_face[0, 3]
    down_face[0, 0] = right_face[0, 6]
    right_face[0, 0] = temp[0, 6]
    right_face[0, 3] = temp[0, 7]
    right_face[0, 6] = temp[0, 8]

    while True:

        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp1[0,4]:
                cv.arrowedLine(frame, sort_list_box[1][:], sort_list_box[5][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[5][:], sort_list_box[7][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[7][:], sort_list_box[3][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[3][:], sort_list_box[1][:], (255, 0, 255), 5, tipLength = 0.2)

        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break   

def front_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(up_face)
    temp1 = np.copy(front_face)
    front_face = rotate_ccw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2,temp1) == True:
            up_face, right_face, front_face, down_face, left_face, back_face = turn_to_right(cap,up_face,right_face,front_face,down_face,left_face,back_face)
            up_face, right_face, front_face, down_face, left_face, back_face = left_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face)
            up_face, right_face, front_face, down_face, left_face, back_face = turn_to_left(cap,up_face,right_face,front_face,down_face,left_face,back_face)
            return up_face,right_face,front_face,down_face,left_face,back_face
    up_face[0, 6] = right_face[0, 0]
    up_face[0, 7] = right_face[0, 3]
    up_face[0, 8] = right_face[0, 6]
    right_face[0, 0] = down_face[0, 2]
    right_face[0, 3] = down_face[0, 1]
    right_face[0, 6] = down_face[0, 0]
    down_face[0, 0] = left_face[0, 2]
    down_face[0, 1] = left_face[0, 5]
    down_face[0, 2] = left_face[0, 8]
    left_face[0, 8] = temp[0, 6]
    left_face[0, 5] = temp[0, 7]
    left_face[0, 2] = temp[0, 8]
    
    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp1[0,4]:
                cv.arrowedLine(frame, sort_list_box[5][:], sort_list_box[1][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[7][:], sort_list_box[5][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[3][:], sort_list_box[7][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[1][:], sort_list_box[3][:], (255, 0, 255), 5, tipLength = 0.2)

        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break 

def back_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):

    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_right(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = right_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_left(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    return up_face,right_face,front_face,down_face,left_face,back_face

def back_cw_2(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_right(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = right_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = right_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_left(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    return up_face,right_face,front_face,down_face,left_face,back_face

def back_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face):

    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_right(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = right_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    up_face, right_face, front_face, down_face, left_face, back_face = turn_to_left(cap,up_face,right_face,front_face,down_face,left_face,back_face)
    return up_face,right_face,front_face,down_face,left_face,back_face


def up_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 0] = right_face[0, 0]
    front_face[0, 1] = right_face[0, 1]
    front_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = back_face[0, 0]
    right_face[0, 1] = back_face[0, 1]
    right_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = left_face[0, 0]
    back_face[0, 1] = left_face[0, 1]
    back_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = temp[0, 0]
    left_face[0, 1] = temp[0, 1]
    left_face[0, 2] = temp[0, 2]
    up_face = rotate_cw(up_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[2][:], sort_list_box[0][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    

def up_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 0] = left_face[0, 0]
    front_face[0, 1] = left_face[0, 1]
    front_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = back_face[0, 0]
    left_face[0, 1] = back_face[0, 1]
    left_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = right_face[0, 0]
    back_face[0, 1] = right_face[0, 1]
    back_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = temp[0, 0]
    right_face[0, 1] = temp[0, 1]
    right_face[0, 2] = temp[0, 2]
    up_face = rotate_ccw(up_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    

def down_cw(cap,up_face,right_face,front_face,down_face,left_face,back_face):

    temp = np.copy(front_face)
    front_face[0, 6] = left_face[0, 6]
    front_face[0, 7] = left_face[0, 7]
    front_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = back_face[0, 6]
    left_face[0, 7] = back_face[0, 7]
    left_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = right_face[0, 6]
    back_face[0, 7] = right_face[0, 7]
    back_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = temp[0, 6]
    right_face[0, 7] = temp[0, 7]
    right_face[0, 8] = temp[0, 8]
    down_face = rotate_cw(down_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    

def down_ccw(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face[0, 6] = right_face[0, 6]
    front_face[0, 7] = right_face[0, 7]
    front_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = back_face[0, 6]
    right_face[0, 7] = back_face[0, 7]
    right_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = left_face[0, 6]
    back_face[0, 7] = left_face[0, 7]
    back_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = temp[0, 6]
    left_face[0, 7] = temp[0, 7]
    left_face[0, 8] = temp[0, 8]
    down_face = rotate_ccw(down_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[8][:], sort_list_box[6][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    

def turn_to_right(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face = np.copy(right_face)
    right_face = np.copy(back_face)
    back_face = np.copy(left_face)
    left_face = np.copy(temp)
    up_face = rotate_cw(up_face)
    down_face = rotate_ccw(down_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[2][:], sort_list_box[0][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[5][:], sort_list_box[3][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[8][:], sort_list_box[6][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break    

def turn_to_left(cap,up_face,right_face,front_face,down_face,left_face,back_face):
    temp = np.copy(front_face)
    front_face = np.copy(left_face)
    left_face = np.copy(back_face)
    back_face = np.copy(right_face)
    right_face = np.copy(temp)
    up_face = rotate_ccw(up_face)
    down_face = rotate_cw(down_face)

    while True:
        frame,detect_plane,sort_list_box=test(cap)
        up_face = np.asarray(up_face)
        front_face = np.asarray(front_face)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if np.array_equal(detect_plane, front_face) == True:
                
                return up_face,right_face,front_face,down_face,left_face,back_face
            
            elif detect_plane[0,4]== temp[0,4]:
                cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[3][:], sort_list_box[5][:], (255, 0, 255), 5, tipLength = 0.2)
                cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_rubik(cap,direction,now_center,next_center):
    while True:
        frame,detect_plane,sort_list_box=test(cap)
        detect_plane = np.asarray(detect_plane.reshape((1,9)))
        if len(sort_list_box)==9:
            if detect_plane[0,4] == next_center:
                return True
            
            elif detect_plane[0,4]== now_center:
                if direction == "up":
                    cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[0][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[7][:], sort_list_box[1][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[8][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
                if direction == "down":
                    cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[6][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[1][:], sort_list_box[7][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[2][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
                if direction == "left":
                    cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[3][:], sort_list_box[5][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
                if direction == "right":
                    cv.arrowedLine(frame, sort_list_box[2][:], sort_list_box[0][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[5][:], sort_list_box[3][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[8][:], sort_list_box[6][:], (255, 0, 255), 5, tipLength = 0.2)
        cv.imshow('a',frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break