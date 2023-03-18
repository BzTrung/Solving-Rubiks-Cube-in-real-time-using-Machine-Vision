import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from rubik_solver import utils
from rotate import *

cube=[]
count = 0
mode = 0
check_steps = 0
cap = cv.VideoCapture(0)

def split_step(steps):
    for i,step in enumerate(steps):
        if step == "L2":
            list.remove(steps,"L2")
            list.insert(steps,i,'L')
            list.insert(steps,i,'L')
        elif step == "R2":
            list.remove(steps,"R2")
            list.insert(steps,i,'R')
            list.insert(steps,i,'R')
        elif step == "U2":
            list.remove(steps,"U2")
            list.insert(steps,i,'U')
            list.insert(steps,i,'U')
        elif step == "D2":
            list.remove(steps,"D2")
            list.insert(steps,i,'D')
            list.insert(steps,i,'D')
        # elif step == "B2":
        #     list.remove(steps,"B2")
        #     list.insert(steps,i,'B')
        #     list.insert(steps,i,'B')
        elif step == "F2":
            list.remove(steps,"F2")
            list.insert(steps,i,'F')
            list.insert(steps,i,'F')
    return steps

while(1):
    if mode == 0:
        list_box = []
        frame, plane,sort_list_box =test(cap)
        try:
            if check_steps == 0:
                cv.putText(frame,"Please, show yellow face",(10,30), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            if plane[1][1] == "y":
                check_steps = 1
            if check_steps == 1:
                if plane[1][1] == "y":
                    cv.arrowedLine(frame, sort_list_box[0][:], sort_list_box[2][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[3][:], sort_list_box[5][:], (255, 0, 255), 5, tipLength = 0.2)
                    cv.arrowedLine(frame, sort_list_box[6][:], sort_list_box[8][:], (255, 0, 255), 5, tipLength = 0.2)
                
                if plane[1][1] == "g":
                    turn_rubik(cap,"down","g","r")
                if plane[1][1] == "r":
                    turn_rubik(cap,"down","r","b")
                if plane[1][1] == "o":
                    turn_rubik(cap,"up","o","b")
                if plane[1][1] == "b":
                    turn_rubik(cap,"right","b","y")
                    mode = 1

        except: pass
      
        cv.imshow('a',frame)

    if mode == 1:
        list_box = []
        frame, plane,sort_list_box =test(cap)
        if plane is not None:
            count = 0
            try:
                if  plane[1][1]== "y":  # up 
                    up_face=plane.reshape((1,9))
                    turn_rubik(cap,"up","y","r")
                    print("Up plane:", plane)
                elif  plane[1][1]== "b":
                    left_face=plane.reshape((1,9)) # left
                    turn_rubik(cap,"left","b","o")
                    print("Left plane:", plane)
                elif  plane[1][1]== "r":
                    front_face=plane.reshape((1,9)) # front
                    turn_rubik(cap,"left","r","b")
                    print("Front plane:", plane)
                elif plane[1][1]== "g": # right
                    right_face=plane.reshape((1,9))
                    turn_rubik(cap,"left","g","r")
                    turn_rubik(cap,"up","r","w")
                    print("Right plane:", plane)
                elif plane[1][1]== "o": # back
                    back_face=plane.reshape((1,9))
                    turn_rubik(cap,"left","o","g")
                    print("Back plane:", plane)
                elif plane[1][1]== "w": # down
                    down_face=plane.reshape((1,9))
                    turn_rubik(cap,"down","w","r")
                    print("Down plane:", plane)
                flatten_arr = np.concatenate((up_face,left_face,front_face,right_face,back_face,down_face)).ravel()
                cube=''.join(map(str, flatten_arr))
                
            except: pass
        else: count +=1
        if len(cube)==54:
            print(cube)
            if cube == "yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww":
                mode = 3
            else:
                try:
                    steps = utils.solve(cube, 'Kociemba')
                    mode = 2
                except:
                    pass
        cv.imshow('a',frame)
        
    if mode == 2:
        print(steps)
        split_step(steps)
        print("split step: ",steps)
        for i,step in enumerate(steps):
            if step == "R":
                up_face,right_face,front_face,down_face,left_face,back_face = right_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "L":
                up_face,right_face,front_face,down_face,left_face,back_face = left_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "U":
                up_face,right_face,front_face,down_face,left_face,back_face = up_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "D":
                up_face,right_face,front_face,down_face,left_face,back_face = down_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "F":
                up_face,right_face,front_face,down_face,left_face,back_face = front_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "B":
                up_face,right_face,front_face,down_face,left_face,back_face = back_cw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "B2":
                up_face,right_face,front_face,down_face,left_face,back_face = back_cw_2( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "R'":
                up_face,right_face,front_face,down_face,left_face,back_face = right_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "L'":
                up_face,right_face,front_face,down_face,left_face,back_face = left_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "U'":
                up_face,right_face,front_face,down_face,left_face,back_face = up_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "D'":
                up_face,right_face,front_face,down_face,left_face,back_face = down_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "F'":
                up_face,right_face,front_face,down_face,left_face,back_face = front_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
            if step == "B'":
                up_face,right_face,front_face,down_face,left_face,back_face = back_ccw( cap,up_face,right_face,front_face,down_face,left_face,back_face)
        mode = 3
    if mode == 3:
        ret,frame = cap.read()
        cv.putText(frame,"DONE",(10,30), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv.imshow('a',frame)
    if cv.waitKey(1) == ord('q'):
        break 
  
cap.release()  
#destroys all window
cv.destroyAllWindows()