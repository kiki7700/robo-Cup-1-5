#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import cv2
import os
import sys
import glob
import numpy as np
import math
import socket
import argparse
import time

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from smartcar.msg import light

intrinsicMat = []
distortionCoe = []
perspective_transform_matrix = []

arguments = [
    [6, 0.28, -0.24, -0.43, -0.151, -0.4, 0.56, 0.25],
    [4, 0.28, 0.21, -0.521, 0.3, -0.508, 0.37, 0.42],
    [8, 0.2, 0, -0.8, 0.3, -0.6, 0.2, 0.5],
    [6, 0.325, 0.23, -0.68, 0.3, -0.6, 0.5, 0.4],
    [2, 0.29, 0, -0.56, 0.3, -0.54, 0.5, 0.4]
]

num_dis_left = 0
num_dis_right = 0
state = 0
times = 0
old_z = 0

# 数据传输
address1=('192.168.43.120',22)  #电脑服务端的主机号和端口号
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
try:
    print('waiting connecting......')
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock1.connect(address1)
    print('connected successful......')
except socket.error as msg:
    print(msg)
    sys.exit(1)

def sendVideo(img, sock):
    result, imgencode = cv2.imencode('.jpg', img, encode_param)  #cv2.imencode() 函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输.
    data = np.array(imgencode)  #创建一个数组
    stringData = data.tobytes()  #tobytes() 方法可以将数组转换为一个机器值数组并返回其字节表示。
    sock.send(str.encode(str(len(stringData)).ljust(16)))  #返回字符串编码后的数据  ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串。如果指定的长度小于原字符串的长度则返回原字符串。
    sock.send(stringData)

# not detect the green light
starter = True
# ignore the lidar
global lidarLaunch
lidarLaunch = False

def initial_parameters():
    global intrinsicMat
    global distortionCoe
    global perspective_transform_matrix
    global perspective_transform_matrix_slope
    global pp_control

    ##ramp_control = True
    intrinsicMat = np.array([[428.264533, -0.2570289, 295.1081],
                             [0, 427.8575, 253.5551],
                             [0, 0, 1]])

    distortionCoe = np.array([-0.38682604, 0.13534661, 8.18363975e-5, -2.8665336872e-4, 0])
    srcps = np.float32([[(180, 132), (1, 285), (639, 285), (461, 130)]])  #srcps为摄像头中拍摄到的标定矩形的图中四角坐标
    dstps = np.float32([[(180,0), (180,480), (460,480), (460,0 )]])      #dstps为设想中转换后对应的四角坐标
    perspective_transform_matrix = cv2.getPerspectiveTransform(srcps, dstps)  #perspective_transform_matrix为利用以上两者计算获得的转换矩阵


def perspectiveTrans(img):
    global perspective_transform_matrix
    if perspective_transform_matrix == []:
        print("Transform failed!")
        return img
    else:
        bird_view_img = cv2.warpPerspective(img, perspective_transform_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)  #图片视角转换  透视变换
        return bird_view_img


def light_detection(img):
    img = img[:400, 430:640]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv1 = hsv.copy()
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_lower = np.array([0, 50, 200])
    red_upper = np.array([5, 255, 255])  # [5,255,255]
    red_mask = cv2.inRange(hsv, red_lower, red_upper)  #cv2.inRange函数设阈值，去除背景部分
    red_target = cv2.bitwise_and(hsv, hsv, mask=red_mask)
    red_target = cv2.erode(red_target, element)  #图像腐蚀
    red_target = cv2.dilate(red_target, element)  #图像膨胀
    red_gray = cv2.cvtColor(red_target, cv2.COLOR_BGR2GRAY)
    r_ret, r_binary = cv2.threshold(red_mask, 127, 255, cv2.THRESH_BINARY)  #图像二值化（阈值）
    r_gray2 = cv2.Canny(r_binary, 100, 200) #边缘检测
    r = r_gray2[:, :] == 255
    count_red = len(r_gray2[r])
    redLight=0
    if count_red > 560:
        red_n = red_n + 1
        print("red_n:"+str(red_n))
        if red_n>5:
            redLight=1
    else:
        redLight = 0
    print("redLight:"+str(redLight))
    green_lower = np.array([80, 95, 230])
    green_upper = np.array([130, 255, 255])
    green_mask = cv2.inRange(hsv1, green_lower, green_upper)
    green_target = cv2.bitwise_and(hsv1, hsv1, mask=green_mask)
    green_target = cv2.erode(green_target, element)
    green_target = cv2.dilate(green_target, element)
    green_gray = cv2.cvtColor(green_target, cv2.COLOR_BGR2GRAY)
    g_ret, g_binary = cv2.threshold(green_mask, 127, 255, cv2.THRESH_BINARY)
    g_gray2 = cv2.Canny(g_binary, 100, 200)
    g = g_gray2[:, :] == 255
    count_green = len(g_gray2[g])
    # print(count_green)
    if count_green > 150:
        greenLight = 1
    else:
        greenLight = 0
    print("greenLight:"+str(greenLight))
    print('red_count: '+ str(count_red))
    print('green_count: ' + str(count_green))
    
    #是否有红绿灯亮
    if greenLight + redLight > 0:
        hasLight = 1
        if redLight == 1:
            Light = 2
        elif greenLight == 1:
            Light = 1
    else:
        hasLight = 0
        Light = 0
    return Light


class command:
    def __init__(self):
        count = 1
        # 摄像头端口号
        self.cap = cv2.VideoCapture(0)  #读取视频或摄像头
        self.pubI = rospy.Publisher('images', Image, queue_size=1)
        self.puborignialI = rospy.Publisher('orignial_images', Image, queue_size=1)
        self.pubB = rospy.Publisher('bird_images', Image, queue_size=1)
        rospy.init_node('command_core', anonymous=True)
        rospy.Subscriber("light_msg", light, RLcallback)
        rospy.Subscriber("laser_cmd", Twist, LScallback)
        self.rate = rospy.Rate(20)
        self.cvb = CvBridge()  #CvBridge是一个ROS库，提供ROS和OpenCV之间的接口。将ROS的图像消息转换为OpenCV图像格式
        
    def spin(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        global state
        global times
        times = 0
        state_0to1 = 95
        state_1to2 = 185 # 240
        state_2to3 = 450
        state_end = 530
        
        global lidarLaunch
        global starter, pub, greenLight, aP, lastP
        global aP_kf
        global last_lane_base, last_LorR
        global final_cmd, cam_cmd , stop_judge_local  #stop_judge_local来判断智能车的启停，true时停止，false时继续前进。
        global roadWidth
        global judge_end_tunnel
        global old_z
        last_lane_base = -1
        last_LorR = 1
        y_nearest = 0
        is_debug = False

        bzw_flag = 0
        old_flag_state = 0    # 上一次标志物的状态
        delay5_isflag_ = 0    # 连续5次检测到相同标志物则判定为该标志物
        in_flags_times = 0    # 进入标志物判断时的时间


        while not rospy.is_shutdown():
            ret, img = self.cap.read()  #参数ret 为True 或者False,代表有没有读取到图片.第二个参数frame表示截取到一帧的图片
            if ret == True:
                if(not lidarLaunch):
                    print("\n\n-----------(times, state) : (%d, %d)-----------" %(times, state))
                    times += 1
                    undstrt = cv2.undistort(img, intrinsicMat, distortionCoe, None, intrinsicMat)  #矫正图像
                    img_socket = cv2.resize(undstrt,(undstrt.shape[1] / 3, undstrt.shape[0] / 3))
                    img_encode1 = cv2.imencode('.jpg', img_socket)[1]
                    data_encode1 = np.array(img_encode1)
                    data1 = data_encode1.tostring()
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(undstrt, cv2.COLOR_BGR2GRAY)

                    if Light is 2:
                        print("......red......")
                        cmd_vel.linear.x = 0
                        final_cmd = cam_cmd
                        pub.publish(final_cmd)
                    else:  #寻线操作
                        print(".........Track.......")
                        if pp_control == True:
                            cam_cmd.angular.x = 1

                        gray_Blur = cv2.GaussianBlur(gray, (5, 5), 0)  #高斯模糊
                        origin_thr = cv2.Canny(gray_Blur, 100, 200)    #边缘检测
                        imgColor = cv2.inRange(undstrt, np.array([200, 200, 200]), np.array([255, 255, 255]))  #设阈值，去除背景部分
                        combinedimg = cv2.bitwise_or(imgColor, origin_thr)
                        origin_thr = cv2.dilate(combinedimg, np.ones((3, 3), np.uint8))  #膨胀函数，用于二值化图像
                        origin_thr = cv2.erode(origin_thr, np.ones((7, 7), np.uint8))  #图像腐蚀
                        binary_warped = perspectiveTrans(origin_thr)  #得到鸟瞰图
                        binary_warp = cv2.erode(binary_warped, np.ones((11, 11), np.uint8)) #图像腐蚀
                        binary_warp = cv2.erode(binary_warped, np.ones((3, 3), np.uint8)) #图像腐蚀
                        binary_warped_socket=binary_warp[320:,:]
                        binary_warped_socket = cv2.resize(binary_warped_socket, (binary_warped_socket.shape[1] / 3, binary_warped_socket.shape[0] / 3))
                        img_encode = cv2.imencode('.jpg', binary_warped_socket)[1] #将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
                        data_encode = np.array(img_encode)
                        data = data_encode.tostring()
                        #s.sendto(data, ('192.168.1.103', 9999))
                        binary_caculate = binary_warped
                        binary_right_top = binary_warped[0:160,320:640]
                        binary_left = binary_warped[320:, 0:320]
                        binary_right = binary_warped[320:, 320:640]
                        warped = perspectiveTrans(gray)  #得到鸟瞰图

                        lines_l = cv2.HoughLinesP(binary_left, 1, np.pi / 180, 50, 50, 0)  # 二值图像中查找直线段
                        lines_r = cv2.HoughLinesP(binary_right, 1, np.pi / 180, 50, 50, 0)
                        lines_rt = cv2.HoughLinesP(binary_right_top, 1, np.pi / 180, 50, 50, 0)
                        theta_l = 0
                        theta_r = 0
                        steerAngle = 0
                        if lines_l is None:
                            pass
                        else:
                            for x0, y0, x1, y1 in lines_l[0]:
                                cv2.line(binary_warped, (x0, y0), (x1, y1), (0, 0, 255),
                                         5)  # 画线  binary_warped：背景图 (x0, y0)：直线起始坐标 (x1, y1)：直线终点坐标 (0, 0, 255)：RGB值  5:5个像素宽度
                                y0 = 160 - y0  # 坐标轴转换
                                y1 = 160 - y1
                                if x0 == x1:
                                    theta_l = 1.57
                                else:
                                    theta_l = math.atan((y0 - y1) / (x0 - x1))  # 返回x的反正切
                        if lines_r is None:
                            pass
                        else:
                            for x0, y0, x1, y1 in lines_r[0]:
                                cv2.line(binary_warped, (x0 + 320, y0), (x1 + 320, y1), (0, 0, 255), 5)
                                y0 = 160 - y0
                                y1 = 160 - y1
                                if x0 == x1:
                                    theta_r = 1.57
                                else:
                                    theta_r = math.atan((y0 - y1) / (x0 - x1))

                        print("theta1 : ", theta_l, " theta2 : ", theta_r)
                        is_bend = if_bend(theta_l, theta_r)
                        cam_cmd.linear.x = speed_x

                        if(is_bend==0):  #直道




#是否是弯道
def if_bend(theta1 , theta2):
    if abs(abs(theta1) - abs(theta2)) * 57 > 8 or theta2==0 or theta1 ==0 or abs(theta1 - theta2) < 0.5:
        return 1
    else:
        return 0

#直道转角计算
def steerAngle_calculate_straight(theta1,theta2,img):
    global num_dis_left
    global num_dis_right
    num_dis_left = 0
    num_dis_right = 0
    straight_k = arguments[state][7]
    
    img_l = img[400:480, :320]
    img_r = img[400:480, 320:]
    area_l = np.where(img_l[:] != 0)
    area_r = np.where(img_r[:] != 0)
    steerAngle = (3.14 - (abs(theta1)+abs(theta2)))
    if abs(abs(theta1) - abs(theta2)) * 57 < 3:
        print("......straight......")
        return 0
    else:
        if len(area_l[0]) > len(area_r[0]):
            steerAngle = steerAngle * 0.55
            print("-----straight_right", steerAngle)
        else:
            steerAngle = -(steerAngle * straight_k)
            print("-----straight_left", steerAngle)
        return steerAngle

#弯道斜率转角计算
def steerAngle_calculate_bend(theta1, theta2,theta_rt):
    global state
    global num_dis_left
    global num_dis_right
    print('num_dis_left = ', num_dis_left, ' num_dis_right = ', num_dis_right)
    delay_timess = arguments[state][0]
    angle_letf_on_delay = arguments[state][2]
    angle_left_after_delay = arguments[state][3]
    angle_right_on_delay = arguments[state][4]
    angle_left = arguments[state][5]
    angle_right = arguments[state][6]

    thetamax = theta2 if abs(theta2)>abs(theta1) else theta1
    if thetamax > 0:
        print("......bend_right......")
        # 状态 0 的丢线处理
        if state == 0 and (abs(theta1 - theta2) < 0.5 or theta1 == 0 or theta2 == 0) and num_dis_right < delay_timess:
            print("right line has disapear,start delay...")
            num_dis_right += 1
            steerAngle = angle_right_on_delay
        # 状态 3 的丟线处理
        elif state == 3:
            if theta1 == 0 and theta2 == 0:
                steerAngle = 0
            elif theta2 == 0:
                steerAngle = 0.15
            else:
                steerAngle = -0.15
        else:
            steerAngle = angle_right
    else:
        print("......bend_left......")
        # 状态0 左转不判定丢线
        if theta1 is 0 or theta2 is 0:
            if state == 3:
                if theta1 == 0 and theta2 == 0:
                    steerAngle = 0
                elif theta2 == 0:
                    steerAngle = 0.15
                else:
                    steerAngle = -0.15
            if num_dis_left < delay_timess:
                print("left line has disapear,start delay...")
                num_dis_left += 1
                steerAngle = angle_letf_on_delay
            elif num_dis_left >= delay_timess:
                steerAngle = angle_left_after_delay
        elif state == 4 and abs(theta1 - theta2) < 0.5:
            if num_dis_left < delay_times:
                print("left line has disapear,start delay...")
                num_dis_left += 1
                steerAngle = angle_letf_on_delay
            elif num_dis_left >= delay_times:
                steerAngle = angle_left_after_delay
        else:
            steerAngle = angle_left
    return steerAngle

def RLcallback(light_msg):
    global stop_judge_local  #判断智能车的启停，true时停止，false时继续前进
    global pp_control
    if light_msg.show_times > 1 and light_msg.red_or_green == 1:  #green led
        pp_control = True
    if light_msg.red_or_green == 2:  #red led
        stop_judge_local = True
    else:
        stop_judge_local = False


def LScallback(laser_cmd):
    global stop_judge_local, lidarLaunch
    global final_cmd
    global judge_end_tunnel
    judge_end_tunnel =  laser_cmd.linear.y
    if stop_judge_local == True:
        print('...停止...')
        laser_cmd.linear.x = 0
        final_cmd = laser_cmd

    if laser_cmd.linear.z >= 0:  #laser control steer 
        print('...雷达...')
        lidarLaunch = True 
        final_cmd = laser_cmd
    
    # 避障
    if laser_cmd.linear.z == 0 and state == 3:  #laser control steer
        print('...避障...')
        lidarLaunch = True 
        final_cmd = laser_cmd
    else:
        lidarLaunch = False
        pass
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #argparse是一个Python模块：命令行选项、参数和子命令解析器。ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    parser.add_argument("--k", type=float, default=1.0)  #添加参数
    parser.add_argument("--thy", type=float, default=0.2)
    parser.add_argument("--b", type=float, default=0)
    parser.add_argument("--trq", type=float, default=0.325)
    initial_parameters()
    args = parser.parse_args()  #解析添加的参数
    global pp_control
    pp_control = False
    global judge_end_tunnel
    judge_end_tunnel = 0

    initial_parameters()
    # 距离映射
    x_cmPerPixel = 80 / 320.0
    y_cmPerPixel = 80 / 320.0
    roadWidth = 80.0 / x_cmPerPixel  # 80.0

    aP = [0.0, 0.0]
    lastP = [0.0, 0.0]
    aP_kf = [0.0, 0.0]
    timer = 0

    # 轴间距
    I = 14
    # 图像坐标系底部与车后轮轴心间距#cm
    D = 38.5
    # 计算cmdSteer的系数，舵机转向与之前相反，此处用正数
    k1 = 4  # 3.6
    k2 = 4
    # steerAngle, cmdSteer;
    global final_cmd, cam_cmd  # laser_cmd = Twist()
    final_cmd = Twist()

    cam_cmd = Twist()
    ##cam_cmd.linear.x = 0.2

    global stop_judge_local
    stop_judge_local = False

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    try:
        cmd = command()
        cmd.spin()
        rospy.spin()
        final_cmd = cam_cmd


    except rospy.ROSInterruptException:
        pass

