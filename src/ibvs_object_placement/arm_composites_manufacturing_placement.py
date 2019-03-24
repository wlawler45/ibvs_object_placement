# Copyright (c) 2018, Rensselaer Polytechnic Institute, Wason Technology LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Rensselaer Polytechnic Institute, nor Wason 
#       Technology LLC, nor the names of its contributors may be used to 
#       endorse or promote products derived from this software without 
#       specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import copy
import rospy
import actionlib
import rospkg
import general_robotics_toolbox as rox
import general_robotics_toolbox.urdf as urdf
import general_robotics_toolbox.ros_msg as rox_msg
from general_robotics_toolbox import ros_tf as tf

import rpi_abb_irc5.ros.rapid_commander as rapid_node_pkg
import safe_kinematic_controller.ros.commander as controller_commander_pkg
#from rpi_arm_composites_manufacturing_process.msg import ProcessState, ProcessStepFeedback
from object_recognition_msgs.msg import ObjectRecognitionAction, ObjectRecognitionGoal

from industrial_payload_manager.payload_transform_listener import PayloadTransformListener
from industrial_payload_manager.srv import UpdatePayloadPose, UpdatePayloadPoseRequest, \
    GetPayloadArray, GetPayloadArrayRequest
import time
import sys
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal, MoveItErrorCodes
import os

import threading
from moveit_commander import PlanningSceneInterface
import traceback
import resource_retriever
import urlparse
from urdf_parser_py.urdf import URDF
from tf.msg import tfMessage
from visualization_msgs.msg import Marker, MarkerArray
import subprocess

### 
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge, CvBridgeError
#from rpi_arm_composites_manufacturing_process.msg import ProcessStepAction, ProcessStepGoal, ProcessState
from ibvs_object_placement.msg import PlacementCommand, PlacementStepAction, PlacementStepGoal,PlacementStepResult,PlacementStepFeedback
import tf
from scipy.linalg import expm
import cv2
import cv2.aruco as aruco
import scipy.misc
from scipy.io import loadmat
from safe_kinematic_controller.msg import ControllerState as controllerstate
from moveit_msgs.msg import RobotTrajectory

from placement_functions import QP_abbirb6640,QP_TwoCam, QP_Cam, trapgen,sort_corners

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class CameraParams:
    def __init__(self):
        #self.camMatrix, self.distCoeff = self.CameraParams(M00,M02,M11,M12,M22,C00,C01,C02,C03,C04)
        self.camMatrix=None
        self.distCoeff=None

    def CameraParams(self, M00,M02,M11,M12,M22,C00,C01,C02,C03,C04):
        camMatrix = np.zeros((3, 3),dtype=np.float64)
        camMatrix[0][0] = M00
        camMatrix[0][2] = M02
        camMatrix[1][1] = M11
        camMatrix[1][2] = M12
        camMatrix[2][2] = M22

        distCoeff = np.zeros((1, 5), dtype=np.float64)
        distCoeff[0][0] = C00
        distCoeff[0][1] = C01
        distCoeff[0][2] = C02
        distCoeff[0][3] = C03
        distCoeff[0][4] = C04

        return camMatrix,distCoeff

class ObjectRecognitionCommander(object):
    def __init__(self):
        self.ros_image=None
        self.ros_image_stamp = rospy.Time(0)
        self.bridge = CvBridge()

    def ros_raw_gripper_2_image_cb(self, ros_image_data):
        self.ros_image = self.bridge.imgmsg_to_cv2(ros_image_data, desired_encoding="passthrough")
        self.ros_image_stamp= ros_image_data.header.stamp
        
        
class PlacementController(object):

    def __init__(self):

        #Subscribe to controller_state 
        self.controller_state_sub = rospy.Subscriber("controller_state", controllerstate, self.callback)
        self.last_ros_image_stamp = rospy.Time(0)
        self.goal_handle=None
        
               
        self.listener = PayloadTransformListener()
        self.rapid_node = rapid_node_pkg.RAPIDCommander()
        self.controller_commander = controller_commander_pkg.ControllerCommander()
        self.object_commander = ObjectRecognitionCommander() 
        # Initilialize aruco boards and parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters =  cv2.aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.adaptiveThreshWinSizeMax=30
        self.parameters.adaptiveThreshWinSizeStep=7
        # ================== Cam 636
        # --- Subscribe to Gripper camera node for image acquisition
        self.ros_gripper_2_img_sub = rospy.Subscriber('/gripper_camera_2/image', Image, self.object_commander.ros_raw_gripper_2_image_cb)
        self.ros_gripper_2_trigger = rospy.ServiceProxy('/gripper_camera_2/trigger', Trigger)
        
        # --- Camera parameters
        self.CamParam = CameraParams()
        # --- Camera pose
        #TODO: Substitute transform in here
        R_Jcam = np.array([[0.9995,-0.0187,-0.0263],[-0.0191,-0.9997,-0.0135],[-0.0261,0.0140,-0.9996]])
        r_cam = rox.hat(np.array([0.0707, 1.1395, 0.2747]))#rox.hat(np.array([- 0.2811, -1.1397,0.0335]))#rox.hat(np.array([- 0.2811, 1.1397,0.0335]))
        self.R_Jcam = np.linalg.inv(np.vstack([ np.hstack([R_Jcam,np.zeros([3,3])]), np.hstack([np.dot(r_cam,R_Jcam),R_Jcam]) ]))
        self.dt = None
        self.iteration=0
        
        self.board_ground = None
        # functions like a gain, used with velocity to track position
        self.FTdata = None
        self.ft_flag = False
        #self.FTdata_0 = self.FTdata
        #self.FTdata_0est = self.compute_ft_est()
        self.result = self.take_image()
        self.client = actionlib.SimpleActionClient("joint_trajectory_action", FollowJointTrajectoryAction)
        # IBVS parameters
        self.du_converge_TH = None
        self.dv_converge_TH = None
        self.iteration_limit = None
        self.Ki = None   
        # Compliance controller parameters
        self.F_d_set1 = None
        self.F_d_set2 = None
        self.Kc = None
        self.ros_data=None
        self.camMatrix=None
        self.distCoeffs=None
        self.ros_gripper_2_cam_info_sub=rospy.Subscriber('/gripper_camera_2/camera_info', CameraInfo, self.fill_camera_data)

    def fill_camera_data(self,ros_data):
        self.ros_data=ros_data
        camMatrix=np.reshape(ros_data.K,(3,3))
        distCoeffs=np.array([ros_data.D])
        #if len(distCoeffs[0]) < 4:
        #    distCoeffs=np.array([[0,0,0,0,0]])
        self.CamParam.distCoeff=distCoeffs
        self.CamParam.camMatrix=camMatrix
        
    
            
            
    def callback(self, data):
        self.FTdata = np.array([data.ft_wrench.torque.x,data.ft_wrench.torque.y,data.ft_wrench.torque.z,\
        data.ft_wrench.force.x,data.ft_wrench.force.y,data.ft_wrench.force.z])
        self.ft_flag=True

    
    def trapezoid_gen(self,target,current_joint_angles,acc,dx):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        goal.trajectory.header.frame_id='/world'
    
        dist = np.array(target-current_joint_angles)
        duration = np.max(np.sqrt(np.abs(9.0*dist/2.0/acc)))
        
        vmax = 1.5/duration
        amax  = 3*vmax/duration
        dmax  = amax
        [x0,v0,a0,ta,tb,tf] = trapgen(0,1,0,0,vmax,amax,dmax,0)
        [xa,va,aa,ta,tb,tf] = trapgen(0,1,0,0,vmax,amax,dmax,ta)
        [xb,vb,ab,ta,tb,tf] = trapgen(0,1,0,0,vmax,amax,dmax,tb)
    		
        duration = np.max(np.sqrt(np.abs(9.0*dist/2.0/acc)))
        vmax = 1.5*dist/duration
        acc = 3*vmax/duration
    	
        p1=JointTrajectoryPoint()
        p1.positions = current_joint_angles
        p1.velocities = np.zeros((6,))
        p1.accelerations = aa*dist
        p1.time_from_start = rospy.Duration(0)
        
        p2=JointTrajectoryPoint()
        p2.positions = np.array(p1.positions) + dist*xa
        p2.velocities = va*dist
        p2.accelerations = np.zeros((6,))
        p2.time_from_start = rospy.Duration(ta)
        
        p3=JointTrajectoryPoint()
        p3.positions = np.array(p1.positions) + dist*xb
        p3.velocities = vb*dist
        p3.accelerations = -ab*dist
        p3.time_from_start = rospy.Duration(tb)
    
        p4=JointTrajectoryPoint()
        p4.positions = target
        p4.velocities = np.zeros((6,))
        p4.accelerations = np.zeros((6,))
        p4.time_from_start = rospy.Duration(tf)
        
        goal.trajectory.points.append(p1)
        goal.trajectory.points.append(p2)
        goal.trajectory.points.append(p3)
        goal.trajectory.points.append(p4)
        
        return goal
        
    def compute_ft_est(self):
        Tran_z = np.array([[0,0,-1],[0,-1,0],[1,0,0]])    
        Vec_wrench = 100*np.array([0.019296738361905,0.056232033265447,0.088644197659430,    
        0.620524934626544,-0.517896661195076,0.279323567303444,-0.059640563813256,   
        0.631460085138371,-0.151143175570223,-6.018321330845553]).transpose()
        T = self.listener.lookupTransform("base", "link_6", rospy.Time(0))
        rg = 9.8*np.matmul(np.matmul(T.R,Tran_z).transpose(),np.array([0,0,1]).transpose())
        A1 = np.hstack([rox.hat(rg).transpose(),np.zeros([3,1]),np.eye(3),np.zeros([3,3])])
        A2 = np.hstack([np.zeros([3,3]),rg.reshape([3,1]),np.zeros([3,3]),np.eye(3)])   
        A = np.vstack([A1,A2])
        return np.matmul(A,Vec_wrench)



    def get_pose(self, board, corners, ids, CamParam):
        #Define object and image points of both tags        
        objPoints, imgPoints 	=	aruco.getBoardObjectAndImagePoints(board, corners, ids)
        objPoints = objPoints.reshape([objPoints.shape[0],3])
        imgPoints = imgPoints.reshape([imgPoints.shape[0],2])
        
        #Get pose of both ground and panel markers from detected corners        
        retVal, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, CamParam.camMatrix, CamParam.distCoeff)
        Rca, b = cv2.Rodrigues(rvec)
        
        return imgPoints,rvec, tvec, Rca, b
    

    def image_jacobian_gen(self, result, corners, ids, CamParam, board_ground,board_panel,id_start_ground, id_board_ground_size, tag_ground_size, loaded_object_points_ground_in_panel_system, display_window):

        rospy.loginfo(str(id_start_ground))
        rospy.loginfo(str(id_board_ground_size))
        rospy.loginfo(str(tag_ground_size))
        rospy.loginfo(str(corners)) #float32
        rospy.loginfo(str(board_ground))  
        rospy.loginfo(str(board_panel)) 

        idx_max = 180
        UV = np.zeros([idx_max,8])
        P = np.zeros([idx_max,3])
        id_valid = np.zeros([idx_max,1])
        
        f_hat_u = CamParam.camMatrix[0][0]
        f_hat_v = CamParam.camMatrix[1][1]
        f_0_u = CamParam.camMatrix[0][2]
        f_0_v = CamParam.camMatrix[1][2]
        
        imgPoints_ground, rvec_ground, tvec_ground, Rca_ground, b_ground = self.get_pose(board_ground, corners, ids, CamParam)
        imgPoints_panel, rvec_panel, tvec_panel, Rca_panel, b_panel = self.get_pose(board_panel, corners, ids, CamParam)
        
        corners_ground = []
        corners_panel = []
        for i_ids,i_corners in zip(ids,corners):
            if i_ids<=(id_start_ground+id_board_ground_size):
                corners_ground.append(i_corners)
            else:
                corners_panel.append(i_corners)
        #rospy.loginfo(str(id_start_ground))
        
        rvec_all_markers_ground, tvec_all_markers_ground, _ = aruco.estimatePoseSingleMarkers(corners_ground, tag_ground_size, CamParam.camMatrix, CamParam.distCoeff)
        rvec_all_markers_panel, tvec_all_markers_panel, _ = aruco.estimatePoseSingleMarkers(corners_panel, 0.025, CamParam.camMatrix, CamParam.distCoeff)
        #rospy.loginfo(str(tvec_all_markers_ground))
        #rospy.loginfo(str(tvec_all_markers_panel))
        tvec_all=np.concatenate((tvec_all_markers_ground,tvec_all_markers_panel),axis=0)
        
        
        for i_ids,i_corners,i_tvec in zip(ids,corners,tvec_all):
            if i_ids<idx_max:
                #print 'i_corners',i_corners,i_corners.reshape([1,8])
                UV[i_ids,:] = i_corners.reshape([1,8]) #np.average(i_corners, axis=1) 
                P[i_ids,:] = i_tvec
                id_valid[i_ids] = 1
    
        
        id_select = range(id_start_ground,(id_start_ground+id_board_ground_size))
        #used to find the height of the tags and the delta change of height, z height at desired position
        Z = P[id_select,2] #- [0.68184539, 0.68560932, 0.68966803, 0.69619578])
        id_valid = id_valid[id_select]
    
        dutmp = []
        dvtmp = []
        
        #Pixel estimates of the ideal ground tag location
        reprojected_imagePoints_ground_2, jacobian2	=	cv2.projectPoints(	loaded_object_points_ground_in_panel_system.transpose(), rvec_panel, tvec_panel, CamParam.camMatrix, CamParam.distCoeff)
        reprojected_imagePoints_ground_2 = reprojected_imagePoints_ground_2.reshape([reprojected_imagePoints_ground_2.shape[0],2])
       
        if(display_window):
            frame_with_markers_and_axis = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            frame_with_markers_and_axis	=	cv2.aruco.drawAxis(	frame_with_markers_and_axis,  CamParam.camMatrix, CamParam.distCoeff, Rca_ground, tvec_ground, 0.2	)
            frame_with_markers_and_axis	=	cv2.aruco.drawAxis(	frame_with_markers_and_axis,  CamParam.camMatrix, CamParam.distCoeff, Rca_panel, tvec_panel, 0.2	)
        
            #plot image points for ground tag from corner detection and from re-projections
            for point1,point2 in zip(imgPoints_ground,np.float32(reprojected_imagePoints_ground_2)):
                cv2.circle(frame_with_markers_and_axis,tuple(point1),5,(0,0,255),3)
                cv2.circle(frame_with_markers_and_axis,tuple(point2),5,(255,0,0),3) 
                
            height, width, channels = frame_with_markers_and_axis.shape
            cv2.imshow(display_window,cv2.resize(frame_with_markers_and_axis, (width/4, height/4)))
            cv2.waitKey(1)
            #Save
            #filename_image = "/home/rpi-cats/Desktop/DJ/Code/Images/Panel2_Acquisition_"+str(t1)+"_"+str(iteration)+".jpg"
            #scipy.misc.imsave(filename_image, frame_with_markers_and_axis)
        
        reprojected_imagePoints_ground_2 = np.reshape(reprojected_imagePoints_ground_2,(id_board_ground_size,8))
        #Go through a particular point in all tags to build the complete Jacobian
        for ic in range(4):
            #uses first set of tags, numbers used to offset camera frame, come from camera parameters               
            UV_target = np.vstack([reprojected_imagePoints_ground_2[:,2*ic]-f_0_u, reprojected_imagePoints_ground_2[:,2*ic+1]-f_0_v]).T
            uc = UV_target[:,0]
            vc = UV_target[:,1]
    
            UV_current = np.vstack([UV[id_select,2*ic]-f_0_u, UV[id_select,2*ic+1]-f_0_v]).T
            #find difference between current and desired tag difference
            delta_UV = UV_target-UV_current
    
            delet_idx = []
            J_cam_tmp =np.array([])
            for tag_i in range(id_board_ground_size):
                if id_valid[tag_i] == 1:
                    tmp = 1.0*np.array([[uc[tag_i]*vc[tag_i]/f_hat_u, -1.0*(uc[tag_i]*uc[tag_i]/f_hat_u + f_hat_u), vc[tag_i],-f_hat_u/Z[tag_i], 0.0, uc[tag_i]/Z[tag_i]],
                                               [ vc[tag_i]*vc[tag_i]/f_hat_v+f_hat_v, -1.0*uc[tag_i]*vc[tag_i]/f_hat_v, -uc[tag_i],0.0, -f_hat_v/Z[tag_i], vc[tag_i]/Z[tag_i]]])
                    if not (J_cam_tmp).any():
                        J_cam_tmp = tmp
                    else:
                        J_cam_tmp= np.concatenate((J_cam_tmp, tmp), axis=0)
                else:
                    delet_idx.append(tag_i)
                    
            delta_UV = np.delete(delta_UV, delet_idx, 0)
            dutmp.append(np.mean(delta_UV[:,0]))
            dvtmp.append(np.mean(delta_UV[:,1]))
            #camera jacobian
            if ic ==0:
                J_cam = J_cam_tmp
                delta_UV_all = delta_UV.reshape(np.shape(delta_UV)[0]*np.shape(delta_UV)[1],1)
                UV_target_all = UV_target.reshape(np.shape(UV_target)[0]*np.shape(UV_target)[1],1)
            else:
                J_cam = np.vstack([J_cam,J_cam_tmp])
                delta_UV_all = np.vstack([delta_UV_all,delta_UV.reshape(np.shape(delta_UV)[0]*np.shape(delta_UV)[1],1)]) 
                UV_target_all = np.vstack([UV_target_all,UV_target.reshape(np.shape(UV_target)[0]*np.shape(UV_target)[1],1)])
    
        du = np.mean(np.absolute(dutmp))
        dv = np.mean(np.absolute(dvtmp))
        print 'Average du of all points:',du
        print 'Average dv of all points:',dv
        
        return du, dv, J_cam, delta_UV_all
    
        
    
    def take_image(self):
        ####################### SET INITIAL POSE BASED ON PBVS #######################
        #### Final Nest Placement Error Calculation ===============================
        #Read new image
        self.last_ros_image_stamp = self.object_commander.ros_image_stamp        
        try:
            self.ros_gripper_2_trigger.wait_for_service(timeout=0.1)
            self.ros_gripper_2_trigger()
        except:
            pass
        wait_count=0
        while self.object_commander.ros_image is None or self.object_commander.ros_image_stamp == self.last_ros_image_stamp:
            if wait_count > 50:
                raise Exception("Image receive timeout")
            time.sleep(0.25)
            wait_count += 1
        self.result = self.object_commander.ros_image
        
    
    def move_to_initial_pose(self):
                
        #Set controller command mode
        self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY, 0.4, [],[])
        time.sleep(0.5)
        
        #open loop set the initial pose to the end pose in the transport path
        #pose_target2 = rox.Transform(rot0, tran0)        
        ##Execute movement to set location
        rospy.loginfo("Executing initial path ====================")
        self.controller_commander.compute_cartesian_path_and_move(self.initial_pose, avoid_collisions=False)
    
    def pbvs_to_stage1(self):
        
        self.take_image() 
        #Detect tag corners in aqcuired image using aruco
        corners, ids, _ = cv2.aruco.detectMarkers(self.result, self.aruco_dict, parameters=self.parameters)    
        #Sort corners and ids according to ascending order of ids
        #rospy.loginfo("tvec difference: %f, %f",corners,ids)
        corners, ids = sort_corners(corners,ids)
        
        # Estimate Poses  
        imgPoints_ground,rvec_ground, tvec_ground, Rca_ground, b_ground = self.get_pose(self.board_ground, corners, ids, self.CamParam)
        imgPoints_panel,rvec_panel, tvec_panel, Rca_panel, b_panel = self.get_pose(self.board_panel, corners, ids, self.CamParam)
        rospy.loginfo("camera params")
        rospy.loginfo(str(self.CamParam.distCoeff))
        rospy.loginfo(str(self.CamParam.camMatrix))
        observed_tvec_difference = tvec_ground-tvec_panel
        observed_rvec_difference = rvec_ground-rvec_panel
        rospy.loginfo(str(type(observed_tvec_difference)))
        rospy.loginfo("============== Difference in pose difference in nest position")
        rospy.loginfo(str(observed_tvec_difference))
        rospy.loginfo(str(observed_rvec_difference))
        
        tvec_err = self.loaded_tvec_difference_stage1-observed_tvec_difference
        rvec_err = self.loaded_rvec_difference_stage1-observed_rvec_difference 
        rospy.loginfo("rvec difference: %f, %f, %f",tvec_err[0],tvec_err[1],tvec_err[2])
        rospy.loginfo("rvec difference: %f, %f, %f",rvec_err[0],rvec_err[1],rvec_err[2])

        # Adjustment
        rospy.loginfo("PBVS to initial Position ====================")
        current_joint_angles = self.controller_commander.get_current_joint_values()
        dx = np.array([0,0,0, -tvec_err[0], tvec_err[1],tvec_err[2]])*0.75
        joints_vel = QP_abbirb6640(np.array(current_joint_angles).reshape(6, 1),np.array(dx))
        goal = self.trapezoid_gen(np.array(current_joint_angles) + joints_vel.dot(1),np.array(current_joint_angles),0.25,np.array(dx))
        
        self.client.wait_for_server()
        self.client.send_goal(goal)
        self.client.wait_for_result()
        res = self.client.get_result()
        if (res.error_code != 0):
            raise Exception("Trajectory execution returned error")

        rospy.loginfo("End of Initial Pose ====================")
        ### End of initial pose 

    def final_adjustment(self):
        ############################## FINAL ADJUSTMENT ##############################    
        rospy.loginfo("Final Adjustment")
        self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY, 0.2, [],[])
    
        self.take_image()    
        
        #Detect tag corners in aqcuired image using aruco
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.result, self.aruco_dict, parameters=self.parameters)    
        #Sort corners and ids according to ascending order of ids
        corners, ids = sort_corners(corners,ids)
        
        # Estimate Poses  
        imgPoints_ground, rvec_ground, tvec_ground, Rca_ground, b_ground = self.get_pose(self.board_ground, corners, ids, self.CamParam)
        imgPoints_panel, rvec_panel, tvec_panel, Rca_panel, b_panel = self.get_pose(self.board_panel, corners, ids, self.CamParam)
 

        observed_tvec_difference = tvec_ground-tvec_panel
        observed_rvec_difference = rvec_ground-rvec_panel    
        rospy.loginfo("============== Difference in nest position (before adjusment)")
        tvec_err = self.loaded_tvec_difference-observed_tvec_difference
        rvec_err = self.loaded_rvec_difference-observed_rvec_difference 
        rospy.loginfo("tvec difference: %f, %f, %f",tvec_err[0],tvec_err[1],tvec_err[2])
        rospy.loginfo("rvec difference: %f, %f, %f",rvec_err[0],rvec_err[1],rvec_err[2])
        
        current_joint_angles = self.controller_commander.get_current_joint_values()
        dx = np.array([0,0,0, -tvec_err[0], tvec_err[1],0])
        joints_vel = QP_abbirb6640(np.array(current_joint_angles).reshape(6, 1),np.array(dx))
        goal = self.trapezoid_gen(np.array(current_joint_angles) + joints_vel.dot(1),np.array(current_joint_angles),0.25,np.array(dx))
        
        self.client.wait_for_server()
        self.client.send_goal(goal)
        self.client.wait_for_result()
        res = self.client.get_result()
        if (res.error_code != 0):
            raise Exception("Trajectory execution returned error")

    	
        self.take_image()
        
        #Detect tag corners in aqcuired image using aruco
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.result, self.aruco_dict, parameters=self.parameters)       
        #Sort corners and ids according to ascending order of ids
        corners, ids = sort_corners(corners,ids)
        
        # Estimate Poses  
        imgPoints_ground, rvec_ground, tvec_ground, Rca_ground, b_ground = self.get_pose(self.board_ground, corners, ids, self.CamParam)
        imgPoints_panel, rvec_panel, tvec_panel, Rca_panel, b_panel = self.get_pose(self.board_panel, corners, ids, self.CamParam)
        
        observed_tvec_difference = tvec_ground-tvec_panel
        observed_rvec_difference = rvec_ground-rvec_panel    
        rospy.loginfo("============== Difference in nest position (after adjusment)")
        tvec_err = self.loaded_tvec_difference-observed_tvec_difference
        rvec_err = self.loaded_rvec_difference-observed_rvec_difference 
        rospy.loginfo("tvec difference: %f, %f, %f",tvec_err[0],tvec_err[1],tvec_err[2])
        rospy.loginfo("rvec difference: %f, %f, %f",rvec_err[0],rvec_err[1],rvec_err[2])


    def release_suction_cups(self):
        ################## RELEASE SUCTION CUPS AND LIFT THE GRIPPER ################## 	
        rospy.loginfo("============ Release Suction Cups...")
        self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY, 0.7, [])
        self.rapid_node.set_digital_io("Vacuum_enable", 0)
        #g = ProcessStepGoal('plan_place_set_second_step',"")
        #process_client.send_goal(g)
        time.sleep(0.5)
    
        Cur_Pose = self.controller_commander.get_current_pose_msg()
        rot_current = [Cur_Pose.pose.orientation.w, Cur_Pose.pose.orientation.x,Cur_Pose.pose.orientation.y,Cur_Pose.pose.orientation.z]
        trans_current = [Cur_Pose.pose.position.x,Cur_Pose.pose.position.y,Cur_Pose.pose.position.z]
        pose_target2 = rox.Transform(rox.q2R([rot_current[0], rot_current[1], rot_current[2], rot_current[3]]), trans_current)
        pose_target2.p[2] += 0.25
    
        rospy.loginfo("============ Lift gripper...")
        self.controller_commander.compute_cartesian_path_and_move(pose_target2, avoid_collisions=False)
        #s=ProcessState()
        #s.state="place_set"
        #s.payload="leeward_tip_panel"
        #s.target=""
        #process_state_pub.publish(s)
        
        
    def ibvs_placement(self):
        # INPUT: 
        #            Raw image from gripper camera (Cam#636)
        #            Robot state
        #            Panel ID - 
        #                   (1) Desired placement calibration results
        #                   (2) Initial position
        #                   (3) Board info 
        #            F/T sensor data
        #            Camera info    
        # OUTPUT: 
        #            Sequential robot joint command to achieve the desired results
        # PARAMETERS: 
        #            IBVS - 
        #                   (1) Convergence threshold (du_converge_TH, dv_converge_TH, iteration_limit)
        #                   (2) Image Jacobian Gain (Ki)
        #            Complaince controller:
        #                   (1) Desired force set points (F_d_set1, F_d_set2)
        #                   (2) Complaince control gain (Kc)
 
        #test = 0            
      
        ################################# BEGIN IBVS #################################
        step_size_min = self.step_size_min
        loaded_object_points_ground_in_panel_system = self.loaded_object_points_ground_in_panel_system_stage_2 
        du = 100.0
        dv = 100.0     
        
        self.FTdata_0 = self.FTdata
        self.FTdata_0est = self.compute_ft_est()
        
        while ((du>self.du_converge_TH) | (dv>self.dv_converge_TH) and (self.iteration<self.iteration_limit)):    
            #try changing du and dv to lower values(number of iterations increases)
            self.iteration += 1
        
            #Print current robot pose at the beginning of this iteration
            Cur_Pose = self.controller_commander.get_current_pose_msg()
            
            self.take_image()
            
            #Detect tag corners in aqcuired image using aruco
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.result, self.aruco_dict, parameters=self.parameters)    
            #Sort corners and ids according to ascending order of ids
            corners, ids = sort_corners(corners,ids)
    
            #check if all tags detected
            if len(ids) >10:#retVal_ground != 0 and retVal_panel !=0:        
                du, dv, J_cam, delta_UV_all = self.image_jacobian_gen(self.result, corners, ids, self.CamParam, self.board_ground,self.board_panel,self.id_start_ground, 
                                                                      self.id_board_row_ground*self.id_board_col_ground,self.tag_ground_size,loaded_object_points_ground_in_panel_system,'Cam0')
       
                dx = QP_Cam(np.dot(J_cam, self.R_Jcam),self.Ki*delta_UV_all)
                dx = dx.reshape([6,1])
       
                # Compliance Force Control
                FTdata_est = self.compute_ft_est()
                if(not self.ft_flag):
                    raise Exception("havent reached callback")
                FTread = self.FTdata - self.FTdata_0 - FTdata_est + self.FTdata_0est
                print '=====================FT============:',FTread[-1], self.FTdata[-1], self.FTdata_0[-1], FTdata_est[-1], self.FTdata_0est[-1]
                if FTread[-1]> (self.F_d_set1+50):
                    F_d = self.F_d_set1                   
                else:
                    F_d = self.F_d_set2
    
                if (self.FTdata==0).all():
                    rospy.loginfo("FT data overflow")
                else:
                    Vz = self.Kc*(F_d - FTread[-1])
                    dx[-1] = dx[-1]+Vz

                current_joint_angles = self.controller_commander.get_current_joint_values()
    
                step_size_tmp = np.linalg.norm(dx)
                if step_size_tmp <= step_size_min:
                    step_size_min = step_size_tmp
                else:
                    dx = dx/step_size_tmp*step_size_min
    
                joints_vel = QP_abbirb6640(np.array(current_joint_angles).reshape(6, 1),np.array(dx))
    
                goal = self.trapezoid_gen(np.array(current_joint_angles) + joints_vel.dot(self.dt),np.array(current_joint_angles),0.25,np.array(dx))
    
              
                self.client.wait_for_server()     
                self.client.send_goal(goal)
                self.client.wait_for_result()
                res = self.client.get_result()
                if (res.error_code != 0):
                    raise Exception("Trajectory execution returned error")

                rospy.loginfo('Current Iteration Finished.')


        rospy.loginfo("============= iteration =============")
        rospy.loginfo("iteration: %f",self.iteration)
        
    def pointarray_to_array(self,data):
        arr=np.zeros((3,len(data)))
        #arr_column=np.empty((3,0))
        
        for i in range(len(data)):
            #arr_column[0]=data[i].x
            #arr_column[1]=data[i].y
            #arr_column[2]=data[i].z
            arr_column=np.array([[data[i].x],[data[i].y],[data[i].z]])
            #rospy.loginfo(str(arr_column))
            arr[:,i]=arr_column.reshape([3,])
            #rospy.loginfo(str(arr))
        #return np.asmatrix(arr,float)
        rospy.loginfo(str(arr))
        return arr
                
        
    def vector3_to_array(self,data):
        arr =np.asarray( [[data.x],[data.y],[data.z]]) 
        '''
        arr=np.empty(3)
        arr[0]=data.x
        arr[1]=data.y
        arr[2]=data.z
        
        arr=[]
        
        arr.append(float(data.x))
        arr.append(float(data.y))
        arr.append(float(data.z))
        '''
        rospy.loginfo(str(arr))
        #arr.reshape([3,1])
        rospy.loginfo(str(arr))
        return arr
    
    #NOT FUNCTIONING
    def aruco_dicts(self,aruco_dict_str):
        aruco_dicts=set()
        aruco_dicts.add(aruco_dict_str)
        if not hasattr(cv2.aruco, next(iter(aruco_dicts))):
            raise ValueError("Invalid aruco-dict value")
        aruco_dict_id=getattr(cv2.aruco, next(iter(aruco_dicts)))
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
        return aruco_dict
        
    def single_camera_placement(self,data,camera_1_ground,camera_1_place):
        #try:
        self.dt = data.ibvs_parameters.IBVSdt
        self.iteration=0
        #aruco_dict_panel=self.aruco_dicts(camera_1_place.dictionary)
        #aruco_dict_ground=self.aruco_dicts(camera_1_ground.dictionary)
        aruco_dict_panel = self.aruco_dicts(camera_1_place.dictionary)
        
        aruco_dict_ground = self.aruco_dicts(camera_1_ground.dictionary)
        self.board_panel = cv2.aruco.GridBoard_create(camera_1_place.markersX, camera_1_place.markersY, camera_1_place.markerLength, camera_1_place.markerSpacing, aruco_dict_panel, camera_1_place.firstMarker)
        
        self.board_ground = cv2.aruco.GridBoard_create(camera_1_ground.markersX, camera_1_ground.markersY, camera_1_ground.markerLength, camera_1_ground.markerSpacing, aruco_dict_ground, camera_1_ground.firstMarker)
        
        self.id_start_ground=camera_1_ground.firstMarker
        self.id_board_row_ground=camera_1_ground.markersX
        self.id_board_col_ground=camera_1_ground.markersY
        self.tag_ground_size=camera_1_ground.markerLength
        self.loaded_object_points_ground_in_panel_system_stage_2 = self.pointarray_to_array(data.point_difference_stage2)
        # --- Load ideal pose difference information from file
        self.loaded_rvec_difference_stage1 = self.vector3_to_array(data.rvec_difference_stage1)
        self.loaded_tvec_difference_stage1 = self.vector3_to_array(data.tvec_difference_stage1)
        self.loaded_tvec_difference_stage1[1]+=0.03
        # --- Load ideal pose difference information from file
        self.loaded_rvec_difference = self.vector3_to_array(data.rvec_difference_stage2)
        self.loaded_tvec_difference = self.vector3_to_array(data.tvec_difference_stage2)
        
        self.du_converge_TH = data.ibvs_parameters.du_converge_th
        self.dv_converge_TH = data.ibvs_parameters.dv_converge_th
        self.iteration_limit = data.ibvs_parameters.iteration_limit
        self.Ki = data.ibvs_parameters.Ki
        # Compliance controller parameters
        self.F_d_set1 = data.compliance_control_parameters.F_d_set1
        self.F_d_set2 = data.compliance_control_parameters.F_d_set2
        self.Kc = data.compliance_control_parameters.Kc
        self.initial_pose=data.initial
        #self.tran0 = np.array([2.15484,1.21372,0.25766])
        #self.rot0 = rox.q2R([0.02110, -0.03317, 0.99922, -0.00468])
        self.step_size_min = data.ibvs_parameters.step_size_min
        '''except Exception as err:
            rospy.loginfo("Input values for placement controller are invalid"+str(err))
            feedback=PlacementStepFeedback()
            
            feedback.error_msg="Input values are invalid"
            self.goal_handle.publish_feedback(feedback)
            self.goal_handle.set_aborted()
        '''
            
        
        self.move_to_initial_pose()
        self.pbvs_to_stage1()
        self.ibvs_placement()
        self.final_adjustment()
        self.release_suction_cups()
        res = PlacementStepResult()
        res.state="Placement_complete"
        

        self.goal_handle.set_succeeded(res)
        
        
    def two_camera_placement(self,data,camera_1_ground,camera_1_place,camera_2_ground,camera_2_place):
        self.dt = data.ibvs_parameters.IBVSdt
        self.iteration=0
        self.board_panel = cv2.aruco.GridBoard_create(camera_1_place.markersX, camera_1_place.markersY, camera_1_place.markerLength, camera_1_place.markerSpacing, camera_1_place.dictionary, camera_1_place.firstMarker)
        self.board_ground = cv2.aruco.GridBoard_create(camera_1_ground.markersX, camera_1_ground.markersY, camera_1_ground.markerLength, camera_1_ground.markerSpacing, camera_1_ground.dictionary, camera_1_ground.firstMarker)
        self.board_panel2 = cv2.aruco.GridBoard_create(camera_2_place.markersX, camera_2_place.markersY, camera_2_place.markerLength, camera_2_place.markerSpacing, camera_2_place.dictionary, camera_2_place.firstMarker)
        self.board_ground2 = cv2.aruco.GridBoard_create(camera_2_ground.markersX, camera_2_ground.markersY, camera_2_ground.markerLength, camera_2_ground.markerSpacing, camera_2_ground.dictionary, camera_2_ground.firstMarker)
        self.loaded_object_points_ground_in_panel_system_stage_2 = self.pointarray_to_array(data.point_difference_stage2)
        # --- Load ideal pose differnece information from file
        self.loaded_rvec_difference_stage1 = self.vector3_to_array(data.rvec_difference_stage1)
        self.loaded_tvec_difference_stage1 = self.vector3_to_array(data.tvec_difference_stage1)
        self.loaded_tvec_difference_stage1[1]+=0.03
        # --- Load ideal pose differnece information from file
        self.loaded_rvec_difference = self.vector3_to_array(data.rvec_difference_stage2)
        self.loaded_tvec_difference = self.vector3_to_array(data.tvec_difference_stage2)
        
        self.du_converge_TH = data.ibvs_parameters.du_converge_th
        self.dv_converge_TH = data.ibvs_parameters.dv_converge_th
        self.iteration_limit = data.ibvs_parameters.iteration_limit
        self.Ki = data.ibvs_parameters.Ki
        # Compliance controller parameters
        self.F_d_set1 = data.compliance_control_parameters.F_d_set1
        self.F_d_set2 = data.compliance_control_parameters.F_d_set2
        self.Kc = data.compliance_control_parameters.Kc
        self.initial_pose=data.initial
        #self.tran0 = np.array([2.15484,1.21372,0.25766])
        #self.rot0 = rox.q2R([0.02110, -0.03317, 0.99922, -0.00468])
        self.step_size_min = data.ibvs_parameters.step_size_min
        self.move_to_initial_pose()
        #self.pbvs_to_stage1()
        #self.ibvs_placement()
        self.final_adjustment()
        self.release_suction_cups()
        res = ProcessStepResult()
        res.state="Placement_complete"
        self.goal_handle.set_succeeded(res)
            
            
            
    

def main():
    rospy.init_node('Placement_Controller', anonymous=True)
    controller = PlacementController()
    controller.move_to_initial_pose()
    controller.pbvs_to_stage1()
    controller.ibvs_placement()
    controller.final_adjustment()
    controller.release_suction_cups()    

if __name__ == "__main__":
    main()
