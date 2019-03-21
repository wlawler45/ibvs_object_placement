from __future__ import absolute_import
#TODO: import placement controller class like shown in next line
from .arm_composites_manufacturing_placement import PlacementController
from ibvs_object_placement.msg import PlacementStepAction, PlacementStepResult,PlacementStepFeedback
import actionlib
import rospy

class PlacementControllerServer(object):
    def __init__(self):
        #TODO: Replace next line with creation of placement controller class, it should still be called controller 
        self.controller=PlacementController()
        self.server=actionlib.ActionServer("placement_step", PlacementStepAction, goal_cb=self.execute_cb,cancel_cb=self.cancel, auto_start=False)
        
        self.server.start()
        self.previous_goal=None
        
    def cancel(self,goal):
        self.previous_goal.set_canceled()
        
    def execute_cb(self, goal):
        self.controller.goal_handle=goal
        goal.set_accepted()
        data = goal.get_goal().data
        camera_1_ground = goal.get_goal().camera1ground
        camera_1_place= goal.get_goal().camera1place
        self.controller.goal_handle=goal
        self.previous_goal=goal
        if len(data.cameras) == 2:
            camera_2_ground= goal.get_goal().camera2ground
            camera_2_place=goal.get_goal().camera2place
            self.controller.two_camera_placement(data,camera1ground,camera1place,camera2ground,camera2place)
        else:
            self.controller.single_camera_placement(data,camera1ground,camera1place)
        
            
        rospy.loginfo(goal.get_goal_status())
        
        res = PlacementStepResult()
        res.state=self.controller.state
        res.target=self.controller.current_target if self.controller.current_target is not None else ""
        res.payload=self.controller.current_payload if self.controller.current_payload is not None else ""
    
        goal.set_succeeded(res)
        
            
def placement_controller_server_main():
    rospy.init_node("placement_controller_server")
    
    
    s=PlacementControllerServer()
    
    rospy.spin()
