#An action call to the placement controller
#panel type being placed
string[] cameras

#The spacing between markers in meters
geometry_msgs/Vector3 rvec_difference_stage1
geometry_msgs/Vector3 tvec_difference_stage1
geometry_msgs/Vector3 rvec_difference_stage2
geometry_msgs/Vector3 tvec_difference_stage2
geometry_msgs/Point[] point_difference_stage2


ibvs_object_placement/IBVSParameters ibvs_parameters
ibvs_object_placement/ComplianceControlParameters compliance_control_parameters


#initial placement pose
geometry_msgs/Pose initial




