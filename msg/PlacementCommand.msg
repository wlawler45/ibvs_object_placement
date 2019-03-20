#An action call to the placement controller
#panel type being placed
int32 cameras_used

#The spacing between markers in meters
float64[] rvec_difference_stage1
float64[] tvec_difference_stage1
float64[] rvec_difference_stage2
float64[] tvec_difference_stage2
float64[] point_difference_stage2

#Compliance control parameters
int32 F_d_set1
int32 F_d_set2
float64 Kc

#IBVS parameters
float64 IBVSdt
int32 du_converge_th
int32 dv_converge_th
int32 iteration_limit
float64 Ki
int32 step_size_min

#initial placement pose
geometry_msgs/Pose initial




