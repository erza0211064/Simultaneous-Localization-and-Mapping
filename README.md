Copy right by TUNG-LIN, YANG / t2yang@eng.ucsd.edu

ECE 276A Project 2 Simultaneous-Localization-and-Mapping

-requirement:
	-opencv 4.0.0.21
	-numpy 1.15.2
	-encoder, imu, lidar data need to put in the same file
	-RGB data put in 'dataRGBD/RGB'+{train data label}
	-disparity data put in 'dataRGBD/Disparity' +{train data label}
-mapping.py
main function, load all data and implement particle filter
Function:
	-draw_map
	load data, implement particle filter, draw grid map and texture map
	-predict
	prediction step in particle filter
	-update 
	update step in particle filter
	-stratified_resample
	resample when weights are too small
	-mapping
	draw map base on lidar
	-time_align
	time align
-encoder.py
Function:
	-get_dynamic
	get motion model
-map_utils.py
Function:
	-texture_mapping
	transform pixel to world frame and draw texture_mapping
	*mapCorrelation
	*bresenham2D
-implement train data20, 21
change dataset in mapping.py function main() dataset parameter, input_name='train'
python mapping.py
-implement test data
change dataset in mapping.py function main() dataset parameter, input_name='test'
python mapping.py
video is include in zip file.
