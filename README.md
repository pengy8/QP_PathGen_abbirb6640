# QP_PathGen_abbirb6640
It is modified from repo: 
https://github.com/ShuyoungChen/quadprog 
for ABB IRB6640 path generation with quadratic programming

ABB ROS packages have to be installed to get the required simulation models:
https://github.com/ShuyoungChen/irb6640

After installing OpenRAVE and or_urdf, run under the downloaded directory in a Linux terminal:
python main_quadprog_pathgen.py
And it will generate a Path file "Joint_All.out" that includes all the joint angles along the path.
