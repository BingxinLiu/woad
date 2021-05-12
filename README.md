# WOAD :  A Wearable Obstacle-Avoid Detector
FILE STURCTURE:  
+WOAD  
|- para  
|---| 3dmap_set copy.yml  // arguments for algorithm  
|- photos  
|---| *L.jpg     //the left photos saved from pictures  
|---| *R.jpg     //the right photos saved from pictures  
|- getfps.cpp // program for testing frame per second  
|- photo.cpp     //take pictures  
|- woad_bm.cpp   //program with BM algorithm  
|- woad_cuda.cpp //program with BM algorithm accelerated by CUDA  
|- woad_sgbm.cpp // program with SGBM, c++ version  
|- woad_sgbm.py // program with SGBM, python version
|- woad_sgm.cpp // program with SGBM, accelerated by CUDA  
|- CMakeLists.txt // makefile
#

INSTALL:
> mkdir build  
> cd build  
> cmake ..  
> make  

You may need modify CMakeLists.txt to compile different files.

#
MODIFICATION:

You may need to modify some part of the program to run correctly.  

- [ ] change 'folder_name' to your local path to woad directory  
- [ ] change variables cameraMatrixL, cameraMatrixR, distCoeffL, distCoeffR, T, and R to your own clibration data.  
- [ ] change photo_width and photo_height variable to your camera size.  
- [ ] change the content in cam0 and cam1 to your camera's capture command.  

#
AUTHORS:  
> Hao Liu  
> Bingxin Liu  
> Guohao Pu  