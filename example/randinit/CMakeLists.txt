cmake_minimum_required(VERSION 3.9)

if(CUDA_ENABLE)
	project(randInit LANGUAGES CXX CUDA)
	include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")
	link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64")
	add_executable(randInit test.cpp)
	target_link_libraries(randInit cuda.lib cudart.lib cudart_static.lib cublas.lib cudadevrt.lib)
else()
	project(randInit)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
	if(${PROJECT_DEBUG} STREQUAL "ON")
       		message(STATUS "build with debug mode")
       		set(CMAKE_CXX_FLAGS "-W -mavx2 -mfma -std=c++17 -g -O0")
	elseif(${PROJECT_DEBUG} STREQUAL "OFF")
       		message(STATUS "build with relase mode")
        	set(CMAKE_CXX_FLAGS "-W -mavx2 -mfma -std=c++17 -fopenmp -g -O3")	
	add_executable(randInit test.cpp)
	endif()
endif()
