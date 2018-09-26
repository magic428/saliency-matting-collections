cmake_minimum_required(VERSION 2.8)
project(salmat)

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
SET( CMAKE_BUILD_TYPE Release )

set( LINKER_LIBS "" )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories("include")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${OpenCV_LIBS})

# boost filesystem
find_package(Boost REQUIRED COMPONENTS
system
filesystem
)
list(APPEND LINKER_LIBS ${Boost_LIBRARIES})

aux_source_directory(src SRC_FILES)
aux_source_directory(src/common SRC_FILES)
aux_source_directory(src/segmentation SRC_FILES)
aux_source_directory(src/saliency SRC_FILES)
aux_source_directory(src/matting SRC_FILES)
# aux_source_directory(src/cluster SRC_FILES)

message("Source Files: " ${SRC_FILES} )
message("Header Files: " ${OpenCV_INCLUDE_DIRS} )
message("Lib Files: " ${LINKER_LIBS} )

add_executable(${PROJECT_NAME} ${SRC_FILES} )
target_link_libraries(${PROJECT_NAME} ${LINKER_LIBS})
