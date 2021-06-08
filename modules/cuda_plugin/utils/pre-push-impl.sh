#!/bin/bash

# cmake build targets / test suities to run
SUITES=(CudaUnitTests CudaFuncTests)
declare -A SUITES_ARGS=( ["CudaFuncTests"]='--gtest_filter=*smoke_AvgPool*netPRC=FP32*:*smoke_MatMul*netPRC=FP32*:*smoke_MaxPool*netPRC=FP32*:*smoke_Convert*netPRC=FP32*:*smoke_*SqueezeUnsqueeze*netPRC=FP32*:*SoftMax*netPRC=FP32*:*smoke_*Transpose*:*smoke_*ConvolutionLayerTest*:*smoke_Add*')

# $EXIT_CODE variable
EXIT_CODE=0

# if $OV_BUILD_PATH environment variable not set try to set it to
# openvino/build assuming openvino directroy is in the same directrory
# as openvino_cuda_plugin
if [ -z "$OV_BUILD_PATH" ]; then
  export OV_BUILD_PATH=$(realpath ../../../../openvino/build)
fi
echo OV_BUILD_PATH=$OV_BUILD_PATH

if [ ! -d $OV_BUILD_PATH ]; then
  echo "Couldn't find OV_BUILD_PATH=$OV_BUILD_PATH"
  echo "Please specify OV_BUILD_PATH enviroment variable"
  echo "e.g. export OV_BUILD_PATH=~/openvino/build"
  exit 2
fi

# building targets and bitwise adding the resulting exit code to
# $EXIT_CODE variable
cd $OV_BUILD_PATH
cmake --build . -j$(nproc) --target ${SUITES[*]}
((EXIT_CODE=EXIT_CODE | $?))

# trying to parse inference_engine_targets.cmake file in
# openvino build directroy to find the location of binaries
# this file contains the following lines
# IMPORTED_LOCATION_DEBUG "/home/user/ov/openvino/bin/intel64/Debug/lib/libinference_engine.so"
# or
# IMPORTED_LOCATION_RELEASE "/home/user/ov/openvino/bin/intel64/Release/lib/libinference_engine.so"
CMAKE_FILE_PATH=$OV_BUILD_PATH/inference_engine_targets.cmake
if [ ! -f $CMAKE_FILE_PATH ]; then
  echo "Couldn't find $CMAKE_FILE_PATH"
  echo "Try running cmake in openvino build"
  exit 3
fi

TYPES=(IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG)
for TYPE in ${TYPES[@]}; do
  GREPPED_STRING=$(grep $TYPE -m1 $CMAKE_FILE_PATH)
  if [ ! -z "$GREPPED_STRING" ]; then
    break
  fi
done
if [ -z "$GREPPED_STRING" ]; then
  echo "Couldn't find any of \"${TYPES[*]}\" in $CMAKE_FILE_PATH"
  exit 4
fi

SO_PATH=$(echo $GREPPED_STRING | grep -Eo '".*"' | tr -d '"')
LIB_PATH=$(dirname $SO_PATH)

# $TESTS_PATH should contain the path to the binaries
TESTS_PATH=$(dirname $(dirname $SO_PATH))
echo TESTS_PATH=$TESTS_PATH

if [ -z "$SO_PATH" ] || [ -z "$LIB_PATH" ] || [ -z "$TESTS_PATH" ]; then
  echo "Couldn't parse GREPPED_STRING=$GREPPED_STRING"
  echo SO_PATH=$SO_PATH
  echo LIB_PATH=$LIB_PATH
  exit 5
fi

if [ ! -d $TESTS_PATH ]; then
  echo "Couldn't find $TESTS_PATH"
  exit 6
else
  cd $TESTS_PATH
fi

# launching each of the specified test suites and bitwise adding its exit codes
# to $EXIT_CODE variable
for SUITE in ${SUITES[@]}; do
  if [ -f $SUITE ]; then
    eval ./"$SUITE" "${SUITES_ARGS[$SUITE]}"
    ((EXIT_CODE=EXIT_CODE | $?))
  else
    echo "Couldn't find $SUITE"
    EXIT_CODE=1
  fi
done

# passing the resulting EXIT_CODE to the caller
exit $EXIT_CODE
