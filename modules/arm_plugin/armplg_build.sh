#!/bin/sh

set -x

BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
UPDATE_SOURCES=${UPDATE_SOURCES:-clean}
WITH_OMZ_DEMO=${WITH_OMZ_DEMO:-ON}

DEV_HOME=`pwd`
OPENCV_HOME=$DEV_HOME/opencv
OPENVINO_HOME=$DEV_HOME/openvino
OPENVINO_CONTRIB=$DEV_HOME/openvino_contrib
ARM_PLUGIN_HOME=$OPENVINO_CONTRIB/modules/arm_plugin
OMZ_HOME=$DEV_HOME/open_model_zoo
STAGING_DIR=$DEV_HOME/armcpu_package



fail()
{
    if [ $# -lt 2 ]; then
      echo "Script internal error"
      exit 31
    fi
    retval=$1
    shift
    echo $@
    exit $retval
}

cloneSrcTree()
{
    DESTDIR=$1
    shift
    SRCURL=$1
    shift
    while [ $# -gt 0 ]; do
        git lfs clone --recurse-submodules --shallow-submodules --depth 1 --branch=$1 $SRCURL $DESTDIR && return 0
        shift
    done
    return 1
}

checkSrcTree()
{
    [ $# -lt 3 ] && fail

    if ! [ -d $1 ]; then
        echo "Unable to detect $1"
        echo "Cloning $2..."
        cloneSrcTree $@ || fail 3 "Failed to clone $2. Stopping"
    else
        echo "Detected $1"
        echo "Considering it as source directory"
        if [ "$UPDATE_SOURCES" = "reload" ]; then
            echo "Source reloading requested"
            echo "Removing existing sources..."
            rm -rf $1 || fail 1 "Failed to remove. Stopping"
            echo "Cloning $2..."
            cloneSrcTree $@ || fail 3 "Failed to clone $2. Stopping"
        elif [ -d $1/build ]; then
            echo "Build directory detected at $1"
            if [ "$UPDATE_SOURCES" = "clean" ]; then
                echo "Cleanup of previous build requested"
                echo "Removing previous build results..."
                rm -rf $1/build || fail 2 "Failed to cleanup. Stopping"
            fi
        fi
    fi
    return 0
}



#Prepare sources
checkSrcTree $OPENCV_HOME https://github.com/opencv/opencv.git master
checkSrcTree $OPENVINO_HOME https://github.com/openvinotoolkit/openvino.git master
checkSrcTree $OPENVINO_CONTRIB https://github.com/openvinotoolkit/openvino_contrib.git master
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
    checkSrcTree $OMZ_HOME https://github.com/openvinotoolkit/open_model_zoo.git develop
fi

#cleanup package destination folder
[ -e $STAGING_DIR ] && rm -rf $STAGING_DIR
mkdir -p $STAGING_DIR

#Build OpenCV
mkdir -p $OPENCV_HOME/build && \
cd $OPENCV_HOME/build && \
PYTHONVER=`ls /usr/include | grep "python3[^m]*$"` && \
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_LIST=imgcodecs,videoio,highgui,gapi,python3 \
      -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DOPENCV_SKIP_PYTHON_LOADER=OFF \
      -DPYTHON3_LIMITED_API=ON -DPYTHON3_PACKAGES_PATH=$STAGING_DIR/opencv/python \
      -DPYTHON3_INCLUDE_PATH=/usr/include/${PYTHONVER} \
      -DPYTHON3_LIBRARIES=/usr/lib/$ARCH_NAME/lib${PYTHONVER}.so \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
      -D CMAKE_USE_RELATIVE_PATHS=ON \
      -D CMAKE_SKIP_INSTALL_RPATH=ON \
      -D OPENCV_SKIP_PKGCONFIG_GENERATION=ON \
      -D OPENCV_BIN_INSTALL_PATH=bin \
      -D OPENCV_INCLUDE_INSTALL_PATH=include \
      -D OPENCV_LIB_INSTALL_PATH=lib \
      -D OPENCV_CONFIG_INSTALL_PATH=cmake \
      -D OPENCV_3P_LIB_INSTALL_PATH=3rdparty \
      -D OPENCV_SAMPLES_SRC_INSTALL_PATH=samples \
      -D OPENCV_DOC_INSTALL_PATH=doc \
      -D OPENCV_OTHER_INSTALL_PATH=etc \
      -D OPENCV_LICENSES_INSTALL_PATH=etc/licenses \
      -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
      -DWITH_GTK_2_X=OFF \
      -DOPENCV_ENABLE_PKG_CONFIG=ON \
      -DPKG_CONFIG_EXECUTABLE=/usr/bin/${ARCH_NAME}-pkg-config \
      $OPENCV_HOME && \
make -j$BUILD_JOBS && \
cmake -DCMAKE_INSTALL_PREFIX=$STAGING_DIR/opencv -P cmake_install.cmake && \
echo export OpenCV_DIR=\$INSTALLDIR/opencv/cmake > $STAGING_DIR/opencv/setupvars.sh && \
echo export LD_LIBRARY_PATH=\$INSTALLDIR/opencv/lib:\$LD_LIBRARY_PATH >> $STAGING_DIR/opencv/setupvars.sh && \
cd $DEV_HOME || fail 11 "OpenCV build failed. Stopping"

#Build OpenVINO
mkdir -p $OPENVINO_HOME/build && \
cd $OPENVINO_HOME/build && \
cmake -DOpenCV_DIR=$STAGING_DIR/opencv/cmake -DENABLE_OPENCV=OFF \
      -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_GAPI_TESTS=OFF \
      -DENABLE_DATA=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$STAGING_DIR/opencv/lib \
      -DENABLE_MYRIAD=ON -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DTHREADING=SEQ -DENABLE_LTO=ON \
      -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
      -DARM_COMPUTE_SCONS_JOBS=$BUILD_JOBS \
      -DIE_EXTRA_MODULES=$ARM_PLUGIN_HOME \
      $OPENVINO_HOME && \
make -j$BUILD_JOBS && \
cmake -DCMAKE_INSTALL_PREFIX=$STAGING_DIR -P cmake_install.cmake && \
ARCHDIR=`ls $OPENVINO_HOME/bin` && \
cd $DEV_HOME || fail 12 "OpenVINO build failed. Stopping"

#OpenVINO python
[ "$UPDATE_SOURCES" = "clean" -a -e $OPENVINO_HOME/pbuild ] && rm -rf $OPENVINO_HOME/pbuild
mkdir -p $OPENVINO_HOME/pbuild && \
cd $OPENVINO_HOME/pbuild && \
cmake -DInferenceEngineDeveloperPackage_DIR=$OPENVINO_HOME/build \
      -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="/usr/bin/${PYTHONVER}" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_DATA=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$STAGING_DIR/opencv/lib \
      -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
      $OPENVINO_HOME/inference-engine/ie_bridges/python && \
make -j$BUILD_JOBS && \
cmake -DCMAKE_INSTALL_PREFIX=$STAGING_DIR -P cmake_install.cmake && \
cd $DEV_HOME || fail 13 "OpenVINO python bindings build failed. Stopping"

#Open Model Zoo
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
  OMZ_DEMOS_BUILD=$OMZ_HOME/build && \
  mkdir -p $OMZ_DEMOS_BUILD && \
  cd $OMZ_DEMOS_BUILD && \
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_PYTHON=OFF \
        -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
        -DInferenceEngine_DIR=$OPENVINO_HOME/build \
        -DOpenCV_DIR=$OPENCV_HOME/build \
        -Dngraph_DIR=$OPENVINO_HOME/build/ngraph \
        $OMZ_HOME/demos && \
  cmake --build $OMZ_DEMOS_BUILD -- -j$BUILD_JOBS && \
  cd $DEV_HOME || fail 16 "Open Model Zoo build failed. Stopping"
  mkdir -p $STAGING_DIR/deployment_tools/inference_engine/demos) && \
  cp -vr $OMZ_DEMOS_BUILD $STAGING_DIR/deployment_tools/inference_engine/demos) || \
  fail 21 "Open Model Zoo package structure preparation failed. Stopping"
fi

#Package creation
cd $STAGING_DIR && \
tar -czvf ../OV_ARM_package.tar.gz ./* || \
fail 23 "Package creation failed. Nothing more to do"

exit 0
