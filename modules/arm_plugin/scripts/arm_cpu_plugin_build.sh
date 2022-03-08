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
        git clone --recurse-submodules --shallow-submodules --depth 1 --branch=$1 $SRCURL $DESTDIR && return 0
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
checkSrcTree $OPENCV_HOME https://github.com/opencv/opencv.git 4.x
checkSrcTree $OPENVINO_HOME https://github.com/openvinotoolkit/openvino.git master
checkSrcTree $OPENVINO_CONTRIB https://github.com/openvinotoolkit/openvino_contrib.git master
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
    checkSrcTree $OMZ_HOME https://github.com/openvinotoolkit/open_model_zoo.git master
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
      -DPYTHON3_LIMITED_API=ON \
      -DPYTHON3_INCLUDE_PATH=/opt/python3.7_arm/include/python3.7m \
      -DPYTHON3_LIBRARIES=/opt/python3.7_arm/lib \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.7/site-packages/numpy/core/include \
      -D CMAKE_USE_RELATIVE_PATHS=ON \
      -D CMAKE_SKIP_INSTALL_RPATH=ON \
      -D OPENCV_SKIP_PKGCONFIG_GENERATION=ON \
      -D OPENCV_BIN_INSTALL_PATH=bin \
      -D OPENCV_PYTHON3_INSTALL_PATH=python \
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
cmake -DCMAKE_INSTALL_PREFIX=$STAGING_DIR/extras/opencv -P cmake_install.cmake && \
echo export OpenCV_DIR=\$INSTALLDIR/extras/opencv/cmake > $STAGING_DIR/extras/opencv/setupvars.sh && \
echo export LD_LIBRARY_PATH=\$INSTALLDIR/extras/opencv/lib:\$LD_LIBRARY_PATH >> $STAGING_DIR/extras/opencv/setupvars.sh && \
mkdir -p $STAGING_DIR/python/python3 && cp -r $STAGING_DIR/extras/opencv/python/cv2 $STAGING_DIR/python/python3 && \
cd $DEV_HOME || fail 11 "OpenCV build failed. Stopping"

#Build OpenVINO
mkdir -p $OPENVINO_HOME/build && \
cd $OPENVINO_HOME/build && \
cmake -DOpenCV_DIR=$STAGING_DIR/extras/opencv/cmake -DENABLE_OPENCV=OFF \
      -DPYTHON_INCLUDE_DIRS="/opt/python3.7_arm/include/python3.7m" \
      -DPYTHON_LIBRARY="/opt/python3.7_arm/lib/libpython3.7m.so" \
      -DENABLE_PYTHON=ON \
      -DNGRAPH_PYTHON_BUILD_ENABLE=ON \
      -DNGRAPH_ONNX_IMPORT_ENABLE=ON \
      -DPYTHON_MODULE_EXTENSION=".so" \
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
      -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE="/opt/python3.7_arm/bin/python3.7m" \
      -DPYTHON_INCLUDE_DIRS=/opt/python3.7_arm/include/python3.7m \
      -DPYTHON_LIBRARIES=/opt/python3.7_arm/lib \
      -DPYTHON_MODULE_EXTENSION=".so" \
      -DPYBIND11_FINDPYTHON=OFF \
      -DPYBIND11_NOPYTHON=OFF \
      -DPYTHONLIBS_FOUND=TRUE \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DENABLE_DATA=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$STAGING_DIR/opencv/lib \
      -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
      $OPENVINO_HOME/src/bindings/python && \
make -j$BUILD_JOBS && \
cmake -DCMAKE_INSTALL_PREFIX=$STAGING_DIR -P cmake_install.cmake && \
cd $DEV_HOME || fail 13 "OpenVINO python bindings build failed. Stopping"

#Open Model Zoo
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
  OMZ_DEMOS_BUILD=$OMZ_HOME/build && \
  mkdir -p $OMZ_DEMOS_BUILD && \
  cd $OMZ_DEMOS_BUILD && \
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_PYTHON=ON \
        -DPYTHON_EXECUTABLE=/usr/local/bin/python3.7m \
        -DPYTHON_INCLUDE_DIR="/opt/python3.7_arm/include/python3.7m" \
        -DPYTHON_LIBRARY="/opt/python3.7_arm/lib" \
        -DCMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS" \
        -DOpenVINO_DIR=$OPENVINO_HOME/build \
        -DOpenCV_DIR=$OPENCV_HOME/build \
        $OMZ_HOME/demos && \
  cmake --build $OMZ_DEMOS_BUILD -- -j$BUILD_JOBS && \
  cd $DEV_HOME || fail 16 "Open Model Zoo build failed. Stopping"
  python3 $OMZ_HOME/ci/prepare-openvino-content.py l $OMZ_DEMOS_BUILD && \
  cp -vr $OMZ_DEMOS_BUILD/dev/. $STAGING_DIR && \
  find $OMZ_DEMOS_BUILD -type d -name "Release" -exec cp -vr {} $STAGING_DIR/extras/open_model_zoo/demos \; || \
  fail 21 "Open Model Zoo package preparation failed. Stopping"
fi

#Package creation
cd $STAGING_DIR && \
tar -czvf ../OV_ARM_package.tar.gz ./* || \
fail 23 "Package creation failed. Nothing more to do"

exit 0
