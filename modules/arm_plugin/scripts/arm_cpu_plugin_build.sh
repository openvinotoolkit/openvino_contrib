#!/bin/sh

set -x

BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
UPDATE_SOURCES=${UPDATE_SOURCES:-clean}
WITH_OPENCV=${WITH_OPENCV:-ON}
WITH_OMZ_DEMO=${WITH_OMZ_DEMO:-ON}

DEV_HOME="$(pwd)"
ONETBB_HOME="$DEV_HOME/oneTBB"
OPENCV_HOME="$DEV_HOME/opencv"
OPENVINO_HOME="$DEV_HOME/openvino"
OPENVINO_CONTRIB="$DEV_HOME/openvino_contrib"
ARM_PLUGIN_HOME="$OPENVINO_CONTRIB/modules/arm_plugin"
OMZ_HOME="$DEV_HOME/open_model_zoo"
STAGING_DIR="$DEV_HOME/armcpu_package"

ONETBB_BUILD="$DEV_HOME/oneTBB/build"
OPENCV_BUILD="$DEV_HOME/opencv/build"
OPENVINO_BUILD="$DEV_HOME/openvino/build"
OMZ_BUILD="$DEV_HOME/open_model_zoo/build"

fail()
{
    if [ $# -lt 2 ]; then
      echo "Script internal error"
      exit 31
    fi
    retval=$1
    shift
    echo "$@"
    exit "$retval"
}

cloneSrcTree()
{
    DESTDIR=$1
    shift
    SRCURL=$1
    shift
    while [ $# -gt 0 ]; do
        git clone --recurse-submodules --shallow-submodules --depth 1 --branch="$1" "$SRCURL" "$DESTDIR" && return 0
        shift
    done
    return 1
}

checkSrcTree()
{
    [ $# -lt 3 ] && fail

    if ! [ -d "$1" ]; then
        echo "Unable to detect $1"
        echo "Cloning $2..."
        cloneSrcTree "$@" || fail 3 "Failed to clone $2. Stopping"
    else
        echo "Detected $1"
        echo "Considering it as source directory"
        if [ "$UPDATE_SOURCES" = "reload" ]; then
            echo "Source reloading requested"
            echo "Removing existing sources..."
            rm -rf "$1" || fail 1 "Failed to remove. Stopping"
            echo "Cloning $2..."
            cloneSrcTree "$@" || fail 3 "Failed to clone $2. Stopping"
        elif [ -d "$1/build" ]; then
            echo "Build directory detected at $1"
            if [ "$UPDATE_SOURCES" = "clean" ]; then
                echo "Cleanup of previous build requested"
                echo "Removing previous build results..."
                rm -rf "$1/build" || fail 2 "Failed to cleanup. Stopping"
            fi
        fi
    fi
    return 0
}


# Prepare sources
checkSrcTree "$ONETBB_HOME" https://github.com/oneapi-src/oneTBB.git master
if [ "$WITH_OPENCV" = "ON" ]; then
    checkSrcTree "$OPENCV_HOME" https://github.com/opencv/opencv.git 4.x
fi
checkSrcTree "$OPENVINO_HOME" https://github.com/openvinotoolkit/openvino.git master
checkSrcTree "$OPENVINO_CONTRIB" https://github.com/openvinotoolkit/openvino_contrib.git master
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
    checkSrcTree "$OMZ_HOME" https://github.com/openvinotoolkit/open_model_zoo.git master
fi

# python variables
python_executable="$(which python3)"
python_library_name=$($python_executable -c "import sysconfig; print(str(sysconfig.get_config_var(\"LDLIBRARY\")))")
python_library_dir=$($python_executable -c "import sysconfig; print(str(sysconfig.get_config_var(\"LIBDIR\")))")
python_library="$python_library_dir/$python_library_name"
python_inc_dir=$($python_executable -c "import sysconfig; print(str(sysconfig.get_config_var(\"INCLUDEPY\")))")
numpy_inc_dir=$($python_executable -c "import numpy; print(numpy.get_include())")

[ -z "$numpy_inc_dir"] && numpy_inc_dir=/usr/lib/python3/dist-packages/numpy/core/include/
! [ -z "$TOOLCHAIN_DEFS" ] && export CMAKE_TOOLCHAIN_FILE="$OPENVINO_HOME/cmake/$TOOLCHAIN_DEFS"

# cleanup package destination folder
[ -e "$STAGING_DIR" ] && rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Build oneTBB
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$ONETBB_BUILD/install" \
    -DTBB_TEST=OFF \
    -S "$ONETBB_HOME" \
    -B "$ONETBB_BUILD" && \
cmake --build "$ONETBB_BUILD" --parallel "$BUILD_JOBS" && \
cmake --install "$ONETBB_BUILD" && \
cd "$DEV_HOME" || fail 11 "oneTBB build failed. Stopping"

# export TBB for later usage in OpenCV / OpenVINO
export TBB_DIR="$ONETBB_BUILD/install/lib/cmake/TBB/"

# Build OpenCV
if [ "$WITH_OPENCV" = "ON" ]; then
    cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DBUILD_LIST=imgcodecs,videoio,highgui,gapi,python3 \
        -DCMAKE_INSTALL_PREFIX="$STAGING_DIR/extras/opencv" \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DOPENCV_SKIP_PYTHON_LOADER=OFF \
        -DPYTHON3_LIMITED_API=ON \
        -DPYTHON3_EXECUTABLE="$python_executable" \
        -DPYTHON3_INCLUDE_PATH="$python_inc_dir" \
        -DPYTHON3_LIBRARIES="$python_library" \
        -DPYTHON3_NUMPY_INCLUDE_DIRS="$numpy_inc_dir" \
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
        -DWITH_GTK_2_X=OFF \
        -DOPENCV_ENABLE_PKG_CONFIG=ON \
        -S "$OPENCV_HOME" \
        -B "$OPENCV_BUILD" && \
    cmake --build "$OPENCV_BUILD" --parallel "$BUILD_JOBS" && \
    cmake --install "$OPENCV_BUILD" && \
    mkdir -pv "$STAGING_DIR/python/python3" && cp -r "$STAGING_DIR/extras/opencv/python/cv2" "$STAGING_DIR/python/python3" && \
    cd "$DEV_HOME" || fail 11 "OpenCV build failed. Stopping"

    # export OpenCV for later usage in OpenVINO
    export OpenCV_DIR="$STAGING_DIR/extras/opencv/cmake"
fi

# Build OpenVINO
cmake -DENABLE_CPPLINT=OFF \
      -DENABLE_PYTHON=OFF \
      -DENABLE_OV_TF_FRONTEND=OFF \
      -DENABLE_OV_PADDLE_FRONTEND=OFF \
      -DENABLE_OV_ONNX_FRONTEND=OFF \
      -DENABLE_TEMPLATE=OFF \
      -DENABLE_TESTS=OFF \
      -DENABLE_GAPI_TESTS=OFF \
      -DENABLE_DATA=OFF \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DARM_COMPUTE_SCONS_JOBS="$BUILD_JOBS" \
      -DIE_EXTRA_MODULES="$ARM_PLUGIN_HOME" \
      -S "$OPENVINO_HOME" \
      -B "$OPENVINO_BUILD" && \
cmake --build "$OPENVINO_BUILD" --parallel "$BUILD_JOBS" && \
cmake --install "$OPENVINO_BUILD" --prefix "$STAGING_DIR" && \
cd "$DEV_HOME" || fail 12 "OpenVINO build failed. Stopping"

# OpenVINO python
[ "$UPDATE_SOURCES" = "clean" ] && [ -e "$OPENVINO_BUILD/pbuild" ] && rm -rf "$OPENVINO_BUILD/pbuild"
cmake -DOpenVINODeveloperPackage_DIR="$OPENVINO_BUILD" \
      -DCMAKE_INSTALL_PREFIX="$STAGING_DIR" \
      -DENABLE_PYTHON=ON \
      -DENABLE_WHEEL=ON \
      -S "$OPENVINO_HOME/src/bindings/python" \
      -B "$OPENVINO_BUILD/pbuild" && \
cmake --build "$OPENVINO_BUILD/pbuild" --parallel "$BUILD_JOBS" && \
cmake --install "$OPENVINO_BUILD/pbuild" && \
cd "$DEV_HOME" || fail 13 "OpenVINO python bindings build failed. Stopping"

# Open Model Zoo
if [ "$WITH_OMZ_DEMO" = "ON" ]; then
  cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DENABLE_PYTHON=ON \
        -DPYTHON_EXECUTABLE="$python_executable" \
        -DPYTHON_INCLUDE_DIR="$python_inc_dir" \
        -DPYTHON_LIBRARY="$python_library" \
        -DOpenVINO_DIR="$OPENVINO_HOME/build" \
        -DOpenCV_DIR="$OPENCV_HOME/build" \
        -S "$OMZ_HOME/demos" \
        -B "$OMZ_BUILD" && \
  cmake --build "$OMZ_BUILD" --parallel "$BUILD_JOBS" && \
  cd "$DEV_HOME" || fail 16 "Open Model Zoo build failed. Stopping"
  python3 "$OMZ_HOME/ci/prepare-openvino-content.py" l "$OMZ_BUILD" && \
  cp -vr "$OMZ_BUILD/dev/." "$STAGING_DIR" && \
  find "$OMZ_BUILD" -type d -name "Release" -exec cp -vr {} "$STAGING_DIR/extras/open_model_zoo/demos" \; || \
  fail 21 "Open Model Zoo package preparation failed. Stopping"
fi

# Package creation
cd "$STAGING_DIR" && \
tar -czvf ../OV_ARM_package.tar.gz ./* || \
fail 23 "Package creation failed. Nothing more to do"

exit 0
