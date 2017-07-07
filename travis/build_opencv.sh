set -x
PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

echo "chekcing dir $OPENCV_INSTALL_PATH/lib"
if [[ -f "$OPENCV_INSTALL_PATH/lib/cv2.so" ]]; then
    echo "cached opencv libs found, skip building"
    cp $OPENCV_INSTALL_PATH/lib/cv2.so $PYTHON_PACKAGES_PATH
    exit 0
else
    echo "no cached opencv libs found, try building from source"
fi
mkdir -p $OPENCV_SRC_PATH && cd $OPENCV_SRC_PATH
wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz
tar -xf $OPENCV_VERSION.tar.gz --strip-components=1
mkdir build && cd build

if [[ ("$OPENCV_VERSION" == 3*) && ("$TRAVIS_PYTHON_VERSION" == 2*) ]]; then
    PYV="PYTHON2"
    NP_INC_NAME=${PYV}_NUMPY_INCLUDE_DIRS
elif [[ ("$OPENCV_VERSION" == 3*) && ("$TRAVIS_PYTHON_VERSION" == 3*) ]]; then
    PYV="PYTHON3"
    NP_INC_NAME=${PYV}_NUMPY_INCLUDE_DIRS
else
    PYV="PYTHON"
    NP_INC_NAME=${PYV}_NUMPY_INCLUDE_DIR
fi

cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_PATH \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_DOCS=OFF \
-DBUILD_opencv_java=OFF \
-DWITH_IPP=OFF \
-D${PYV}LIBS_FOUND=ON \
-D${PYV}_EXECUTABLE=$(python -c "import sys; print(sys.executable)") \
-D${PYV}_VERSION_STRING=$(python -c "from platform import python_version; print(python_version())") \
-D${PYV}_INCLUDE_PATH=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D${PYV}_PACKAGES_PATH=$PYTHON_PACKAGES_PATH \
-D${NP_INC_NAME}=$(python -c "import os; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))") \
-D${PYV}_NUMPY_VERSION=$(python -c "import numpy; print(numpy.version.version)") \
..

make -j2
make install
cp $PYTHON_PACKAGES_PATH/cv2*.so $OPENCV_INSTALL_PATH/lib/cv2.so
cd $TRAVIS_BUILD_DIR