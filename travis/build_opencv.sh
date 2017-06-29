PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

echo "chekcing dir $OPENCV_INSTALL_PATH/lib"
if [[ -d "$OPENCV_INSTALL_PATH/lib" ]]; then
    echo "cached opencv libs found, skip building"
    cp $OPENCV_INSTALL_PATH/lib/cv2.so $PYTHON_PACKAGES_PATH
    exit 0
else
    echo "no cached opencv libs, start building"
fi
mkdir -p $HOME/opencv_src && cd $HOME/opencv_src
wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz
tar -xf $OPENCV_VERSION.tar.gz --strip-components=1
mkdir build && cd build

COMMON_OPTIONS="-DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_PATH -DBUILD_opencv_java=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DWITH_IPP=OFF -DBUILD_DOCS=OFF"
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then 
    cmake $COMMON_OPTIONS \
    -DBUILD_opencv_python3=OFF \
    -DPYTHON2_EXECUTABLE=$(python -c "import sys; print(sys.executable)") \
    -DPYTHON2_VERSION_STRING=$(python -c "from platform import python_version; print(python_version())") \
    -DPYTHON2_INCLUDE_PATH=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON2_PACKAGES_PATH=$PYTHON_PACKAGES_PATH \
    -DPYTHON2_NUMPY_INCLUDE_DIRS=$(python -c "import os; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))") \
    -DPYTHON2_NUMPY_VERSION=$(python -c "import numpy; print(numpy.version.version)") \
    ..
else 
    cmake $COMMON_OPTIONS \
    -DBUILD_opencv_python2=OFF \
    -DPYTHON3_EXECUTABLE=$(python -c "import sys; print(sys.executable)") \
    -DPYTHON3_VERSION_STRING=$(python -c "from platform import python_version; print(python_version())") \
    -DPYTHON3_INCLUDE_PATH=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON3_PACKAGES_PATH=$PYTHON_PACKAGES_PATH \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import os; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))") \
    -DPYTHON3_NUMPY_VERSION=$(python -c "import numpy; print(numpy.version.version)") \
    ..
fi
make -j
make install
cp $PYTHON_PACKAGES_PATH/cv2*.so $OPENCV_INSTALL_PATH/lib/cv2.so
cd $TRAVIS_BUILD_DIR