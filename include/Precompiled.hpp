#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <istream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <stack>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

// before OpenCV 2.2.0
#ifdef OPENCV_USE_OLD_HEADERS
    #include "cv.h"
    #include "highgui.h"
#else
    #include <opencv2/core/version.hpp>
    #if CV_MAJOR_VERSION == 2
        #include <opencv2/core/core.hpp>
        #include <opencv2/imgproc/imgproc.hpp>
        #include <opencv2/highgui/highgui.hpp>
    #elif CV_MAJOR_VERSION >= 3
        #include <opencv2/core.hpp>
        #include <opencv2/imgproc.hpp>
        #include <opencv2/highgui.hpp>
    #endif
#endif
