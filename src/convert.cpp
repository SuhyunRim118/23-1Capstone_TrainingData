#include <iostream>
#include <fstream>
#include <string>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <termios.h>
#include <cstdlib>
#include <fstream> 
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include "opencv4/opencv2/opencv.hpp"

#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;


using namespace std;
using namespace cv;

int main() {
    int cnt = 0;

    while(1) {
        String copyfile = "/home/suhyun/e2eyolo/train_final/extra_data/1/1-1/data/frame";
        copyfile = copyfile + to_string(cnt) + ".jpg";

        Mat copy = imread(copyfile);
        //imshow("frame", copy);

        resize(copy, copy, Size(320,160));

        String framename = "/home/suhyun/e2eyolo/train_final/extra_data/1/1-1/data/frame";
        framename = framename + to_string(cnt)+".jpg";
            
        imwrite(framename, copy);
        cnt++;
    }
}