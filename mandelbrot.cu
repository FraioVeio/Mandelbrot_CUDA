#include <iostream>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;

#define RES4K
//#define CENTERSELECT
//#define SHOWWINDOW

#ifdef RESFHD
int resx = 1920;
int resy = 1080;
#else
#ifdef RES4K
int resx = 3840;
int resy = 2160;
#else
int resx = 320;
int resy = 240;
#endif
#endif

double cx = -0.70057617982774456, cy = 0.29002126143480489;
double scale = 1;



__global__ void mandelPixel(int resx, int resy, double *img, double scale, double cnx, double cny, int iterations) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int x = index%resx;
    int y = index/resx;

    if(x < resx && y < resy) {
        x -= resx/2;
        y -= resy/2;
        double cx = 2.0*x/(resx<resy?resx:resy);
        double cy = 2.0*y/(resx<resy?resx:resy);


        cx *= scale;
        cy *= scale;
        cx += cnx;
        cy += cny;

        double zx = 0, zy = 0;
        int i;
        for(i=0;i<iterations;i++) {
            double zxn = zx*zx-zy*zy;
            double zyn = 2*zx*zy;

            zx = zxn + cx;
            zy = zyn + cy;


            if(sqrt(zx*zx+zy*zy)>2)
                break;
        }

        img[index] = log((double)i*0.1+1);
    }
}

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
     if  (event == EVENT_LBUTTONDOWN)
     {
         int nx = x - resx/2;
         int ny = y - resy/2;

         double cnx = cx, cny = cy;
         cx = 2.0*nx/(resx<resy?resx:resy);
         cy = 2.0*ny/(resx<resy?resx:resy);
         cx *= scale;
         cy *= scale;
         cx += cnx;
         cy += cny;
     } else if  (event == EVENT_RBUTTONDOWN) {

     } else if  (event == EVENT_MBUTTONDOWN) {

     } else if (event == EVENT_MOUSEMOVE) {

     }
}

int main() {
    std::cout.precision(17);

    double *img_G, *img;
    cudaMalloc((void**)&img_G, resx*resy * sizeof(double));
    img = (double*) calloc(resx * resy, sizeof(double));

    int threadsPerBlock = 1024;


    Size S(resx, resy);
    VideoWriter video("out.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 60, S, true);
    if (video.isOpened() == false)
    {
        std::cout << "Cannot save the video to a file" << std::endl;
        return -1;
    }

    #ifdef SHOWWINDOW
    namedWindow("Mandelbrot", 1);
    setMouseCallback("Mandelbrot", mouseCallback, 0);
    #endif

    int frame = 0;
    int maxframe = 760*4;
    while(frame < maxframe) {
        frame ++;
        scale = pow(10.0, -frame*0.005);

        int iterations = (int)((1+log(1/scale))*1000);
        //int iterations = 1000;
        std::cout << frame << "/" << maxframe << ": " << scale << ", " << iterations << std::endl;
        mandelPixel<<<resx*resy/threadsPerBlock+1, threadsPerBlock>>>(resx, resy, img_G, scale, cx, cy, iterations);

        cudaDeviceSynchronize();

        cudaMemcpy(img, img_G, resx*resy * sizeof(double), cudaMemcpyDeviceToHost);


        double max = -std::numeric_limits<double>::infinity();
        double min = std::numeric_limits<double>::infinity();

        for(int i=0;i<resx*resy;i++) {
            if(img[i] > max)
                max = img[i];
            if(img[i] < min)
                min = img[i];
        }

        for(int i=0;i<resx*resy;i++) {
            img[i] = 255*((img[i]-min)/(max-min));
        }

        Mat A(resy,resx,CV_64FC1);
        memcpy(A.data, img, resx*resy*sizeof(double));

        Mat A1;
        A.convertTo(A, CV_8UC1);
        applyColorMap(A, A1, COLORMAP_HOT);

        video.write(A1);
        #ifdef SHOWWINDOW
        imshow("Mandelbrot", A1);
        #endif


        #ifdef CENTERSELECT
        int key = waitKey();
        #else
        int key = waitKey(1);
        #endif
        //std::cout << key << std::endl;
        if(key == 27)
            break;

        if(false) {
            if(key == 82) { //Su
                cy += 0.1*scale;
            }

            if(key == 84) { //Giu
                cy -= 0.1*scale;
            }

            if(key == 83) { //Destra
                cx -= 0.1*scale;
            }

            if(key == 81) { //Sinistra
                cx += 0.1*scale;
            }

            if(key == 97) { //Zoom+
                scale -= scale*0.1;
            }

            if(key == 115) { //Zoom-
                scale += scale*0.1;
            }
        }
        #ifdef CENTERSELECT
        break;
        #endif
    }
    video.release();
    std::cout << "cx = " << cx << ", cy = " << cy << std::endl;
    return 0;
}
