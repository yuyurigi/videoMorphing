#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
using namespace cv; //OpenCVのnamespaceを使うコンパイラを指示します

static const int ZURE = 50; //フレーム差

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
    ofxCvColorImage color1, color2; //First and second oriiginal images
    ofxCvGrayscaleImage gray1, gray2; //Decimated grayscaled images
    ofxCvColorImage imageDecimated1, imageDecimated2;
    ofxCvFloatImage flowX, flowY; //Resulted optical flow in x and y axes
    
    int w, h; //Decimated size of input images
    int W, H;
    
    ofxCvGrayscaleImage planeX, planeY;
    
    ofxCvFloatImage idX, idY; //idX(x,y) = x, idY(x,y) = y
    ofxCvFloatImage mapX, mapY;
    ofxCvFloatImage bigMapX, bigMapY;
    
    ofxCvFloatImage fx, fy;
    ofxCvFloatImage weight;
    
    float morphValue; //[0, 1]
    ofxCvColorImage morph, prev_morph; //Resulted morphed image
    
    //Inverting the mapping (mapX, mapY), with antialiasing
    void inverseMapping(ofxCvFloatImage &mapX, ofxCvFloatImage &mapY);
    
    //Making image morphing
    void updateMorph(float morphValue, int morphImageIndex);
    
    void multiplyByScalar( ofxCvFloatImage &floatImage, float value );
    
    ofVideoPlayer video;
    int count, version;
    bool bPause;
    
    ofImage prevImage[ZURE];
};
