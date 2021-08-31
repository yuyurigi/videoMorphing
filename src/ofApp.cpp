#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    video.load("pexels-sunsetoned-7235795.mov");
    video.play();
    bPause = false; //false : ビデオを止める、　true : ビデオを再生
    
    count = 0;
    version = 1;
    
    //opencvイメージの初期化（なくても動く）
    //以下のコードを入れておくとコンソールに[warning]・[notaice]の文が出なくなる
    color1.allocate(video.getWidth(), video.getHeight());
    color2.allocate(video.getWidth(), video.getHeight());
    float decimate = 0.3; //Decimate images to 30%
    W = color1.width;
    H = color1.height;
    w = color1.width*decimate;
    h = color1.height*decimate;
    gray1.allocate(w, h);
    gray2.allocate(w, h);
    imageDecimated1.allocate(w, h);
    imageDecimated2.allocate(w, h);
    planeX.allocate(w, h);
    planeY.allocate(w, h);
    mapX.allocate(w, h); //w and h is size of gray1 image
    mapY.allocate(w, h);
    bigMapX.allocate(W, H);
    bigMapY.allocate(W, H);
    fx.allocate(w, h);
    fy.allocate(w, h);
    weight.allocate(w, h);
    prev_morph.allocate(W, H);
    morph.allocate(W, H);
    
    //create idX, idy
    idX.allocate(w, h);
    idY.allocate(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            idX.getPixelsAsFloats()[ x + w * y ] = x;
            idY.getPixelsAsFloats()[ x + w * y ] = y;
        }
    }
}

//--------------------------------------------------------------
void ofApp::update(){
    video.update();
    
    
    if(video.isFrameNew()){
        if (count < ZURE) {
            //最初のフレームを取得
            prevImage[count].setFromPixels(video.getPixels());
        } else {
            ofImage imageOf1, imageOf2;
            int re = count%ZURE;
            imageOf1 = prevImage[re];
            imageOf2.setFromPixels(video.getPixels());
            color1.setFromPixels(imageOf1);
            color2.setFromPixels(imageOf2);
            
            //High-quality resize
            imageDecimated1.scaleIntoMe(color1, CV_INTER_AREA);
            gray1 = imageDecimated1; //グレースケール
            
            //High-quality resize
            imageDecimated2.scaleIntoMe(color2, CV_INTER_AREA);
            gray2 = imageDecimated2; //グレースケール
            
            Mat img1 = cvarrToMat(gray1.getCvImage(), true);
            Mat img2 = cvarrToMat(gray2.getCvImage(), true);
            Mat flow;
            //オプティカルフローの計算
            calcOpticalFlowFarneback(img1, img2, flow, 0.7, 3, 11, 5, 5, 1.1, 0);
            //flowを別々の画像に分割
            vector<Mat> flowPlanes;
            split(flow, flowPlanes);
            //Copy float planes to ofxCv images flowX and flowY
            IplImage iplX( flowPlanes[0] );
            flowX = &iplX;
            IplImage iplY( flowPlanes[1] );
            flowY = &iplY;
            
            //Flow image
            planeX = flowX;
            planeY = flowY;
            
            morphValue = 1;
            updateMorph(morphValue, version);
            
            if(version == 1){
                prevImage[re] = morph.getPixels();
            } else if (version == 2 || version == 3){
                prevImage[re] = imageOf2;
            }
        }
        
        count++;
    }

}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(255, 255, 255);
    ofSetColor(255, 255, 255);
    if(morph.bAllocated){
        morph.draw(0, 0, ofGetWidth(), ofGetHeight());
    }
}
//--------------------------------------------------------------
//Making image morphing
void ofApp::updateMorph( float morphValue, int morphImageIndex )
{
    //Get pointers to pixels data
    float *flowXPixels = flowX.getPixelsAsFloats();
    float *flowYPixels = flowY.getPixelsAsFloats();
    float *mapXPixels = mapX.getPixelsAsFloats();
    float *mapYPixels = mapY.getPixelsAsFloats();
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = x + w * y; //pixels' index
            mapXPixels[i] = x + flowXPixels[i]*morphValue;
            mapYPixels[i] = y + flowYPixels[i]*morphValue;
        }
    }
    mapX.flagImageChanged(); //ピクセル値が変更されたことを知らせる
    mapY.flagImageChanged();
    
    inverseMapping(mapX, mapY);
    
    //bigMapX and bigMapY have type ofxCvFloatImage
    bigMapX.scaleIntoMe(mapX, CV_INTER_LINEAR);
    bigMapY.scaleIntoMe(mapY, CV_INTER_LINEAR);
    multiplyByScalar(bigMapX, 1.0*W/w);
    multiplyByScalar(bigMapY, 1.0*H/h);
    
    //Do warping
    if(morphImageIndex == 1 || morphImageIndex == 3){
        prev_morph = color1;
    } else if(morphImageIndex == 2){
        if(count == ZURE){
            prev_morph = color1;
        } else {
            prev_morph = morph;
        }
    }
    morph = prev_morph;
    morph.remap(bigMapX.getCvImage(), bigMapY.getCvImage());
    
}
//--------------------------------------------------------------

void ofApp::multiplyByScalar( ofxCvFloatImage &floatImage, float value )
{
    int w2 = floatImage.width;
    int h2 = floatImage.height;
    float *floatPixels = floatImage.getPixelsAsFloats();
    for (int y=0; y<h2; y++) {
        for (int x=0; x<w2; x++) {
            floatPixels[ x + w2 * y ] *= value;
        }
    }
    floatImage.flagImageChanged();
}
//--------------------------------------------------------------

void ofApp::inverseMapping(ofxCvFloatImage &mapX, ofxCvFloatImage &mapY){
    fx.set(0);
    fy.set(0);
    weight.set(0);
    
    float *mapXPixels = mapX.getPixelsAsFloats();
    float *mapYPixels = mapY.getPixelsAsFloats();
    float *fxPixels = fx.getPixelsAsFloats();
    float *fyPixels = fy.getPixelsAsFloats();
    float *weightPixels = weight.getPixelsAsFloats();
    
    for (int y = 0; y < h; y ++) {
        for (int x = 0; x < w; x++) {
            float MX = mapXPixels[ x + w * y ];
            float MY = mapYPixels[ x + w * y ];
            
            int mx0 = int( MX );
            int my0 = int( MY );
            int mx1 = mx0 + 1;
            int my1 = my0 + 1;
            float weightX = MX - mx0;
            float weightY = MY - my0;
            
            mx0 = ofClamp( mx0, 0, w-1 );    //Bound
            my0 = ofClamp( my0, 0, h-1 );
            mx1 = ofClamp( mx1, 0, w-1 );
            my1 = ofClamp( my1, 0, h-1 );
            for (int b=0; b<2; b++) {
                for (int a=0; a<2; a++) {
                    int x1 = ( a == 0 ) ? mx0 : mx1;
                    int y1 = ( b == 0 ) ? my0 : my1;
                    int i1 = x1 + w * y1;
                    float wgh = ( ( a == 0 ) ? ( 1 - weightX ) : weightX )
                        * ( ( b == 0 ) ? ( 1 - weightY ) : weightY );
                    fxPixels[ i1 ] += x * wgh;
                    fyPixels[ i1 ] += y * wgh;
                    weightPixels[ i1 ] += wgh;
                }
            }
        }
    }
    //Compute map for non-zero weighted pixels
    int zeros = 0;        //Count of zeros pixels
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int i = x + w * y;
            float X = fxPixels[ i ];
            float Y = fyPixels[ i ];
            float Weight = weightPixels[ i ];
            if ( Weight > 0 ) {
                X /= Weight;
                Y /= Weight;
            }
            else {
                X = x;
                Y = y;
                zeros++;
            }
            mapXPixels[ i ] = X;
            mapYPixels[ i ] = Y;
            weightPixels[ i ] = Weight;
        }
    }
    //Fill zero-weighted pixels by weighting of near non-zero weighted pixels
    const int rad = 2;
    const int diam = 2 * rad + 1;
    float filter[ diam * diam ];
    float sum = 0;
    for (int b=-rad; b<=rad; b++) {
        for (int a=-rad; a<=rad; a++) {
            float wgh = rad + 1 - max( abs( a ), abs( b ) );
            filter[ a+rad + diam * (b+rad) ] = wgh;
            sum += wgh;
        }
    }
    for (int i=0; i<diam*diam; i++) {
        filter[ i ] /= sum;
    }

    int zeros0 = -1;
    while ( zeros > 0 && (zeros0 == -1 || zeros0 > zeros) ) {
        zeros0 = zeros;
        zeros = 0;
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                int i = x + w * y;
                if (weightPixels[ i ] < 0.0001 ) {
                    float mX = 0;
                    float mY = 0;
                    float mWeight = 0;
                    int x1, y1, i1;
                    float wgh;
                    for (int b = -rad; b<=rad; b++) {
                        for (int a = -rad; a<=rad; a++) {
                            x1 = x + a;
                            y1 = y + b;
                            if ( ofInRange( x1, 0, w-1 ) && ofInRange( y1, 0, h-1 ) ) {
                                i1 = x1 + w * y1;
                                if ( weightPixels[ i1 ] >= 0.0001 ) {
                                    wgh = filter[ a+rad + diam * (b+rad) ] * weightPixels[ i1 ];
                                    mX += mapXPixels[i1] * wgh;
                                    mY += mapYPixels[i1] * wgh;
                                    mWeight += wgh;
                                }
                            }
                        }
                    }
                    if ( mWeight > 0 ) {
                        mapXPixels[ i ] = mX / mWeight;
                        mapYPixels[ i ] = mY / mWeight;
                        weightPixels[ i ] = mWeight;
                    }
                    else {
                        zeros++;
                    }
                }
            }
        }
    }

    mapX.flagImageChanged();
    mapY.flagImageChanged();
}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'S' || key == 's') {
            ofImage myImage;
            myImage.grabScreen(0, 0, ofGetWidth(), ofGetHeight());
            myImage.save(ofGetTimestampString("%Y%m%d%H%M%S")+"##.png");
        }
    if (key == '1') {
        version = 1;
    }
    if (key == '2') {
        version = 2;
    }
    if (key == '3') {
        version = 3;
    }
    if (key == ' ') { //ビデオを止める
        bPause = !bPause;
        video.setPaused(bPause);
    }
    if (key == OF_KEY_UP) { //ビデオのスピードをあげる
        float speed = video.getSpeed();
        speed += 0.1;
        video.setSpeed(speed);
    }
    if (key == OF_KEY_DOWN) { //ビデオのスピードをさげる
        float speed = video.getSpeed();
        speed -= 0.1;
        video.setSpeed(speed);
    }
    if (key == 'R' || key == 'r') { //ビデオのスピードをリセット
        video.setSpeed(1);
    }

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
