/*
Example of doing object recognition with
Histogram of Oriented Gradients and Libsvm

Hand gesture training images from
Sébastien Marcel - Hand Posture and Gesture Datasets
http://www.idiap.ch/resource/gestures/

Depends on OpenCV for Processing:
https://github.com/atduskgreg/opencv-processing

and Processing-SVM, a libsvm wrapper:
https://github.com/atduskgreg/Processing-SVM
*/

// import processing video library and OpenCV for Processing
import processing.video.*;
import gab.opencv.*;

// import key OpenCV classes
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.core.Core;
// java array helper
import java.util.Arrays;

// Processing object used for video capture
Capture video;

// Classifier will use Libsvm (could try other classifiers: AdaBoost, KNN, RandomForest)
Libsvm classifier;
OpenCV opencv;

// Settings for display size
// and object detection window
int w = 640;
int h = 480;
int rectW = 150;
int rectH = 150;

PImage testImage;
String[] labelNames = {"A", "B", "C", "V", "Five", "Point"};

void setup() {
  // set the sketch size
  size(w + 50, h/2 + 50);
  
  // Initialize OpenCV to expect 50x50 images
  opencv = new OpenCV(this, 50, 50);
  // Initialize the classifier
  classifier = new Libsvm(this);
  // set number of features based on gradient size from HoG settings (see gradientsForImage())
  classifier.setNumFeatures(1728);

  // Initialize live video capture at 1/2 size
  video = new Capture(this, w/2, h/2);
  video.start();

  // Load the training images from the data/train folder
  java.io.File folder = new java.io.File(dataPath("train"));
  String[] trainingFilenames = folder.list();
  for (int i = 0; i < trainingFilenames.length; i++) {
    print(trainingFilenames[i]);
    // Determine label for each image by parsing filename, i.e. "A-blah.jpg"
    String gestureLabel = split(trainingFilenames[i], '-')[0];
    // load image into Processing
    PImage img = loadImage(dataPath("train/" + trainingFilenames[i]));
    
    // set integer label based on parsed filename
    int label = 0;
    if (gestureLabel.equals("A")) {
      label = 0;
    }
    if (gestureLabel.equals("B")) {
      label = 1;
    }
    if (gestureLabel.equals("C")) {
      label = 2;
    }
    if (gestureLabel.equals("V")) {
      label = 3;
    }
    if (gestureLabel.equals("Five")) {
      label = 4;
    }
    if (gestureLabel.equals("Point")) {
      label = 5;
    }
    println(" " + label);
    
    // Calculate the Histogram of Oriented Gradients for this image
    float[] hog = gradientsForImage(img);
    // Create a new Sample with the HoG as the feature vector and the appropriate label
    Sample sample = new Sample(hog, label);
    // Add the sampel to the classifier's training set
    classifier.addTrainingSample(sample);
  }
  // train the classifier
  classifier.train();
  
  // init scratch image for faster resizing  
  testImage = createImage(w, h, RGB);
}

void draw() {
  // clear the background to black
  background(0);
  // display the input video
  image(video, 0, 0);

  // draw a red rectangle in the center of the video
  // the pixels inside this rectangle will be used as the target for recognition
  noFill();
  stroke(255, 0, 0);
  strokeWeight(5);
  rect(video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH);
  
  // copy the pixels from the center of the screen into the test image
  testImage.copy(video, video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH, 0, 0, 50, 50);
  testImage.updatePixels();
  image(testImage, 0, video.height);

  // create an array to store the confidences for each class (libsvm only)
  double[] confidences = new double[6];
  // create a sample from the HoG of the current image:
  Sample testSample = new Sample(gradientsForImage(testImage));
  // use the classifier to predict the class.
  // it returns the class label and populates the confidence array
  double prediction = classifier.predict(testSample, confidences);
  
  // display the prediction and the confidence values 
  // for each of the classes
  fill(255);
  text(labelNames[(int)prediction], video.width + 10, 25);
  String confString = "";
  for (int i = 0; i < confidences.length; i++) {
    confString += "[" + labelNames[i] + "] " + confidences[i] + "\n";
  }
  text(confString, video.width + 10, 45);
}

// This function uses OpenCV to calculate the
// Histogram of Oriented Gradients for an image
// The settings here determine the size of the feature vector
// This function must be applied consistently to both training
// and test images to get a meaningful result
// NB. that it resizes all images to an identical size before
// calculating the HoG
float[] gradientsForImage(PImage img) {
  // resize the images to a consistent size:
  img.resize(50,50);
  // load resized image into OpenCV
  opencv.loadImage(img);

  // settings for HoG calculation
  Size winSize = new Size(40, 24);
  Size blockSize = new Size(8, 8);
  Size blockStride = new Size(16, 16);
  Size cellSize = new Size(2, 2);
  int nBins = 9;
  Size winStride = new Size(16, 16);
  Size padding = new Size(0, 0);

  HOGDescriptor descriptor = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);

  MatOfFloat descriptors = new MatOfFloat();
  MatOfPoint locations = new MatOfPoint();
  descriptor.compute(opencv.getGray(), descriptors, winStride, padding, locations);

  return descriptors.toArray();
}

// Required callback function for Processing video capture
void captureEvent(Capture c) {
  c.read();
}

