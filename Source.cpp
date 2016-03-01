#include <iostream>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

//using namespace std;
using namespace cv;

	// this function draws an outline in the shape of a rectangle where is detects a face
void draw(std::vector<Rect> faces, Mat imageFrame)
{
	for (int i = 0; i < faces.size(); i++)
	{
		Point p1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point p2(faces[i].x, faces[i].y);

		rectangle(imageFrame, p1, p2, cvScalar(0, 255, 0, 0), 1, 8, 0);
	}
}

int		main(int argc, const char** argv)
{
		// create cascade for face reco
	CascadeClassifier face_reco, face_reco_profile;

	face_reco.load("haarcascade_frontalface_alt.xml"); // loading haarcascade library
	face_reco_profile.load("haarcascade_profileface.xml");

	VideoCapture cam;		// setup video capturing device (a.k.a webcam)
	cam.open(1);			// link it to the device [0 = default cam] (USBcam is default 'cause I disabled the onbord one IRRELEVANT!)
	if (!cam.isOpened())	// check if we succeeded
	{
		std:: cout << "Couldn't load the video cam!!" << std:: endl;
		waitKey(1);
		return -1;
	}

		// images used in the process
	Mat frame, grayFrame, flippedFrame, flippedGrayFrame;

	namedWindow("display", WINDOW_AUTOSIZE); // window to display the results

	bool active = true;
	while (active)        // starting infinit loop
	{
		cam >> frame; // put captured-image frame in frame

		cvtColor(frame, grayFrame, CV_BGR2GRAY); // convert to gray and equalize
		equalizeHist(grayFrame, grayFrame);

			// create an array to store the faces found
		std:: vector<Rect> frontFaces, facesLeftProfile, facesRightProfile;
		
//~~~~~~~~~~~~~~~~~~~~~ FRONTAL FACES ~~~~~~~~~~~~~~~~~~
			// find and store the frontal faces
		face_reco.detectMultiScale(grayFrame, frontFaces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			// draw an outline for the frontal faces
		draw(frontFaces, frame);
		
//~~~~~~~~~~~~~~~~~~~~~ PROFILE FACES ~~~~~~~~~~~~~~~~~~~~~
//left side

			// find and store the profile faces
		face_reco_profile.detectMultiScale(grayFrame, facesLeftProfile, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			// draw an outline for the profile faces
		draw(facesLeftProfile, frame);

//right side

		flip(grayFrame, flippedGrayFrame, 1);//flip the image so it will be the left side
		flip(frame, flippedFrame, 1);
		
			// find and store the profile faces
		face_reco_profile.detectMultiScale(flippedGrayFrame, facesRightProfile, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			// draw an outline for the profile faces
		draw(facesRightProfile, flippedFrame);

		flip(flippedFrame, frame, 1);//i flip it back so the normal image will be shown not the mirrored one
//----------------

		imshow("display", frame); // display the result

		if (waitKey(1) >= 0)  // break the loop
			active = false;
	}

	return 0;
}