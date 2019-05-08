# Image-Colorization

Automatic colorization of gray-scale images using deep learning and OpenCV is a technique to colorize gray-scale images without involvement of a human. Conventional techniques used for colorizing images need human intervention.The proposed technique uses deep convolutional neural networks and has a number of advantages. The technique will reduce manual work, speed up the process and improve the accuracy. 

Procedure used :-
The entire (simplified) process can be summarized as
Convert all training images from the RGB color space to the Lab color space.
1.	Use the L channel as the input to the network and train the network to predict the abchannels.
2.	Combine the input L channel with the predicted ab channels.
3.	Convert the Lab image back to RGB.

Follow the steps :-
1.	Download the Caffe Model from this sharable link : https://drive.google.com/open?id=1-4xN5qzlNjIr7t2xUvxmGnbbNzNj3LDX.
2.	Put it in this folder (Image Colorization) that you will download or clone.
3.	Run the cmd command in the folder and use the argument --input to attach the images you want to color.
4.	Run :- python Image_Color.py --input (name of the image with .jpg/.jpeg/.png).
5.	And done !!! You’ll get your colorized image saved in the folder itself.

	
