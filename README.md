# image_classifier

I used imagemagick to batch transform, grayscale, and (soon) resize images, though I may just do this through tensorflow.

#magick mogrify -path rotated -rotate 180 *.png

#magick mogrify -path path\grayscale -grayscale *.png

#magick mogrify -path path\grayscale -colorspace Gray *.png

These serve to give us as large a dataset as possible (84 for RGB and grayscale, respectively) from which we can use to create training and testing sets.

---

The estimator I have been working on involves building a Keras model (leveraging pre-trained VGG16 model's convolution layers as a base) to do binary classification (Task 1 v. Task 2). 

The model I expect to have a conv base, a flattened layer, and two dense layers (one ReLu and one Sigmoid as activation functions respectively). 

The Keras model will have tensorflow as its backend estimator. Hopefully I can get this to visualize properly using tensorboard. 

Will begin using grayscale images, get a reading. Then create TFRecord in whcih we start creating functions that wrap an integer along with raw bytes for image and label respectively. 

This will finally allow us to create an input function that is basically a feed dictionary and returns tuples. 

We'll use tf.estimator.train_ and _evaluate - these are helpful utility functions that let's us export these estimator models which we will then use to predict images in the test files setting labels to None. 
