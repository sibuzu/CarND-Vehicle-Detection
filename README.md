# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of this project is to write a software pipeline to detect vehicles in a video.  It compared the performances of several approaches, such as HOG, CNN and YOLO.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Build and train a CNN classifier and comparing with HOG method (Extra Works)
* Comparing YOLO detection results with HOG and CNN (Extra Works) 

---
## Histogram of Oriented Gradients (HOG)

All codes could be found in IPython notebook `Vechicle-Detection.ipynb`

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in section `2. HOG features`.  The hog function in the scikit-image module was used.  The images were converted to Gray images and applied with hog in these examples.  However, all hog vector features of the three YCbCr channels were used in the classifier. Some examples were shown in the following.

<table border="1">
<tr>
<td><img src="./examples/HOG_example.png"/></td>
</tr>
</table>

### 2. Explain how you settled on your final choice of HOG parameters.

HOG function ussed 9 orientation bins, 8 pixels per cell and 2 cells per block. Other combinations were tested but no obvious improvement. 

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The codes are in the section `3. Color histogram features`, `4. Spatial Binning of Color features`, `5. Feature Extractions` and `6. Training and testing Support Vector Classifier`.  To improve the performance of the classifier, the color histogram features and spatial binning of color features were also used. Color histogram features used3 32 bins and spatial binning size is (32,32).  The dataset has splitted into 80% training set and 20% testing set.  All features of the training set were trained by a linear SVM classifier, `LinearSVC()`.  At final, SVM classifier achieved 97.47% test accuracy. The accuracy score might be a little different after restarting IPython Kernel.

### 4. Classification using CNN (EXTRA WORKS).

The codes are in the section `7. Classification using CNN`.  A CNN model was built as follows:
<pre>
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 31, 24)        672       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 36)        7812      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 48)        15600     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 80)                649040    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                810       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11        
=================================================================
Total params: 673,945
Trainable params: 673,945
Non-trainable params: 0
_________________________________________________________________
</pre>

The same training and testing dataset in the previous section had been trained and tested.  After 5 epoches trained, the CNN classifier achieved 99.10% test accuracy.  The accuracy score might be a little different after restarting IPython Kernel.


## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The codes are in the section `8. Sliding Window Implementation`.  Search region started from 600 to the right in x-axis and from 380 to 656 in y-axis. The overlap of search windows was 0.7 in both x- and y- directions.  Three scale search windows, (64,64), (96, 96) and (128,128), were applied.  Different scale windows were applied in different regions.  The serached regions were shown in the following image.  The blue boxes were (64,64) search windows, the green ones were (96, 96) and the red ones were (128, 128).

<table border="1">
<tr>
<td><img src="./examples/search_boxes.png"/></td>
</tr>
</table>

### 2. Sliding window search with HOG

The codes are in the section `9. Search Cars by HOG`.  Ultimately I searched on three scale windows, using YCbCr 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  The followings showed some example images.  The cars in the images were detected.  However, there are a lots of false detections, which may be reduced by the heat map algorithm. 

<table border="1">
<tr>
<td><center>Original Image</center></td>
<td><center>Bounding Boxed Detected by HOG</center></td>
</tr>
<tr>
<td><img src="./test_images/test1.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test1.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test2.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test2.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test3.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test3.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test4.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test4.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test5.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test5.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test6.jpg" width="300"/></td>
<td><img src="./output_images/hog_boxes_test6.jpg" width="300"/></td>
</tr>
</table>

### 3. Sliding window search with CNN (Extra Works)

The codes are in the section `10. Search Cars by CNN (Extra Works)`. The same search windows strategy was also applied with CNN classifier we had trained in the previous section.  The following showed some examples.  The CNN classifier seemed more robust with less false detections than the HOG classifier.

<table border="1">
<tr>
<td><center>Original Image</center></td>
<td><center>Bounding Boxed Detected by CNN</center></td>
</tr>
<tr>
<td><img src="./test_images/test1.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test1.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test2.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test2.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test3.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test3.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test4.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test4.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test5.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test5.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./test_images/test6.jpg" width="300"/></td>
<td><img src="./output_images/cnn_boxes_test6.jpg" width="300"/></td>
</tr>
</table>

### 4. HOG with heat map
There were overlapping detections and false positive detections on the images found in the previous sections. A heat-map is built from these detections in order to combine overlapping detections and remove false positives.  Some example results were shown in the following.

<table border="1">
<tr>
<td><center>Bounding Boxes by HOG</center></td>
<td><center>Heat Map</center></td>
<td><center>Results by HOG + Search Windows + Heat Map</center></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test1.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test1.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test1.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test2.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test2.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test2.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test3.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test3.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test3.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test4.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test4.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test4.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test5.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test5.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test5.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test6.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test6.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test6.jpg" width="300"/></td>
</tr>
</table>

### 5. CNN with heat map
A heat-map is built from these detections by CNN classifier.  The results showed CNN approach was better than the HOG approach.

<table border="1">
<tr>
<td><center>Bounding Boxes by HOG</center></td>
<td><center>Heat Map</center></td>
<td><center>Results by CNN + Search Windows + Heat Map</center></td>
</tr>
<tr>
<td><img src="./output_images/cnn_boxes_test1.jpg" width="300"/></td>
<td><img src="./output_images/cnn_heat_test1.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test1.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/cnn_boxes_test2.jpg" width="300"/></td>
<td><img src="./output_images/cnn_heat_test2.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test2.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/cnn_boxes_test3.jpg" width="300"/></td>
<td><img src="./output_images/cnn_heat_test3.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test3.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/cnn_boxes_test4.jpg" width="300"/></td>
<td><img src="./output_images/cnn_heat_test4.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test4.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/cnn_boxes_test5.jpg" width="300"/></td>
<td><img src="./output_images/cnn_heat_test5.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test5.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_boxes_test6.jpg" width="300"/></td>
<td><img src="./output_images/hog_heat_test6.jpg" width="300"/></td>
<td><img src="./output_images/hog_found_test6.jpg" width="300"/></td>
</tr>
</table>

### 5. Compare the performances of the approaches HOG, CNN and YOLO.

We also appplied [YOLOv3](https://pjreddie.com/darknet/yolo/) to detect cars and their positions.  The pretrained weights trained by COCO dataset were used.  A little codes were modified to only detect car class instead of detecting 10 classes.  The following showed the example results of HOG, CNN and YOLO.  Obviously, YOLO performed better then CNN(+Search Windows+Heat Map) and HOG(+Search Windows+Heat Map).  CNN was also better than HOG approach.

<table border="1">
<tr>
<td><center>HOG+SearchWin+HeatMap</center></td>
<td><center>CNN+SearchWin+HeatMap</center></td>
<td><center>YOLO v3</center></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test1.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test1.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test1.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test2.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test2.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test2.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test3.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test3.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test3.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test4.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test4.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test4.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test5.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test5.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test5.jpg" width="300"/></td>
</tr>
<tr>
<td><img src="./output_images/hog_found_test6.jpg" width="300"/></td>
<td><img src="./output_images/cnn_found_test6.jpg" width="300"/></td>
<td><img src="./output_images/yolo_test6.jpg" width="300"/></td>
</tr>
</table>

---

## Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Three videos were made by the three different approaches, which HOG+SearchWindows+HeatMap, CNN+SearchWindows+HeatMap, and YOLO.  Obviously, YOLO had the best performance.

* Video for HOG+SearchWindows+HeatMap is [here](https://youtu.be/H0nqXUTUR7A).
* Video for CNN+SearchWindows+HeatMap is [here](https://youtu.be/4trAJrGXDgI).
* Video for YOLO is [here](https://youtu.be/gwbX6sd2neE).

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I took most of time to tuning HOG approach parameters. It was inefficient and tedious.  Even not to mention YOLO, our second project, Traffice Sign Classifier, were more general and more difficult than car-existed classifier.  I didn't knwo why we used HOG, instead of CNN, to classifed the problems.  The HOG approach was slow, searching windows only seached limited region, couldn't detect other objects more than cars, and worse performance than CNN or YOLO.  I could implement CNN in 20 lines and almost no tuning to get a better performance.   I didn't thought HOG approach could be used in the real world and it might waste time to re-implement this approach.

The is the final project of Term1.  I expected the final project would be a challenge project which used all useful techniques learned or not yet learned in the classes.   But I am depressed.  Everyone loves a challenge project but hates a tedious project.
