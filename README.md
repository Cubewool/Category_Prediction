# Category_Prediction
Implement Convolutional Neural Network (CNN) in Python with tensorflow to predict picture category

The goals:

•	To understand the steps to train/test the model for image classification.

•	Understand architecture of CNN and how to connect each layer together by using TensorFlow.

This project build a Convolutional Network classifier using these packages: Tensorflow, Keras (for CIFAR-10), Numpy, and OpenCV (for reading image). The data set used in this project will be CIFAR-10. The common used classifiers are SVM and Softmax.

First Creat folder "./model"

Use command "python CNNclassify.py train" to train the model, save the model in a folder named “model” after finish the training.

Use command "python CNNclassify.py test xxx.png" to test the model. This command will (i) load your model from your folder “model” in the previous step. And it will read “xxx.png” and predict the output, and (2) visualize the output of first CONV layer of the trained model for each filter (e.g., 32 visualization results), and save the visualization results as “CONV_rslt.png” as shown in Fig. 3.
The testing result would match your image type when the classifier achieves high accuracy.


*Please be aware that the computational cost of CONV layer is very high and the training process may take quite long. 
