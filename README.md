# The Fecundity CNN Classifier Model

## Introduction
Using the classified data from the Classifier Site, we were able to train the first version of the fecundity model! When you classified an image it was placed into 
a class which represented the number of eggs in the image (0, 1, ..., 9). These images are used for the training, testing, and validation of our model.

## Model Specifications
You can see our training code in ```cnn_classifier.py```. The model is a sequential CNN which has three convolutional layers with ReLU activation and Max Pooling,
a flatten layer to convert 2D  feature maps to 1D, and two dense layers, the last one using softmax activation for classification.

After training and testing, with the data, we got an accuracy of 88% and an overall mean square error (MSE) of 0.16824.

## Inference and Evaluation
With the 04-29 caps, we did inference of the model. Here is the model predicting counts of images it has never seen before live, note the Predicted Number of Eggs statement at the bottom, that is the model figuring out the number of eggs in the image:

[inference_demo.webm](https://github.com/user-attachments/assets/7c83e4a6-f190-43d4-9f43-6266448c70b7)

The model was accurate was nearly all the eggs except for the third one, where the correct answer is 3. 

Here, we graph the overall MSE with the MSE of each counted egg class:

![mse_fig](https://github.com/user-attachments/assets/856fe14a-80e0-49cc-a12e-26c730726052)

The model performs decently well across the classes!
With the classified data from the site, we were able to get out the first iteration of the Fecundity model.
