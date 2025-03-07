**The Fecundity CNN Classifier Model**

Using the classified data from the Classifier Site, we were able to train the first version of the fecundity model! When you classified an image it was placed into 
a class which represented the number of eggs in the image (0, 1, ..., 9). These images are used for the training, testing, and validation of our model.

You can see our training code in ```cnn_classifier.py```. The model is a sequential CNN which has three convolutional layers with ReLU activation and Max Pooling,
a flatten layer to convert 2D  feature maps to 1D, and two dense layers, the last one using softmax activation for classification.

After training and testing, with the data, we got an accuracy of 88% and an overall mean square error (MSE) of 0.16824.

With the 04-29 caps, we did inference of the model. Here is the model predicting counts of images it has never seen before live:

Here, we graph the overall MSE with the MSE of each counted egg class:

![mse_fig](https://github.com/user-attachments/assets/856fe14a-80e0-49cc-a12e-26c730726052)

The model performs decently well across the classes!