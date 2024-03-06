# KNN-MNIST-Fashion

## Introduction
In this notebook, I aimed to examine the KNN algorithm and illustrate the importance of selecting the appropriate number of neighbors. To implement this method, I opted for the Fashion-MNIST image dataset. 

Fashion-MNIST is an assortment of fashion product images designed to present a more complex challenge for machine learning classification than the original MNIST digit dataset. This dataset is a standard tool for training and testing various machine learning models.

To demonstrate the importance of selecting the right number of neighbors, I initially explored model overfitting. Next, I decided to plot a graph illustrating how the accuracy varies on both the training and testing sets as a function of the number of neighbors. 
This method demonstrates that when the number of neighbors is too small, the accuracy of the training set becomes excessively high, indicating that the algorithm is essentially memorizing the dataset. 
Alternatively, when the number of neighbors is too large, the accuracy of both the training and testing sets begins to decline, suggesting that the model implemented by the algorithm is too simplistic or underfitted.
This first method allowed for the identification of the optimal number of neighbors. 

Additionally, I employed a second method involving hyperparameter tuning through GridSearchCV on the training set. This method provides the optimal number of neighbors by conducting a grid search using a 5-fold cross-validation, as specified with cv=5. Both methods led me to determine that the optimal number of neighbors is 6.

## Composition of the Dataset
The MNIST Fashion dataset is composed of 70,000 images of fashion products from 10 categories with 7,000 images per category. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. 

## Dataset Details
Every image in the dataset is a 28x28 pixel grayscale image, amounting to 784 pixels per image. An example of the dataset images for each category is shown below.
 

![image](https://github.com/angelicamiotti/KNN-MNIST-Fashion/assets/8940755/243c7378-5df0-4db1-85ad-1cd3a042647a)


The Fashion-MNIST dataset can be found within keras.datasets.fashion_mnist
It consists of two sets of numpy arrays: the training set and the test set. The training set includes an array X_training of shape (60000, 28, 28), where each 28x28 array corresponds to the pixel values of an individual image, and y_training, an array of labels with shape (60000,). Similarly, the test set contains an array X_testing with dimensions (10000, 28, 28) and the labels array y_testing with dimensions (10000,). Pixel values range from 0 to 255, indicating the lightness or darkness of each pixel.

## Labeling System
The arrays y_training and y_testing store the labels, which are integers from 0 to 9. Each integer corresponds to a different category of fashion item as described below.

 ![image](https://github.com/angelicamiotti/KNN-MNIST-Fashion/assets/8940755/8330ec5a-f6cb-465e-8394-f59e1d0f9414)

Source: https://keras.io/api/datasets/fashion_mnist/ 
## Implementation

### Data Preprocessing

The Fashion-MNIST dataset was preprocessed to facilitate effective model training:

- Reshaping: The images, originally in a 3D array (60000, 28, 28), were reshaped into a 2D array (60000, 784) to convert each image into a flat vector.
- Standardization: The pixel values were standardized using StandardScaler, which adjusts the distribution of each feature to have a mean of zero and a standard deviation of one. 

### Model Training and Evaluation
- K-Nearest Neighbors (KNN): The KNN algorithm was employed as the classifier. We experimented with different values of 'k' to find the optimal number of neighbors.
- Sampling: To expedite computational time, a subset of the dataset was sampled for model training and validation.
- Grid Search: A grid search was conducted to tune hyperparameters and validate the model's performance across different 'k' values.
- Performance Metrics: The accuracy was calculated for both the training and testing sets to evaluate model generalizability.
- Hyperparameter Tuning with Cross-Validation: GridSearchCV was utilized to perform exhaustive hyperparameter tuning across a range of 'k' values on the training set. The cross-validation approach within GridSearchCV ensures that each hyperparameter setting is thoroughly evaluated on different subsets of the training set, mitigating the risk of overfitting and providing a robust estimate of the model's expected performance on unseen data.


## Results

In the results, k=6 provided a balance between the accuracy for the training set and the generalization to the testing set, suggesting it as the optimal number of neighbors for this particular KNN implementation on the Fashion-MNIST dataset.

In addition to evaluating performance metrics, GridSearchCV was employed to systematically determine the best number of neighbors for our KNN model. This comprehensive hyperparameter tuning process involved cross-validation to ensure the robustness and reliability of our findings. The results from GridSearchCV corroborated our initial analysis, confirming that k = 6 indeed offers the most balanced performance. 




## Note
For this task, I created a new environment with the following versions:

- Numpy 1.19.5
- Tensorflow 2.3
- Pandas 1.1.5
- Python 3.8
- Matplotlib 3.6.2

## License
The copyright for Fashion-MNIST is held by Zalando SE. Fashion-MNIST is licensed under the MIT license.

