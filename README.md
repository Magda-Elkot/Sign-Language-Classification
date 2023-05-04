# Sign-Language-Classification
  This project aims to develop a deep learning model that can accurately classify sign language gestures from images. 
  The model is designed using a convolutional neural  network (CNN) architecture, and it is trained on the Sign Language MNIST dataset.
  
## Dataset
  The dataset used for this project can be found at : https://www.kaggle.com/datasets/datamunge/sign-language-mnist
  
  The Sign Language MNIST dataset contains 27,455 grayscale images of sign language gestures, each with a resolution of 28x28 pixels. 
  The images are labeled according to the corresponding sign language gesture, with 24 unique labels in total.

## Data Preprocessing
  Before training the model, the dataset is preprocessed using the following steps:

    1. Loading the dataset using the load_data() function from the tensorflow.keras.datasets module.
    2. Checking for missing values and correlations between the features and target variable.
    3. Encoding the labels using LabelBinarizer and normalizing the pixel values of the images between 0 and 1.
    4. Applying data augmentation techniques using ImageDataGenerator to increase the size of the training dataset and improve 
     the model's generalization performance.
 
## Model Architecture
  The model is designed using a CNN architecture consisting of three convolutional layers with increasing filter sizes, max pooling layers, 
  and a dense layer with a dropout rate of 0.25. 
  The output layer contains 24 units (corresponding to the 24 sign language gestures) with a softmax activation function, 
  which predicts the class probabilities.

## Model Training
  The model is compiled using the Adam optimizer and categorical cross-entropy loss function, and it is trained for 
  125 epochs on the augmented training dataset. 
  The model is evaluated on the test dataset using accuracy as the evaluation metric.

## Results
  The trained model achieves an accuracy of 0.9666 on the test dataset and a validation accuracy of 0.9902, indicating that 
  it can accurately classify sign language gestures from images. 
  The model's performance is further improved by applying data augmentation techniques during training.

## Conclusion
  This project provides valuable experience in deep learning model design, data preprocessing, and data augmentation techniques. 
  By developing a deep learning model that can accurately classify sign language gestures, we can potentially improve accessibility 
  for individuals who use sign language as their primary mode of communication.

## Contact
  For more information about this project, please contact Magda El-Romany at magdaalromany@gmail.com .

## Acknowledgements
  This project was completed as part of the internship program at SYNC INTERN'S.
