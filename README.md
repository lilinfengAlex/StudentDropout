# StudentDropoutPredictionSystem
CollegeStudentDropoutPrediction

System Requirement:
Python 3
Numpy
Pandas
sklearn
Tensorflow
Keras
Django
matplotlib

Project Introduction:
This project processes Excel data, selects features of higher importance, trains neural network models, and predicts the likelihood of college students dropping out.

Files Introduction：
feature_select.py--Reduce the count of features from over 50 to 10
dropout.py--Normalize data and train ANN model
model.h5--Saved model

Website:
This project uses the Django framework to build web pages. After the web page obtains the input data, the saved model is called, and the output result is returned to the web page.
