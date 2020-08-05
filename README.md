# Data Mining and Image Classifier 
The aim of this repository is to build deep NN classifier that distinguishes between different BMI types of people. 
The BMI metric is defined as:
<p align="center">
  <img src="http://www.sciweavers.org/tex2img.php?eq=%24BMI%20=%20\operatorname{Height}%2F\operatorname{Weight}^2%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=">
</p>
\Where:
\* x < 18.5 --> Underweight
\* 18.5 <= x < 25 --> Normal
\* 25 <= x < 30 --> Overweight
\* 30 <= x < 35 --> Medium Obesity
\* x >= 35 --> Super Obesity


This repository demonstrates how to mine data from the web using 'Selenium' + Data and features creation. Then, training image classifier by pytorch transfer learning.


### Recommended:
* python=3.6
* selenium=3.141.0
* torch=1.5.1
* torchvision=0.5.0
* sklearn=0.20.1
* PIL=7.2.0

## Mining data & Dataset and feature creation
* [Dataset_Mining.ipynb](https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/Dataset_Mining.ipynb)
* [Dataset_Creation.ipynb](https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/Dataset_Creation.ipynb)

## Training image classifier
* [train_classifier.py](https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/train_classifier.py)
* [predict_classifier.py](https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/predict_classifier.py)

## Data example
* [data_example.csv](https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/data_example.csv)


<p align="center">
  <img src="https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/1.jpg?raw=true">
</p>

