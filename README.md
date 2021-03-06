# BMI Image Classifier & Data Mining
The aim of this repository is to build deep NN classifier that distinguishes between different BMI types of people. 
<br>The BMI metric is defined as:
<p align="left">
  <img src="http://www.sciweavers.org/tex2img.php?eq=%24BMI%20=%20\operatorname{Height}%2F\operatorname{Weight}^2%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=">
</p>
Where:
<br>x < 18.5 --> Underweight
<br>18.5 <= x < 25 --> Normal
<br>25 <= x < 30 --> Overweight
<br>30 <= x < 35 --> Medium Obesity
<br>x >= 35 --> Super Obesity


<br>This repository demonstrates how to mine data from the web using 'Selenium' + Data and features creation. Then, training image classifier by pytorch transfer learning.


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
* If you're interested in the full dataset, please write me a mail:
`Maintainer` [Jonatan Koren](https://jonykoren.github.io/) (jonykoren@gmail.com)

<p align="center">
  <img src="https://github.com/jonykoren/Data_Mining_and_Image_Classifier/blob/master/1.jpg?raw=true">
</p>

