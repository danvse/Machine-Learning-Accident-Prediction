# Machine-Learning-Accident-Prediction

How to set up to run the file
Download conda and anaconda
open conda terminal and create an enviroment
conda create --name <my-env>

install these package files:
# Base packages
conda install pandas numpy matplotlib seaborn scikit-learn -y

# statsmodels and pmdarima for time series
conda install statsmodels -y
conda install -c conda-forge pmdarima -y

# TensorFlow
conda install -c conda-forge tensorflow -y

# KaggleHub
conda install pip -y
pip install kagglehub

# Optional
conda install notebook -y
conda install -c conda-forge ipywidgets -y

