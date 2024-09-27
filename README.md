Network Anomaly Detection
-------------------------

Detecting attacks on IP network using Node2Vec embedding representation and supervised classification.

# Instructions 

## Setup

----
1. **Download data**
- Download data file from the following link 
 https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848
- Create the directory `data/raw`
- Extract `NF-UNSW-NB15-v2.csv` file to `data/raw` directory

2. **Install requirmeents**
- Recommended : Use conda or virtualenv to create a new environment (optional)
- From the root directory of the project, run the following command to install the depencencies
```bash
  pip install -r pkg/requirements.txt
```
 
 ####  Note : How to setup pyenv
 
   ``pyenv install 3.10.5``

   ``pyenv virtualenv 3.10.5 network_ad_env``
   
   ``cd [PROJECT ROOT DIRECTORY]``

   ``pyenv local network_ad_env``
  
  

## Exploratory Data Analysis (EDA)

- Complete the notebook notebooks/eda.ipynb

- Select relevant features for the model based on Correlation and Domain knowledge


## Data Preprocessing

- Write  a general preprocessing script callled : `netflow_v2_preprocessing.py` to clean the data and save the relevant features in new csv file

- Ensure that you also perform a randomized train/test split of the data in that script


### Autoencoder
 
 - Introduction to Autoencoders model

 - Write a new preprocessing script called `netflow_v2_preprocessing_autoencoder.py` to clean to use the output of the previous script \
and perform OneHotEncoding of the categorical features, Standardization of the numerical features
   
 - Implement the Autoencoder model in the `script/autoencoder.py` script
` 
 
 

 




