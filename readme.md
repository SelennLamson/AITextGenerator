# AI TEXT GENERATOR

## Context

This project has been realized as a part of Natural Langage Processing class teached by Matthias Gallé, *Naverlabs* for the MSc AI  (CentraleSupelec 2019/2020). Project members are: 

- Gaël de Léséleuc
- Alexandre Duval
- Thomas Lamson

## Project description

A detailed description of the project can be found in the *report* folder. Roughly, the idea is to develop an automatic advanced writing tool to help authors in their task. Concretely, the final intended product is a web-service where authors can write text and ask for paragraph automatic generation based on the previous text, the following text, desired size/theme and an optional summary of the content to generate.

## Source code structure

- **json_generation** module handles all the text preprocessing : from raw text to json file containing the novel split by paragraph and for each of them several informations : the list of entities inside the paragraph, size, summaries, etc 
- **torch_loader** module is used to load and vectorize on the fly the preprocess json file so that in can be directly feed into a GPT2 model for fine-tuning
- **model_training** contains the script to fine-tune the GPT2 model, it is simply a small adaptation of huggingface/run_langage_modeling to allow the use of our custom dataset
- **model_evaluation** module will be used to evaluate the output quality of a fine-tune GPT2 model 
- **model_use** module interface our GPT2 fine-tune model with the web_service backend 
- **web_server** contains the back_end interface of our web service. The web service front-end has been pushed to a separate repo and can been found at : https://github.com/WeazelDev/AITextGeneratorFront
- **third_party** folder contains several framework that has been cloned directly into our project 

## Download the data

To download the training data, download and extract the following archive in the ```data``` folder:

https://drive.google.com/open?id=1wfWNo4p91OPHUSBYJ9Bv-oMNawxv_tAH

## Run fine-tuning script 

To fine-tune the GPT2 model : 

```
git clone https://github.com/WeazelDev/AITextGenerator.git
cd AITextGenerator
pip install -r requirements.txt
```

Then download the preprocess text data from : # TODO ADD LINK TO DATA and then simply run the *train* script

```
./train 
```

