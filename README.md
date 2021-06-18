# CS-439 Group Project - Variance Reduction methods for GANs
This repository consists project implementation for [**EPFL CS-439**](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439) course.

# Introduction
In the folder ```scripts``` you can find all the code for the implemetation of the **ExtraAdam** and [**Lookahead**](https://arxiv.org/pdf/1907.08610.pdf) algorithms. In the folder ```extra_var_reduction``` you can find implementation of the [**Extragradient with Variance Reduction (EGVR)**](https://arxiv.org/pdf/2102.08352.pdf) algorithm (with some our modifications, for more details look into report).

Also there is an implementation of GAN models in the ```models.py``` which were trained by the mentioned algorithms. The goal of the project to compare perfomance of _new_ algorithm **EGVR** developed for convex-concave MinMax problems on more complicated tasks as GANs with respect to the popular\efficient algorithms as **ExtraAdam** and **Lookahead**. 

**`TeamMembers`**: **Denys Pushkin**, **Yaroslav Kivva**, **Zhecho Mitev**

# Quickstart
To reproduce our results you should open the ```run.ipynb``` notebook from the root of the repository and follow the instructions inside notebook.

_Remark: All the models was trained with GPUs on Colab and it takes approx 1-2 hours for each experiment! If you run it with CPU it could take a lot of time to fit the models properly._

## Google Colab
To run the notebook in the colab you should do next steps:
1. Copy all files from the git into some folder ``` *project_dir* ``` at your Google Drive.
2. Open run.ipynb with Colab
3. Mount notebook to your google drive
4. Run command ```import os; os.chdir(*project_dir*)``` --- your notebook will be considered as in the folder ``` *project_dir* ``` and so all imports would be consedered correctly
5. Run the experiments in the notebook

## Implemetation
