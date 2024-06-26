IMPORTANT: THIS GITHUB REPO IS STILL WIP

# JobMajorClf: An end-to-end MLOps project with AWS Sagemaker

### Table of contents
1. [Example](#example)
2. [Example2](#example2)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)

This small code repository showcases the basics of developing an MLOps project. With AWS Sagemaker, we have at hand all the tooling required to build an end-to-end pipeline that takes a Machine Learning project from idea to production.

In this project, we will train a Job title classifier, produce corresponding preprocessing, training and evaluation scripts, and integrate them into a Sagemaker pipeline which outputs a model artifact ready for deployment in a production environment.


## Prerequisites
**- Access to an AWS account**
**TODO others**

## Jupyter Notebook "Start Here.ipynb"
The notebook "Start Here.ipynb" contains the python code to complete this project. Below we describe the steps to correclty setup the environment for running the notebook:

1. Login to your AWS Sagemaker Studio account.
2. In the File menu, open New/Terminal
3. Move to the directory on which this tutorial repo is to be downloaded:
  `cd ~`
4. Clone the repository:
`git clone https://github.com/rgmartin/job-major-clf.git`
5. Move into the new directory that hosts the repo:
`cd job-major-clf`
6. In the file explorer provided by Sagemaker studio (folder icon in the left menu), navigate to the recently created repo folder `job-major-clf` and open the notebook `Start Here.ipynb`.
7. Select the Image `Data Science 3.0` in the pop-up dialog that appears. For the purposes of this project, we can select `ml.t3.medium` as our instance type. Press Select.
8. Once the notebook kernel starts, you can run each of the notebook cells one by one, to produce the desired result. 

The remaining sections of this tutorial describe each of the sections of the `Start Here.ipynb` notebook, which you can follow at the same time that you run the corresponding notebook cells.


## Setup
To begin, setup the libraries that are needed for this project

```python
%pip install --upgrade pip sagemaker
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
import time
import os
import json
import boto3
import numpy as np  
import pandas as pd 
import sagemaker
from time import gmtime, strftime, sleep

sagemaker.__version__
```
... and the constants: 

```python
#Variables for handling Sagemaker sdk
boto_session = boto3.Session()
region = boto_session.region_name
bucket_name = sagemaker.Session().default_bucket()
bucket_prefix = "job-major-clf"  
sm_session = sagemaker.Session()
sm_client = boto_session.client("sagemaker")
sm_role = sagemaker.get_execution_role()

# Pipeline objects
project = "test-job-major-clf"
pipeline_name = f"{project}-pipeline"
pipeline_model_name = f"{project}-model"
model_package_group_name = f"{project}-model-group"
endpoint_config_name = f"{project}-endpoint-config"
endpoint_name = f"{project}-endpoint"

#Instance types and counts
process_instance_type = "ml.c5.xlarge"
train_instance_count = 1
train_instance_type = "ml.m5.xlarge"

#S3 urls for data
train_s3_url = f"s3://{bucket_name}/{bucket_prefix}/train"
validation_s3_url = f"s3://{bucket_name}/{bucket_prefix}/validation"
test_s3_url = f"s3://{bucket_name}/{bucket_prefix}/test"

output_s3_url = f"s3://{bucket_name}/{bucket_prefix}/output"
```

The variables related to Sagemaker sdk are required for interacting with the `boto3` library, which provides programatic interaction with the Sagemaker service using python. The pipeline objects are names utilized to identify the multiple sagemaker objects required to produce our MLOps pipeline. Instance types and counts describe which computational resources are going to be utilized for processing our datasets and training our models. Finally, s3 urls variables correspond to the locations of our training, validation and test splits in the S3 storage service provided by aws, as well as the location on which the output produced by the evaluation of our model is to be stored.

Let's print the resulting variables for future reference:
```python
print(sm_role)
print(f"Train S3 url: {train_s3_url}")
print(f"Validation S3 url: {validation_s3_url}")
print(f"Test S3 url: {test_s3_url}")
print(f"Data baseline S3 url: {baseline_s3_url}")
print(f"Evaluation metrics S3 url: {evaluation_s3_url}")
print(f"Model prediction baseline S3 url: {prediction_baseline_s3_url}")
```

## The dataset

The O*NET database [[https://www.onetonline.org/]] contains hundreds of standardized and occupation-specific descriptors on multiple occupations covering the entire U.S. economy. It is continually updated by specialists in a broad range of industries and made available to the public at no cost. 

The O*NET taxonomy [[https://www.onetcenter.org/taxonomy.html]] assigns one of each of the 23 Major groups, 98 Minor groups, 459 broad occupations, 867 detailed SOC occupations. This is done by assigning specific codes (see the taxonomy website) to each occupation. For example, *"Neurodiagnostic Technologists"* are assigned the code 29-2099.01, on which 

- the first two digits 29 represent the Major Group "Healthcare Practitioners and Technical Occupations",
- the next two digits 20 represent the Minor Group  "Health Technologists and Technicians",
- the next two digits 99   represent the broad occupation group "All Other health technicians and technologist" and
- the final two digits 01 represent the detailed occupation "Neurodiagnostic Technologists".

In this project, we will focus in classifying standard job titles into one of the 23 Major groups listed in the O*NET database. Each of these groups (classes) has standard and alternate titles associated with them, as listed in the public files `Occupation Data.xlsx` [[https://www.onetcenter.org/dl_files/database/db_28_1_excel/Occupation%20Data.xlsx]] and `Alternate titles.xlsx`[[https://www.onetcenter.org/dl_files/database/db_28_1_excel/Alternate%20Titles.xlsx]] made available by O*NET.  This results in a labeled dataset for our multilabel classification task on which each of the titles (standard and alternates) is assigned a single class (Major group).

Let's download the two source files into our local `data` folder:
```bash
# Download datasets
!wget -P data/ https://www.onetcenter.org/dl_files/database/db_28_1_excel/Occupation%20Data.xlsx
!wget -P data/ https://www.onetcenter.org/dl_files/database/db_28_1_excel/Alternate%20Titles.xlsx
```
Now, a quick inspection of their content:

```python

df_alt = pd.read_excel('data/Occupation Data.xlsx')
df_occ = pd.read_excel('data/Alternate Titles.xlsx')

df_alt.head()
df_occ.head()

```
```
O*NET-SOC Code	Title	Description
0	11-1011.00	Chief Executives	Determine and formulate policies and provide o...
1	11-1011.03	Chief Sustainability Officers	Communicate and coordinate with management, sh...
2	11-1021.00	General and Operations Managers	Plan, direct, or coordinate the operations of ...
3	11-1031.00	Legislators	Develop, introduce, or enact laws and statutes...
4	11-2011.00	Advertising and Promotions Managers	Plan, direct, or coordinate advertising polici...


O*NET-SOC Code	Title	Alternate Title	Short Title	Source(s)
0	11-1011.00	Chief Executives	Aeronautics Commission Director	NaN	08
1	11-1011.00	Chief Executives	Agency Owner	NaN	10
2	11-1011.00	Chief Executives	Agricultural Services Director	NaN	08
3	11-1011.00	Chief Executives	Arts and Humanities Council Director	NaN	08
4	11-1011.00	Chief Executives	Bank President	NaN	09
```

## The pipeline

After successful completion of all the steps listed in `Start here.ipynb`, the resulting processing/training/evaluation pipeline will be automatically run on the Sagemaker environment. A diagram of the pipeline can be seen by accessing Home/Projects/<project_name>/Pipelines/<pipeline_name>/<execution_name>:

![alt text](Diagram.png "Pipeline diagram")

It's not the goal of this tutorial to dive into the details of each of these steps. Here an eagle-eye overview: the `preprocess-data` step transforms each of the labeled records into tokenized datasets that are suitable for training a classifier using Bert-like language models. The `train` step proceeds with the training of a distilBERT model to classify each record into one of the job majors. The performance of the obtained model is assessed with the `evaluate-model` by computing the f1 global score of the predictions against the true labels of the job titles. If the score surpasses a desired treshold (f1), the `check-test-score`  will submit the model to the Model Registry with the `RegisterModel` step, otherwise, the pipeline fails with the `fail` step output.

