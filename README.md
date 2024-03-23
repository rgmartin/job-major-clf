IMPORTANT: THIS GITHUB REPO IS STILL WIP

# Tutorial: Developing a simple end-to-end MLOps project with AWS Sagemaker

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



These commands will show that each Title and Alternate titles is associated with a single O*NET-SOC Code. As discussed before, we will only be working with the major occupation group for our classification task, that is, the first two digits of the Codes. 

Now we will upload these files to AWS S3 storage for easy access during the preprocessing stage of our MLOps pipeline:

```python

try:
    input_s3_urls
except NameError:      
    # If input_s3_url is not defined, upload the dataset to S3 and store the path
    input_s3_urls = [
        sagemaker.Session().upload_data(
            path="data/Alternate Titles.xlsx",
            bucket=bucket_name,
            key_prefix=f"{bucket_prefix}/input"
        ),
        sagemaker.Session().upload_data(
            path="data/Occupation Data.xlsx",
            bucket=bucket_name,
            key_prefix=f"{bucket_prefix}/input"
        ),
    ]
        
        
        
    print(f"Uploaded datasets to {input_s3_urls}")
```
## Create the ML Project programatically

AWS provides multiple Software Development Kits (SDK) that allow accessing its services programatically. This is a convenient alternative to accessing their UI that allow for better traceability and versioning of the project creation process. The `boto3` library exposes the AWS SDK in the `python` language. We are going to use it to retrieve the MLOps project template provided in the `service catalog` for building and training a Machine Learning model:

```python
#Define clients and services
sm = boto3.client("sagemaker")
sc = boto3.client("servicecatalog")

sc_provider_name = "Amazon SageMaker"
sc_product_name = "MLOps template for model building and training"

#Find the ProductId for the MLOps project template
p_ids = [p['ProductId'] for p in sc.search_products(
    Filters={
        'FullTextSearch': [sc_product_name]
    },
)['ProductViewSummaries'] if p["Name"]==sc_product_name]

if not len(p_ids):
    raise Exception("No Amazon SageMaker ML Ops products found!")
elif len(p_ids) > 1:
    raise Exception("Too many matching Amazon SageMaker ML Ops products found!")
else:
    product_id = p_ids[0]
    print(f"ML Ops product id: {product_id}")


#Retrieve Provisioning Artifact ID

provisioning_artifact_id = sorted(
    [i for i in sc.list_provisioning_artifacts(
        ProductId=product_id
    )['ProvisioningArtifactDetails'] if i['Guidance']=='DEFAULT'],
    key=lambda d: d['Name'], reverse=True)[0]['Id']
```

No we create the project using the obtained product and artifact ID from the AWS ServiceCatalog. A waiting cycle is introduced until the project creation is completed

```python
#Define project name
project_name = f"test-job-major-clf"

# create SageMaker project
r = sm.create_project(
    ProjectName=project_name,
    ProjectDescription="Model build project",
    ServiceCatalogProvisioningDetails={
        'ProductId': product_id,
        'ProvisioningArtifactId': provisioning_artifact_id,
    },
)

print(r)

#Wait for the project creation to be complete
while sm.describe_project(ProjectName=project_name)['ProjectStatus'] != 'CreateCompleted':
    print("Waiting for project creation completion")
    sleep(10)
    
print(f"MLOps project {project_name} creation completed!")

```
## Clone the default project code to AWS Studio File system
Clone the project default code to the Studio file system:
1. Choose **Home** in the Studio sidebar
2. Select **Deployments** and then select **Projects**
3. Click on the name of the project you created to open the project details tab
4. In the project tab, choose **Repositories**
5. In the **Local path** column for the repository choose **clone repo....**
6. In the dialog box that appears choose **Clone Repository**

## Update the default code with our classifier pipeline

Now that we have a local copy of the created MLOps project in AWS Sagemaker studio, we must replace the default pipeline `abalone` with the one provided for this project in the `jobmajorclf` folder of this tutorial repository.




## Third Example
## [Fkurth Example](http://www.fourthexample.com)



