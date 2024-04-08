# LinkedIn Dream Job Scorer

This project was implemented as part of a "Data Gathering and Managing" Lab course, Technion. Our product is added to a position page in LinkedIn, and offers the user unique, personalized insights as well as interview practice.

### Team Members
- [Renana Shahak](renanas@campus.technion.ac.il)
- [Gili Cohen](gili.cohen@campus.technion.ac.il)
- [Naomi Derel](naomi.derel@campus.technion.ac.il)

## Contents

1. [Introduction](#introduction)
2. [Data Gathering](#data-gathering)
3. [Models](#models)
    - [Company Similarity Model](#company-similarity-model)
    - [User Success Score Model](#user-success-score-model)
    - [Interview Practice Model](#interview-practice-model)
4. [Simulation Example](#simulation-example)

## Introduction

This repository contains a 'proof of concept' for the LinkedIn Dream Job Scorer, with the code necessary to recreate our data-scraping, our analysis and the different ML models we implemented. 

**Pipeline of our product functionality:**


## Data Gathering

Our project is dependent on exclusive LinkedIn datasets by BrightData. They contain 'companies' and 'profiles' data scraped from appropriate LinkedIn pages. 

Additional data is included in the 'data' folder, contains the following files.



## Models

### Company Similarity Model

Relevant code is available in the file `company_similarity_model.ipynb`. 

To conduct a similarity analysis, we used the ‘similar’ property in the data as our positive labels and additionally sampled a negative selection of companies. We used this data to train a Random Forest model which predicts our binary similarity label. During inference, we combine the company we want to enrich the data for with our sample of informative companies, and find the closest one. 

For this algorithm, we suggested the following features grounded in our domain knowledge: 
- NLP similarity features: comparing the tf-idf vectors with cosine similarity. We used this method on the name, ‘about’ description, and slogan of the company.
- Binary comparison (identical or not) between the industries and meta-industries of the companies, as they are nominal values and cannot be numerically compared. 
- Numerical features such as the number of locations and the company's size as numerical values.  
- Interaction feature between the sizes of the companies, to enhance the effect. 

After feature importance analysis, we retained only the features related to the name, 'about' section, and slogan, given the pivotal role of NLP analysis, along with the numerical size feature, which proved to be impactful.

<div style="background-color:white; width:70%; padding:10px; border-radius:10px; align:center">
    <img src="images/feature_analysis_reduced_similarity_model.png" width="100%">
</div>
<p>

The trained model is available only through our databricks, under the following path:

`/FileStore/shared_uploads/naomi.derel@campus.technion.ac.il/companies_model/rf_companies_similarity`

### User Success Score Model

### Interview Practice Model

*Note: This model incorporates the use of Gemini, therefore requires a personal API key, which will be removed after the project is graded for security reasons.*

Relevant code is available in the file `interview_practice_model.ipynb`.

## Simulation Example




