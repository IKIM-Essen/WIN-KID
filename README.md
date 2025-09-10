# WIN-KID <img src="resources/images/WINKID_Logo_web.svg" alt="WIN-KID Logo" align="right" width="120"/>


This repository contains all code and resources used accompanying the [WIN-KID](https://www.medizin.uni-muenster.de/win-kid/startseite.html) project.  
It includes data preprocessing pipelines, feature engineering, and machine learning models for **predicting antimicrobial resistance (AMR)** from bacterial whole-genome sequencing data.

---

## 📖 About the Project

Antimicrobial resistance (AMR) is a major global health challenge.  
In this work, we analyze bacterial genomes from the **BV-BRC** database and locally sequenced isolates to develop machine learning models for predicting resistance phenotypes.  

We compared two approaches:
- **One-layer Random Forest (RF):** Independent models per antibiotic.  
- **Stacked Random Forest (RF):** Layered approach leveraging inter-antibiotic relationships.  

Evaluation was performed on *E. coli*, *K. pneumoniae*, and *A. baumannii* using Accuracy, F1-score, and ROC AUC.

---

## 📂 Repository Structure

- `resources/` – figures and diagrams for the paper  
- `workflow/` – python source code for data processing and modeling
- `rf_models/` – previously trained and saved models
- `INSTALLATION.md` – installation guide  
- `USER_GUIDE.md` – user guide  

---

## ⚙️ Installation

Follow the steps in [INSTALLATION.md](./INSTALLATION.md) to set everything up.  
The repository provides an `environment.yml` file for reproducibility.

---

## 🚀 Usage

Detailed usage instructions are provided in [USER_GUIDE.md](./USER_GUIDE.md).  
This includes preprocessing data, training models, and reproducing figures and evaluation metrics.


---

## 📧 Contact

For questions, please contact the WIN-KID team at [IKIM Essen](https://www.ikim.uk-essen.de/groups/ds).  
