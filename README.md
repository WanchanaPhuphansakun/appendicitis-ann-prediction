# Appendicitis Prediction using Artificial Neural Networks

This project develops an Artificial Neural Network (ANN) model to predict appendicitis using clinical patient data from the Regensburg Pediatric Appendicitis dataset.

## Project Overview

The system was built in Google Colab using TensorFlow/Keras. It takes patient clinical features as input and predicts whether appendicitis is present.

## Results

- Accuracy: 0.8846
- AUROC: 0.9445
- Sensitivity: 0.8925
- Specificity: 0.8730

## Project Structure

appendicitis-prediction-ann/
│
├── data/       # dataset
├── model/      # saved trained model and metadata
├── notebooks/  # notebook version of the project
├── report/     # report document
├── results/    # plots and confusion matrix
└── src/        # python scripts

## Files

- `data/Regensburg Pediatric Appendicitis.csv` — dataset
- `model/appendicitis_tf.keras` — saved trained ANN model
- `model/appendicitis_tf_meta.json` — metadata including feature columns
- `notebooks/appendicitis-prediction-ann.ipynb` — notebook implementation
- `report/appendicitis-prediction-ann Report Wanchana Phuphansakun.docx` — project report

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## How to Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Open the notebook in Jupyter or Google Colab, or run the Python scripts.

3. Ensure the dataset, saved model, and metadata are available in the correct folders.

## Dataset

The project uses the Regensburg Pediatric Appendicitis dataset.

## Notes

The metadata file stores:
- target column
- encoded feature columns
- label mapping strategy

This helps make the saved model reproducible.
