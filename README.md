# Velo 🚀
Turn ideas into trained ML models instantly.

## What it does
Velo takes a natural language problem, finds or generates a dataset, trains multiple ML models, and returns the best one ready for download.

## Features
- Dataset upload or AI generation
- Automatic data cleaning
- Multi-model training (RF, GB, Logistic, MLP)
- Leaderboard + best model selection
- Download trained model (.pkl)
- Prediction API

## Run locally
pip install -r requirements.txt
uvicorn app:app --reload

## Example
Input: "Predict student exam scores"
Output: Trained model + dataset + predictions
