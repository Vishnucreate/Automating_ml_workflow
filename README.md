AutoML Workflow & Azure ML Integration

📖 Table of Contents

Overview

Features

Architecture

Usage

Azure ML Deployment

Folder Structure

Installation

Contributing

License

🔍 Overview

This project demonstrates an end-to-end AutoML workflow using Python and Azure Machine Learning. It includes data ingestion, preprocessing, model training, and automated deployment to Azure ML.

✨ Features

Data Loading: CSV ingestion (Songs.csv).

NLP Preprocessing: SpaCy-based text cleaning and entity recognition.

OpenAI Integration: Custom prompts via import_openai.py.

Model Training & Scoring: Example training script ML_app.py.

Automated Deployment: Sample Azure ML pipeline YAML.

🏗 Architecture

flowchart TD
  A[Load Data] --> B[Text Preprocessing]
  B --> C[Feature Extraction]
  C --> D[Model Training]
  D --> E[Evaluation]
  E --> F[Deploy to Azure ML]

🚀 Usage

# Clone the ML_app folder
cd Automating_ml_workflow/ML_app

# Run the main application
default python
python ML_app.py --data Songs.csv --output results/

# Perform SpaCy text tasks
python Spacy_task.py --input Songs.csv --output cleaned.csv

# Validate with spaCy checker
python spacy_checker.py --file cleaned.csv

# Test OpenAI integration
python import_openai.py --prompt "Your custom text prompt"

🌐 Azure ML Deployment

Define pipeline in aml_pipeline.yaml.

Upload workspace config or set environment variables.

Submit run:

az ml job create --file aml_pipeline.yaml

🗂 Folder Structure

ML_app/
├── ML_app.py            # Main AutoML application script
├── Songs.csv            # Sample dataset
├── Spacy_task.py        # Text preprocessing with spaCy
├── import_openai.py     # OpenAI API helper functions
├── spacy_checker.py     # Validation of spaCy outputs
└── README.md            # This file

⚙️ Installation

# Create virtual env
git clone https://github.com/Vishnucreate/Automating_ml_workflow.git
cd Automating_ml_workflow/ML_app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

🤝 Contributing

Contributions are welcome! Please:

Submit issues for bugs

Open pull requests for new features or improvements

📄 License

Licensed under the MIT License. See LICENSE for details.
