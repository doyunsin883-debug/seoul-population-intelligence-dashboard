# Seoul Population Intelligence Dashboard

A Streamlit-based data analytics dashboard for analyzing Seoul 250m grid-level population data.

This project analyzes native population, foreign population, hourly population trends, spatial concentration, foreign nationality distribution, clustering patterns, and machine learning-based population prediction.

---

## Project Overview

This project was built as a personal data analytics portfolio project.

The dashboard provides:

- Seoul 250m grid-level population visualization
- Hourly population trend detection
- Administrative area and grid-level concentration analysis
- Age group analysis
- Foreign nationality analysis
- Area-nationality concentration analysis
- Spatial clustering using KMeans
- Population prediction using machine learning models

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- PyProj

---

## Main Features

### 1. Seoul Population Map

Visualizes population concentration by 250m grid using interactive maps.

### 2. Trend Detection

Analyzes hourly changes in total population, native population, and foreign population.

### 3. Area & Age Analysis

Identifies crowded administrative areas, grids, and age groups.

### 4. Foreign Nationality Analysis

Analyzes which foreign nationality groups are concentrated in which administrative areas.

### 5. Clustering

Uses KMeans clustering to classify grid areas based on population patterns.

### 6. Machine Learning Prediction

Predicts population using Random Forest and Gradient Boosting models.

---

## Dataset

The project uses Seoul 250m grid-level stay population data.

Data files used:

- Native population data
- Foreign population by nationality data

Recommended project structure:

```text
seoul-population-intelligence-dashboard/
│
├── worlds.py
├── README.md
├── requirements.txt
└── data/
    ├── SEOUL_STYTIME_04_250M_OPEN_NATIVE_20260629.csv
    └── SEOUL_STYTIME_05_250M_OPEN_FORN_LONG_20260629.csv
```

If the CSV files are not inside the `data` folder, the dashboard will ask the user to upload the files from the sidebar.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run worlds.py
```

---

## Key Learning Points

Through this project, I practiced:

- Real-world data preprocessing
- Spatial data visualization
- Exploratory data analysis
- Population trend analysis
- Foreign nationality distribution analysis
- Clustering
- Machine learning regression
- Dashboard development
- Portfolio documentation

---

## Author

**Shin Do Yun**  
Gachon University  
Industrial Engineering Student
