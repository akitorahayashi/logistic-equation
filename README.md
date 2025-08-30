# Logistic Equation Analysis Tool (Streamlit Version)

## Overview

This project is an interactive web application for analyzing time-series data by fitting it to a logistic equation and making future predictions. It is useful for modeling S-shaped growth phenomena, such as population trends or product adoption rates.

Users can upload an Excel file for analysis via the web UI, interactively set the parameter search range, and view the analysis results (graphs, prediction data) directly in the browser.

## Key Features

- **Interactive UI**: An intuitive and easy-to-use web interface built with [Streamlit](https://streamlit.io/).
- **File Upload**: Directly upload Excel files for analysis from your browser.
- **Dynamic Parameter Settings**: Set the search ranges for carrying capacity `K` and growth rate `γ` in real-time using UI sliders and number inputs.
- **Real-time Result Display**: Analysis results are displayed in organized tabs:
    - **Fitting Result**: Check the optimized parameters and a graph showing the fit to the actual data.
    - **Forecast**: View the future prediction graph based on the model.
    - **Download Data**: Download the prediction data as an Excel file.
- **High-Precision Numerical Analysis**: Solves the logistic differential equation using the 4th-order Runge-Kutta method to build a highly accurate model (the core logic is unchanged from the original version).

## Requirements

- Python: `==3.12.11`
- Key Libraries:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `openpyxl`
  - `matplotlib`
  - `scikit-learn`

## Dependencies

This project uses [Poetry](https://python-poetry.org/) to manage dependencies.

Install the required libraries with the following command:
```bash
poetry install
```

## Directory Structure

```
.
├── src/
│   ├── components/
│   │   ├── sidebar.py
│   │   └── results_display.py
│   ├── config/
│   ├── model/
│   └── main.py  <-- Application entry point
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## Usage

### 1. Launching the Application

Run the following command in the project root directory to launch the web application. It will automatically open in your browser.

```bash
streamlit run src/main.py
```
Alternatively, you can use the Poetry script:
```bash
poetry run run
```
Or the Makefile command:
```bash
make run
```

### 2. Using the Web Application

Once the application is running, follow the instructions in the left-hand sidebar:

1.  **Upload Data File**:
    - Upload an Excel file (`.xlsx`) containing the time-series data you want to analyze.
    - The first column should be time. You may provide either a 0-based time index (0,1,2,...) or calendar years (e.g., 1950, 1951, ...). The app automatically normalizes years by subtracting the first year to start from t=0. The second column is the observed value. No headers are needed.
2.  **Set Parameter Search Range**:
    - Specify the search range and step size for **Carrying Capacity (K)** and **Growth Rate (γ)**.
    - For large K values, you can select units (e.g., thousands, millions, billions) for easier input.
3.  **Set Forecast Period**:
    - Set the start year of the data and how many years into the future you want to predict.
4.  **Run Analysis**:
    - Click the "Run Analysis" button to start the parameter search and future forecast.

Once the analysis is complete, the results (optimal parameters, graphs, and prediction data) will be displayed in the main panel.
