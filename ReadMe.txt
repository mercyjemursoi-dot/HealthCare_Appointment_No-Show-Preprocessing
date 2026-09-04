# Healthcare Appointment No-Show Preprocessing

This repository contains a Python-based data cleaning and feature engineering pipeline for a healthcare dataset. The goal of this project is to prepare raw appointment data (`HealthData.csv`) for machine learning models to predict **Patient No-Shows** at the time of booking.

## Project Overview

Predicting whether a patient will show up for their appointment is crucial for clinic efficiency. This script focuses on:
1.  **Preventing Data Leakage:** Removing variables that are only known *after* the appointment occurs.
2.  **Data Cleaning:** Handling missing values and removing inconsistent records.
3.  **Feature Engineering:** Converting categorical and temporal data into numerical formats suitable for Scikit-Learn or XGBoost models.

## Tech Stack
*   **Python 3.x**
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.

## Dataset Requirements
The script expects a file named `HealthData.csv` in the root directory. Key columns used include:
*   `appointment_outcome`: The target variable (Attended, No-Show, Cancelled).
*   `booking_lead_days`: Days between booking and the appointment.
*   `distance_to_clinic_km`: Numerical distance.
*   `gender`, `appointment_type`, `age_group`: Categorical features.

## Preprocessing Steps

The script follows a 6-step pipeline:

### 1. Address Data Leakage
Removes columns that would not be available at the moment a patient books an appointment (e.g., `waiting_time_minutes`, `reminder_sent`).

### 2. Target Engineering
*   Removes "Cancelled" records to focus strictly on the binary classification: **No-Show (1)** vs. **Attended (0)**.

### 3. Missing Value Imputation
*   Fills missing values in `distance_to_clinic_km` using the **median** value to maintain data distribution.

### 4. Sanity Checks & Data Integrity
*   Converts date columns to datetime objects.
*   Removes records where the appointment date occurs before the booking date (negative lead days).
*   Standardizes the `gender` column.

### 5. Categorical Encoding
*   **Binary Encoding:** Gender (Female/Male/Other).
*   **One-Hot Encoding:** Appointment Type and Age Group.
*   **Ordinal Encoding:** Appointment Time (Morning < Afternoon < Evening).
*   **Cyclical Mapping:** Appointment Day of the week.

### 6. Final Cleanup
*   Drops redundant string/date columns.
*   Removes duplicate rows.
*   Exports the final dataset to `Cleaned_Data.csv`.

4.  **Output:**
    The script will output the initial and final shapes of the data and generate a `Cleaned_Data.csv` file ready for training.

