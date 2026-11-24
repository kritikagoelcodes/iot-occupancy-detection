IoT-Based Indoor Occupancy Detection — Complete Demo Folder

CONTENTS
- app.py: Streamlit user interface for interactive occupancy prediction.
- requirements.txt: Python packages required by the app.
- RUN_APP_WINDOWS.bat: Windows shortcut to install dependencies and launch the app.
- iot_occupancy_detection_five_algorithms.ipynb: Updated notebook containing the five selected algorithms and evaluation metrics.
- Occupancy.csv: YOUR DATASET FILE MUST BE PLACED HERE.

IMPORTANT DATASET STEP
The ZIP does not contain Occupancy.csv because the CSV itself has not been uploaded in this conversation.
Copy your existing Occupancy.csv into this exact folder, beside app.py.

HOW TO RUN ON WINDOWS
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Copy Occupancy.csv into the extracted folder.
4. Double-click RUN_APP_WINDOWS.bat.
   Alternatively, open a terminal in this folder and run:
       python -m pip install -r requirements.txt
       python -m streamlit run app.py
5. Open the local URL shown in the terminal (usually http://localhost:8501).

HOW TO DEMONSTRATE
1. Show the model evaluation metrics at the top.
2. Change Temperature, Humidity, Light, CO2 and HumidityRatio.
3. Click Predict Occupancy.
4. Show the predicted class and estimated probability.

MODEL / EVALUATION NOTE
The interactive app uses Random Forest as the demonstration model. It trains from Occupancy.csv on app startup. The displayed metrics are computed on a stratified 25% held-out test set. The demo model is then fitted on the full dataset for interactive predictions.

The notebook contains the five-model comparison. Run all notebook cells after placing Occupancy.csv in the same folder. Verify the notebook's dataset paths are relative to the folder before presenting it.

SCOPE NOTE
This is a dataset-driven prediction demonstration with manually entered readings. It does not connect to physical ESP32/Arduino sensors, MQTT, or live room hardware. Describe physical sensor integration as future work unless you implement it.
