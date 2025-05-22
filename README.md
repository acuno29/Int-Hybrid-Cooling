# 🚀 Intelligent Hybrid Cooling System

## 📌 Project Title
**Design and Development of Intelligent Hybrid Cooling:**  
*Adaptive Temperature and Humidity Monitoring Control for Electrical Rooms*

---

## 📘 Overview

This project presents a Python-based intelligent hybrid cooling system designed to monitor and control temperature and humidity levels in electrical rooms. Utilizing **Support Vector Machine (SVM)** algorithms, the system predicts environmental conditions and adjusts cooling mechanisms accordingly to maintain optimal conditions.

---

## 📁 Repository Contents

- `SVM_TRAIN_TEMP_HUM.py` – Main Python script implementing the SVM model for temperature and humidity prediction  
- `temperature_humidity_data.csv` – Dataset containing historical temperature and humidity readings used for training the model  
- `Instructions.pdf` – Detailed documentation outlining the system's design, implementation, and operational guidelines  

---

## 🛠️ Prerequisites

Before running the code, ensure your system has the following:

- **Python**: Version 3.6 or higher – [Download here](https://www.python.org/downloads/)
- **pip**: Python package installer (usually included with Python)

### Required Python Libraries

Install the following libraries if not already installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (for optional visualization)

---

## 📥 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/acuno29/Int-Hybrid-Cooling.git
```

### 2. Navigate to the Project Directory

```bash
cd Int-Hybrid-Cooling
```

### 3. (Optional) Create and Activate a Virtual Environment

```bash
python -m venv venv
```

#### On Windows:

```bash
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🚀 Running the Application

Make sure you are in the project directory:

```bash
cd Int-Hybrid-Cooling
```

Then, run the main Python script:

```bash
python SVM_TRAIN_TEMP_HUM.py
```

### This script will:

- Load and preprocess the dataset from `temperature_humidity_data.csv`
- Train the SVM model
- Predict temperature and humidity
- Optionally visualize the results using matplotlib

---

## 📄 Additional Documentation

Refer to `Instructions.pdf` in the repository for a comprehensive overview of the system architecture, methodology, and usage.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas or improvements:
- Fork the repository
- Make your changes
- Submit a pull request

Please follow best practices and include clear documentation.

---

## 📧 Contact

For questions, feedback, or collaboration opportunities, open an issue on the GitHub repository or contact the maintainer directly.

---
