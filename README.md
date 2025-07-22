# Churn Prediction App

A Streamlit web application that predicts customer churn risk for telecom companies using machine learning.

## Prerequisites

* **Python >= 3.12.3** (Recommended: use [pyenv](https://github.com/pyenv/pyenv))

* **Poetry** (Dependency manager - [installation guide](https://python-poetry.org/docs/))

* **Git** (Version control system - [installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

* **GNU Make** (Required for build automation):
   - **Linux**:  Usually pre-installed. If not, install it via your package manager (e.g., `sudo apt install make` on Debian/Ubuntu).
   - **macOS**: Install via  [Homebrew](https://brew.sh)  (`brew install make`)
   - **Windows**: Install via [Chocolatey](https://chocolatey.org/install) (`choco install make`).

## Installation

1. **Clone this repository**:
```bash
   git clone https://github.com/Jhonsilvaa/churn-prediction-app.git
```

2.  **Navigate to project directory**:
```bash
   cd churn-prediction-app
```
3. **Install dependencies with Make**:
```bash
    make install
```

## Running the App
To run the application:

1. **Start the Streamlit app**:

```bash
    make run
```
> The application will start at http://localhost:8501

2. **Input customer data**:
Once the app is running, use the sidebar to input customer details such as gender, senior citizen, tenure, etc.

3. **Get predictions**:
After filling in the required fields, click the "Check Churn Risk" button to see the prediction and Churn or Retention Probability.

## License

This project is licensed under the GNU General Public License v3.0, as indicated by the LICENSE file in the repository. 