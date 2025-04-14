# HR_PlotModes
Repository for plotting the mode shapes in the Hospital Real

## 1. Prerequisites
Make sure you have the following prerequisites set up:

- Python 3.x (tested with Python 3.11.5)

## 2. Installation

**Clone the repository** to your local machine with the following command:
```bash
git clone https://github.com/asanchezlc/HR_PlotModes.git

**Create a virtual environment**: Navigate to the project directory and create a virtual environment. Virtual environments help manage project-specific dependencies.
``` bash
python -m venv .venv
```

**Activate the virtual environment**: Activate the virtual environment based on your operating system.
``` bash
# En Windows
.\.venv\Scripts\activate
# En macOS/Linux
source .venv/bin/activate
```

**Update pip**: Update pip to the latest version.
``` bash
python -m pip install --upgrade pip
```

**Install dependencies:** Install the necessary dependencies and libraries. Run the following command with the virtual environment activated.
``` bash
pip install .
```

## 3. Files

The `data` folder contains the input files for the project:

- **Geometry file (.txt)**: Contains the geometry of the *Hospital Real*. This includes node coordinates and connectivity information.
- **OMA data file (.json)**: OMA data obtained via API download.

