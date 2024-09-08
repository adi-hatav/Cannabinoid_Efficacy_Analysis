# Machine-Learning Approach Reveals Pain Relief by Medical Cannabis is Not a Placebo Effect

## Overview
This study investigates whether the pain relief provided by medical cannabis (MC) is more than just a placebo effect. We use a machine learning model to assess the predictive power of chemical compounds in cannabis alongside demographic and clinical features. Our findings highlight the significance of chemical composition in pain relief.

### Data Availability
The data used in this study will be made available in the supplemental table of the paper after its acceptance.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adi-hatav/Cannabinoid_Efficacy_Analysis.git
   cd Cannabinoid_Efficacy_Analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Prepare your data**: The data used in this project will be available as a supplementary table after the paper is accepted. Once the data is available:
    - Place the dataset in the root directory.
    - Update the data path in the `main.ipynb` notebook or scripts as needed.

2. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook
   ```
   - Open the `main.ipynb` file and run the cells to see the analysis and model training steps.

### Code Structure

- **main.ipynb**: The main Jupyter notebook that contains the entire workflow of the machine learning model, data preprocessing, and analysis.
- **utils.py**: Contains helper functions for data preprocessing, feature importance extraction, model tuning, and visualization.
- **requirements.txt**: A list of dependencies needed to run the project.
- **data/**: Directory to store the dataset.

### Reproducibility Notice

Please note that while efforts have been made to ensure reproducibility (by setting `random_state` for all stochastic processes), some slight variations in results may still occur across different runs of the code. These variations are caused by factors such as:
   
1. **Model Sensitivity**: The RandomForest algorithm can exhibit slight variations in results, especially when the data changes or due to floating-point precision differences in different environments.

2. **Floating-Point Precision**: Due to the nature of floating-point arithmetic, some differences can occur, particularly when dealing with large datasets or iterative calculations.

These differences are generally small and should not affect the overall conclusions.

## Acknowledgments
This research was conducted by Adi Hatav and the team from the Technion - Israel Institute of Technology under the supervision of Dvir Aran and David Meiri.
