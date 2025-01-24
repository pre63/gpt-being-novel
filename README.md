# GTP Being Creative

GPT creating algos for funsies.

## How to use
Install
```
make install
```

This will execute all tests and output the results for each algorithm. Assertions within the test scripts validate the performance benchmarks.
```
make test
```

### **Expected Outputs**
- **DynamicTaskPrioritizer**: Outputs the prioritized task order and validates correctness.
- **PENN**: Outputs the Mean Squared Error (MSE) of predictions, ensuring it meets the threshold.
- **DSE**: Outputs the number of anomalies detected and ensures data erosion performs as expected.
- **SAMNN**: Outputs the MSE of predictions, validating adaptive modular learning.

Enjoy!

### **Algorithms**

#### **1. Dynamic Task Prioritizer (DTP)**
- **File**: `DynamicTaskPrioritization/dtp_model.py`
- **Description**: A graph-based model for dynamic prioritization of tasks based on dependencies and real-time feedback.
- **Test**: Asserts the correctness of task prioritization with predefined feedback.

#### **2. Predictive Entanglement Neural Network (PENN)**
- **File**: `PredictiveEntanglementNN/penn_model.py`
- **Description**: A neural network architecture that entangles feature relationships to enhance prediction accuracy in noisy environments.
- **Test**: Asserts that the model achieves a low Mean Squared Error (MSE) on synthetic data.

#### **3. Data Shape Erosion (DSE)**
- **File**: `DataShapeErosion/dse_model.py`
- **Description**: A density-based anomaly detection algorithm that simulates data erosion to identify outliers.
- **Test**: Asserts that the algorithm detects anomalies correctly and performs expected levels of data erosion.

#### **4. Self-Assembling Modular Neural Network (SAMNN)**
- **File**: `SelfAssemblingModularNN/samnn_model.py`
- **Description**: A neural network that dynamically constructs specialized modules based on feature clusters, adapting during training.
- **Test**: Asserts that the model achieves a low MSE on synthetic data.

### **Contributing**
Feel free to submit pull requests or open issues for improvements or feature suggestions.
