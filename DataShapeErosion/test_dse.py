import numpy as np
from dse_model import DataShapeErosion

# Simulate data
data = np.random.rand(100, 2)
dse = DataShapeErosion(erosion_rate=0.02, min_neighbors=30)  # Conservative erosion
dse.fit(data)

# Perform erosion
eroded_data = dse.erode(steps=5)
anomalies = dse.detect_anomalies()

print("Number of Anomalies Detected:", len(anomalies))
print("Remaining Data Points After Erosion:", len(eroded_data))
print("Test Passed: Anomalies < 20:", len(anomalies) < 20)
print("Test Passed: Eroded Data Points > 50:", len(eroded_data) > 50)
