import numpy as np
from sklearn.neighbors import NearestNeighbors


class DataShapeErosion:
  def __init__(self, erosion_rate=0.1, min_neighbors=5):
    self.erosion_rate = erosion_rate
    self.min_neighbors = min_neighbors

  def fit(self, data):
    self.data = data
    self.nbrs = NearestNeighbors(n_neighbors=self.min_neighbors).fit(data)

  def erode(self, steps=5):
    eroded_data = self.data.copy()
    for _ in range(steps):
      erosion_scores = self._compute_erosion_scores(eroded_data)
      threshold = np.percentile(erosion_scores, self.erosion_rate * 100)
      eroded_data = eroded_data[erosion_scores >= threshold]
    return eroded_data

  def _compute_erosion_scores(self, data):
    distances, _ = self.nbrs.kneighbors(data)
    return np.mean(distances, axis=1)

  def detect_anomalies(self):
    erosion_scores = self._compute_erosion_scores(self.data)
    threshold = np.percentile(erosion_scores, 100 - self.erosion_rate * 100)
    return self.data[erosion_scores < threshold]
