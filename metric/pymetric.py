import numpy as np

class Metric:

    def __init__(self):
        self.metric = None

    def set_minkowski_metric(self) -> 'numpy array':
        """
        Returns the 4x4 minkowski metric with diagonal of {1, -1, -1, -1}
        """
        self.metric = np.multiply(np.identity(4), np.array([1, -1, -1, -1]))

        return self.metric

    def contract(self, a: 'np.array', b: 'numpy array', **kawgs) -> 'float':
        if 'type' in kawgs:
            if kawgs['type'] is 'transverse':
                return (a[1] * b[1] + a[2] * b[2])
            else:
                pass
        else:

            return float(np.dot(a, np.dot(self.metric, b)))