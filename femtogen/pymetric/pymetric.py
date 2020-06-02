import numpy as np

class Metric:

    def __init__(self):
        self.metric = None

    def set_minkowski_metric(self) -> 'numpy array':
        """
        Returns the 4x4 minkowski pymetric with diagonal of {1, -1, -1, -1}
        """
        self.metric = np.multiply(np.identity(4), np.array([1, -1, -1, -1]))

        return self.metric

    def contract(self, a: 'np.array', b: 'numpy array', **kawgs) -> 'float':
        '''

        :param a: array one
        :param b: array two
        :param kawgs: qualifies type of contraction. Default contraction is four-vector. Optional contraction of
        three vector in the case of transverse.

        :return: scalar value of contraction.
        '''
        try:
            assert a.size == b.size

            if 'type' in kawgs:
                if kawgs['type'] is 'transverse':
                    return (a[1] * b[1] + a[2] * b[2])
                else:
                    pass
            else:

                return float(np.dot(a, np.dot(self.metric, b)))
        except AssertionError:
            print('Array lengths are required to be equal.\n  a.size = {0}\n  b.size = {1}'.format(a.size, b.size))
        except Exception as ex:
            print('Error found in array contraction: {exception}'.format(exception=ex))