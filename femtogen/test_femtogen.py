import unittest

from femtogen.femtogen import FemtoGen

class MyTestCase(unittest.TestCase):
    '''
    Super simple unit test to be expanded on at a later time.
    '''
    femto = FemtoGen()

    def test_generate_cross_section(self):
        print('Testing Femtogen.generate_cross_section() ....')

        # Check that exception is properly raised on bad cross section type
        self.assertRaises(Exception, self.femto.generate_cross_section(0, error=None, type='wrong_type'))

    def test_read_data_file(self):
        # Check that invalid file
        print('Testing Femtogen.read_data_file() ....')
        self.assertRaises(AssertionError, self.femto.read_data_file,'random_file_that_doesnt_exist.csv')


if __name__ == '__main__':
    unittest.main()
