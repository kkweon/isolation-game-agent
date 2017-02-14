import records
import unittest
import os

class TestRecords(unittest.TestCase):
    def tearDown(self):
        file_list = os.listdir('.')
        filename = 'test.csv'
        if filename in file_list:
            os.remove(filename)

    def test_can_save_csv(self):
        filename = 'test.csv'
        records.save(filename, [['Stduent', 0.9], ['IB_Improved', 0.8]])

        file_list = os.listdir('.')
        self.assertIn(filename, file_list)

    def test_can_match_data(self):
        filename = 'test.csv'
        records.save(filename, [['Student', 0.9], ['IB_Improved', 0.8]])
        with open(filename, 'r') as f:
            data = f.readlines()
        self.assertEqual(data[0].strip().split(','), ['name', 'win_ratio'])
        self.assertEqual(data[1].strip().split(','), ['Student', '0.9'])





if __name__ == '__main__':
    unittest.main()