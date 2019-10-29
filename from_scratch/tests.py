import unittest

from mystats import avg

class TestAvg(unittest.TestCase):
    def test_list_avg(self):
        data = [1, 3, 5]
        result = avg(data)
        self.assertEqual(result, 3.0)

if __name__ == "__main__":
    unittest.main()