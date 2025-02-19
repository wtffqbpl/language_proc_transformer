#! coding: utf-8

import unittest
from unittest.mock import patch
from io import StringIO
import collections


class IntegrationTest(unittest.TestCase):
    # class collections.Counter([iterable-or-mapping])
    # The constructor of the counter can be called in any one of the following ways:
    #  1. With a sequence of items
    #  2. With a dictionary containing keys and counts
    #  3. With keyword arguments mapping strings names to counts

    def test_collections(self):
        # A counter is a container that stores elements as dictionary keys,
        # and their counts are stored as dictionary values.
        my_list = [1, 1, 2, 3, 4, 5, 3, 2, 3, 4, 2, 1, 2, 3]

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(collections.Counter(my_list))
            act_output = mock_stdout.getvalue().strip()

        expected_output = "Counter({2: 4, 3: 4, 1: 3, 4: 2, 5: 1})"
        self.assertEqual(expected_output.strip(), act_output)

        # Find the most common elements and their counts.
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(collections.Counter(my_list).most_common(2))
            act_output = mock_stdout.getvalue().strip()

        expected_output = "[(2, 4), (3, 4)]"
        self.assertEqual(expected_output.strip(), act_output)

        # Print items
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(collections.Counter(my_list).items())
            act_output = mock_stdout.getvalue().strip()

        expected_output = "dict_items([(1, 3), (2, 4), (3, 4), (4, 2), (5, 1)])"
        self.assertEqual(expected_output.strip(), act_output)

        # Print keys (the elements)
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(collections.Counter(my_list).keys())
            act_output = mock_stdout.getvalue().strip()

        expected_output = "dict_keys([1, 2, 3, 4, 5])"
        self.assertEqual(expected_output.strip(), act_output)

        # Print values (the counts)
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(collections.Counter(my_list).values())
            act_output = mock_stdout.getvalue().strip()

        expected_output = "dict_values([3, 4, 4, 2, 1])"
        self.assertEqual(expected_output.strip(), act_output)

    @staticmethod
    def print_counter(counter):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print(counter)
            return mock_stdout.getvalue().strip()

    def test_counter_2(self):
        # Creating a counter from a list
        ctr1 = collections.Counter([1, 2, 2, 3, 3, 3])
        act_output = self.print_counter(ctr1)
        expected_output = "Counter({3: 3, 2: 2, 1: 1})"
        self.assertEqual(expected_output.strip(), act_output)

        # Create a counter from a dictionary
        ctr2 = collections.Counter({1: 2, 2: 3, 3: 1})
        act_output = self.print_counter(ctr2)
        expected_output = "Counter({2: 3, 1: 2, 3: 1})"
        self.assertEqual(expected_output.strip(), act_output)

        # Creating a counter from a string
        ctr3 = collections.Counter('hello')
        act_output = self.print_counter(ctr3)
        expected_output = "Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})"
        self.assertEqual(expected_output.strip(), act_output)


if __name__ == "__main__":
    unittest.main(verbosity=True)
