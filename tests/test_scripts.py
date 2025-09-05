import unittest
import sys
from os.path import dirname, join
from io import StringIO
import create_ngawest_dataset

`
class TestMyModule(unittest.TestCase):

    def test_nga_west2(self):
        # Save originals
        original_argv = sys.argv
        original_stdout = sys.stdout

        try:
            # Patch argv and stdout
            sys.argv = [
                join(dirname(__file__), "Metadata_Avail.csv"),
                dirname(__file__)
            ]
            sys.stdout = StringIO()

            # Run main
            create_ngawest_dataset.main()

            # Get printed output
            output = sys.stdout.getvalue()
            self.assertIn("foo", output)
            self.assertIn("bar", output)
        finally:
            # Restore originals
            sys.argv = original_argv
            sys.stdout = original_stdout

if __name__ == "__main__":
    unittest.main()
