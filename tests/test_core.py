import unittest
import pandas as pd
from spark_framework.core import calc_stat_local


class TestCore(unittest.TestCase):
    def test_calc_stats_local(self):
        pdf1 = pd.DataFrame([(1, 1.123), (1, 2.456)], columns=["id", "a"])
        print(pdf1)
        pdf2 = calc_stat_local(pdf1, ["mean", "std", "skew"], [0.5], {"a": "astat_"}, "id", a_return_uppercase=False, a_reset_index=True,
                               a_round_to_decimal_places=2)
        print(pdf2)
