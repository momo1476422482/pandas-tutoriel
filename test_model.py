import pandas as pd
from model import analyse_csv
from pathlib import Path


ac = analyse_csv(Path(__file__).parent / "data.csv")


def test_get_number_observation():
    assert ac.get_number_observation() == 9582


def test_impute_missing_values():
    ac.impute_missing_values()
    assert ac.get_data_frame().shape[0] == 9585
    assert ac.get_data_frame()["Experience"].isnull().values.any()
