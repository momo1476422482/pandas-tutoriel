import pandas as pd
from model import analyse_csv
from pathlib import Path

ac = analyse_csv(Path(__file__).parent / "data.csv")


def test_get_number_observation():
    assert ac.get_number_observation() == 9582


def test_impute_missing_values():
    ac.impute_missing_values()
    df = ac.get_data_frame()
    assert df.loc[(df["Experience"].isna())].shape[0] == 0


def test_predict_major():
    ac.impute_missing_values()
    ac.transform_feature(ac.get_most_used_technology(40))
    ac.predict_major()
    df = ac.get_data_frame()
    assert df.loc[(df["Metier"].isna())].shape[0] == 0
