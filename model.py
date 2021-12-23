import pandas as pd
from pathlib import Path
from typing import Dict


class analyse_csv:
    def __init__(self, path_data) -> None:
        self.df = pd.read_csv(path_data)
        self.df['Experience']=self.df['Experience'].str.replace(',', '.').astype(float)





    def get_number_observation(self) -> int:
        return self.df.shape[0]

    def get_data_frame(self) -> pd.DataFrame:
        return self.df

    def impute_missing_values(self) -> None:
        if self.df["Experience"].isnull().values.any():
            median= self.df.query("Metier== 'Data scientist'")['Experience'].median(skipna=True)
            mean = self.df.query('Metier == "Data engineer"')['Experience'].mean()
            print('median',median)
            self.df[['Experience']]=self.df.query("Metier== 'Data scientist'")[['Experience']].fillna(median,inplace=True)
            print(self.df)
            self.df.query('Metier == "Data engineer"').Experience.fillna(mean)



    """
    def get_moy_experience(self, path_data: Path) -> Dict[str:int]:
        df = self.load_data(path_data)
        result = {
            "Data scientist": df[df["Metier"] == "Data scientist"].Experience.mean(),
            "Data engineer": df[df["Metier"] == "Data engineers"].Experience.mean(),
            "Data architecte": df[df["Metier"] == "architecte"].Experience.mean(),
            "Lead data scientist": df[
                df["Metier"] == "Lead data scientist"
            ].Experience.mean(),
        }
        return result
        """


# ===============================================
if __name__ == '__main__':
    ac = analyse_csv(Path(__file__).parent / "data.csv")
    ac.impute_missing_values()
