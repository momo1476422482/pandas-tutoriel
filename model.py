import pandas as pd
from pathlib import Path
from typing import Dict
import numpy as np
from collections import Counter

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
            mean = self.df.query('Metier == "Data engineer"')['Experience'].mean(skipna=True)

            self.df.loc[(self.df['Metier']=='Data scientist'),'Experience']=self.df.loc[(self.df['Metier']=='Data scientist'),'Experience'].fillna(median)


            self.df.loc[(self.df['Metier']=='Data engineer'),'Experience']=self.df.loc[(self.df['Metier']=='Data engineer'),'Experience'].fillna(mean)




    def get_moy_experience(self) -> Dict[str,int]:
        df = self.df
        result = {
            "Data scientist": df[df["Metier"] == "Data scientist"].Experience.mean(skipna=True),
            "Data engineer": df[df["Metier"] == "Data engineer"].Experience.mean(skipna=True),
            "Data architect": df[df["Metier"] == "architecte"].Experience.mean(),
            "Lead data scientist": df[
                df["Metier"] == "Lead data scientist"
            ].Experience.mean(),
        }
        return result

    def get_most_used_technology(self):
        res=[]
        ee=self.df['Technologies'].str.split('/').to_numpy()
        for i in ee :
            res+=i

        print(Counter(res))



# ===============================================
if __name__ == '__main__':
    ac = analyse_csv(Path(__file__).parent / "data.csv")
    ac.impute_missing_values()
    print(ac.get_moy_experience())
    ac.get_most_used_technology()
