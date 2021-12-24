import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class analyse_csv:
    # ==========================================
    def __init__(self, path_data) -> None:
        self.df = pd.read_csv(path_data)
        self.df['Experience'] = self.df['Experience'].str.replace(',', '.').astype(float)

    # ==========================================
    def get_number_observation(self) -> int:
        return self.df.shape[0]

    # ==========================================
    def get_data_frame(self) -> pd.DataFrame:
        return self.df

    # ==========================================
    def impute_missing_values(self) -> None:
        if self.df["Experience"].isnull().values.any():
            median = self.df.query("Metier== 'Data scientist'")['Experience'].median(skipna=True)
            mean_data_eng = self.df.query('Metier == "Data engineer"')['Experience'].mean(skipna=True)
            mean_data_arch = self.df.query('Metier == "Data architecte"')['Experience'].mean(skipna=True)
            mean_lead_data = self.df.query('Metier == "Lead data scientist"')['Experience'].mean(skipna=True)
            self.df.loc[(self.df['Metier'] == 'Data scientist'), 'Experience'] = self.df.loc[
                (self.df['Metier'] == 'Data scientist'), 'Experience'].fillna(median)
            self.df.loc[(self.df['Metier'] == 'Data engineer'), 'Experience'] = self.df.loc[
                (self.df['Metier'] == 'Data engineer'), 'Experience'].fillna(mean_data_eng)
            self.df.loc[(self.df['Metier'] == 'Data architecte'), 'Experience'] = self.df.loc[
                (self.df['Metier'] == 'Data architecte'), 'Experience'].fillna(mean_data_arch)
            self.df.loc[(self.df['Metier'] == 'Lead data scientist'), 'Experience'] = self.df.loc[
                (self.df['Metier'] == 'Lead data scientist'), 'Experience'].fillna(mean_lead_data)

    # ============================================================================================
    def get_moy_experience(self) -> Dict[str, int]:
        df = self.df
        result = {
            "Data scientist": df[df["Metier"] == "Data scientist"].Experience.mean(skipna=True),
            "Data engineer": df[df["Metier"] == "Data engineer"].Experience.mean(skipna=True),
            "Data architect": df[df["Metier"] == "Data architecte"].Experience.mean(skipna=True),
            "Lead data scientist": df[
                df["Metier"] == "Lead data scientist"
                ].Experience.mean(skipna=True),
        }
        return result

    def plot_moy_experience(self):

        sns.boxplot(data=self.df, x='Metier', y='Experience')
        plt.savefig(str(Path(__file__).parent / 'moy_experience.png'))

    # ==========================================
    def get_most_used_technology(self) -> List[str]:
        res = []
        tech_list = self.df['Technologies'].str.split('/').to_numpy()
        for i in tech_list:
            res += i
        res_unique = np.unique(res)
        res_unique_count = np.zeros(res_unique.shape)
        index_res = 0
        for j in res_unique:
            res_unique_count[index_res] = res.count(j)
            index_res += 1
        return res_unique[np.argsort(res_unique_count)[::-1][0:5]]


# ===============================================
if __name__ == '__main__':
    ac = analyse_csv(Path(__file__).parent / "data.csv")
    ac.impute_missing_values()
    print(ac.get_moy_experience())
    ac.plot_moy_experience()
    print(ac.get_most_used_technology())
