import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


class analyse_csv:
    # ==========================================
    def __init__(self, path_data) -> None:
        self.df = pd.read_csv(path_data)
        self.df["Experience"] = (
            self.df["Experience"].str.replace(",", ".").astype(float)
        )

    # ==========================================
    def get_number_observation(self) -> int:
        return self.df.shape[0]

    # ==========================================
    def get_data_frame(self) -> pd.DataFrame:
        return self.df

    # =============================================================================
    def impute_missing_values(self) -> None:
        if self.df["Experience"].isnull().values.any():
            median = self.df.query("Metier== 'Data scientist'")["Experience"].median(
                skipna=True
            )
            mean_data_eng = self.df.query('Metier == "Data engineer"')[
                "Experience"
            ].mean(skipna=True)
            mean_data_arch = self.df.query('Metier == "Data architecte"')[
                "Experience"
            ].mean(skipna=True)
            mean_lead_data = self.df.query('Metier == "Lead data scientist"')[
                "Experience"
            ].mean(skipna=True)
            self.df.loc[
                (self.df["Metier"] == "Data scientist"), "Experience"
            ] = self.df.loc[
                (self.df["Metier"] == "Data scientist"), "Experience"
            ].fillna(
                median
            )
            self.df.loc[
                (self.df["Metier"] == "Data engineer"), "Experience"
            ] = self.df.loc[
                (self.df["Metier"] == "Data engineer"), "Experience"
            ].fillna(
                mean_data_eng
            )
            self.df.loc[
                (self.df["Metier"] == "Data architecte"), "Experience"
            ] = self.df.loc[
                (self.df["Metier"] == "Data architecte"), "Experience"
            ].fillna(
                mean_data_arch
            )
            self.df.loc[
                (self.df["Metier"] == "Lead data scientist"), "Experience"
            ] = self.df.loc[
                (self.df["Metier"] == "Lead data scientist"), "Experience"
            ].fillna(
                mean_lead_data
            )

    # ============================================================================================
    def get_moy_experience(self) -> Dict[str, int]:
        df = self.df
        result = {
            "Data scientist": df[df["Metier"] == "Data scientist"].Experience.mean(
                skipna=True
            ),
            "Data engineer": df[df["Metier"] == "Data engineer"].Experience.mean(
                skipna=True
            ),
            "Data architect": df[df["Metier"] == "Data architecte"].Experience.mean(
                skipna=True
            ),
            "Lead data scientist": df[
                df["Metier"] == "Lead data scientist"
                ].Experience.mean(skipna=True),
        }
        return result

    # ============================================================================================
    def plot_moy_experience(self) -> None:
        sns.boxplot(data=self.df, x="Metier", y="Experience")
        plt.savefig(str(Path(__file__).parent / "moy_experience.png"))

    # ============================================================
    def get_most_used_technology(self, number: int) -> List[str]:
        res = []
        tech_list = self.df["Technologies"].str.split("/").to_numpy()
        for i in tech_list:
            res += i
        res_unique = np.unique(res)
        res_unique_count = np.zeros(res_unique.shape)
        index_res = 0
        for j in res_unique:
            res_unique_count[index_res] = res.count(j)
            index_res += 1
        return res_unique[np.argsort(res_unique_count)[::-1][0:number]].tolist()

    # ==========================================
    def transform_feature(self, list_tech: List[str]) -> None:
        list_tech = list(map(lambda x: x.lower(), list_tech))

        def transform_diplome(input_str: str) -> int:
            if input_str == "Bachelor":
                return 1
            elif input_str == "Master":
                return 2
            elif input_str == "Phd":
                return 3
            else:
                return 0

        def transform_techno(input_list: List[str]) -> np.ndarray:
            res = []
            input_list = list(map(lambda x: x.lower(), input_list))

            for li in list_tech:
                if input_list.count(li) > 0:
                    res += [1]
                else:
                    res += [0]

            return np.array(res)

        def transform_major(input_str: str) -> int:
            if input_str == "Data scientist":
                return 0
            elif input_str == "Data engineer":
                return 1
            elif input_str == "Data architecte":
                return 2
            else:
                return 3

        self.df['Diplome'] = self.df['Diplome'].transform(transform_diplome)
        self.df["Technologies"] = self.df["Technologies"].str.split("/").to_numpy()
        self.df["Technologies"] = self.df["Technologies"].transform(transform_techno)
        self.df['Metier_transforme'] = self.df['Metier'].transform(transform_major)

    # ==========================================
    @staticmethod
    def aggregate_features(df: pd.DataFrame) -> np.ndarray:
        return np.c_[np.vstack(df['Technologies']), df['Experience'].to_numpy().reshape(-1, 1), df[
            'Diplome'].to_numpy().reshape(-1, 1)]

    # ==========================================
    def run_cluster(self) -> None:
        features = self.aggregate_features(self.df)
        km = KMeans(n_clusters=2)
        km.fit(features)
        labels = km.labels_
        self.df['label_cluster'] = labels
        print(self.df.groupby('label_cluster').count())
        plt.figure()
        sns.scatterplot(data=self.df.head(100), x=self.df.head(100).index, y='Experience', hue='label_cluster')
        plt.savefig(Path(__file__).parent / 'cluster.png')

    # ==========================================
    def predict_major(self) -> None:
        df_without_na = self.df.dropna()
        df_na = self.df[self.df['Metier'].isna()]

        features_not_nan = self.aggregate_features(df_without_na)
        features_nan = self.aggregate_features(df_na)
        X_train, X_test, y_train, y_test = train_test_split(features_not_nan, df_without_na['Metier_transforme'],
                                                            test_size=0.25)
        # RandomForest
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=123456)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        self.df.loc[(self.df['Metier'].isna()), 'Metier'] = rf.predict(features_nan).reshape(-1, 1)

        print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
        print(f'Mean accuracy score: {accuracy:.3}')

        # Adaboost
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(f'Mean accuracy score: {accuracy:.3}')

        # ===============================================
    def __call__(self,path_result):
        self.impute_missing_values()
        print(ac.get_moy_experience())
        self.plot_moy_experience()
        print(self.get_most_used_technology(10))
        self.transform_fgieature(ac.get_most_used_technology(40))
        self.run_cluster()
        self.predict_major()
        self.df.to_csv(path_result)




# ===============================================
if __name__ == "__main__":
    ac = analyse_csv(Path(__file__).parent / "data.csv")
    ac(Path(__file__).parent / "result.csv")

