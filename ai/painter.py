import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore


class Painter:
    def __init__(self, load_csv, load_dir=None):
        # Inicializa os dados
        if not load_csv:
            self.data = pd.DataFrame(columns=["episode reward", "episode", "Method"])
        else:
            self.load_dir = load_dir
            if self.load_dir and os.path.exists(self.load_dir):
                print(f"==Reading {self.load_dir}.")
                self.data = pd.read_csv(self.load_dir).iloc[:, 1:]
                print("==Reading complete")
            else:
                print(
                    f"==There is no file in {self.load_dir}, "
                    "Painter has automatically created the csv."
                )
                self.data = pd.DataFrame(
                    columns=["episode reward", "episode", "Method"]
                )

        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self, label):
        self.xlabel = label

    def setYlabel(self, label):
        self.ylabel = label

    def setTitle(self, label):
        self.title = label

    def setHueOrder(self, order):
        """Define a ordem do agrupamento na visualização"""
        self.hue_order = order

    def addData(self, dataSeries, method, x=None, smooth=True):
        if smooth:
            dataSeries = self.smooth(dataSeries)

        size = len(dataSeries)
        if x is not None:
            if len(x) != size:
                raise ValueError("x must have the same length as dataSeries")

        # Criar o DataFrame de uma só vez para evitar concatenações desnecessárias
        new_data = pd.DataFrame(
            {
                "episode reward": dataSeries,
                "episode": x if x is not None else np.arange(1, size + 1),
                "Method": method,
            }
        )

        self.data = pd.concat([self.data, new_data], ignore_index=True)

    def drawFigure(
        self,
        style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "darkgrid",
    ):
        """
        style: darkgrid, whitegrid, dark, white, ticks
        """
        if style not in ["white", "dark", "whitegrid", "darkgrid", "ticks"]:
            raise ValueError(
                "Invalid style. Choose from 'white', 'dark', 'whitegrid', "
                "'darkgrid', 'ticks'."
            )
        sns.set_theme(style=style)
        sns.set_style(rc={"linewidth": 1})
        print("==Drawing...")
        sns.relplot(
            data=self.data,
            kind="line",
            x="episode",
            y="episode reward",
            hue="Method",
            hue_order=self.hue_order,
        )
        plt.title(self.title if self.title is not None else "", fontsize=12)
        plt.xlabel(self.xlabel if self.xlabel is not None else "")
        plt.ylabel(self.ylabel if self.ylabel is not None else "")
        print("==Finished drawing!")
        plt.show()

    def saveData(self, save_dir):
        self.data.to_csv(save_dir, index=False)
        print(f"==Data has been saved to path {save_dir}!")

    def addCsv(self, add_load_dir):
        """Merge another csv file into load_dir's csv file."""
        add_csv = pd.read_csv(add_load_dir).iloc[:, 1:]
        self.data = pd.concat([self.data, add_csv], axis=0, ignore_index=True)

    def deleteData(self, delete_data_name):
        """Delete data for a specific method."""
        self.data = self.data[~self.data["Method"].isin([delete_data_name])]
        print(f"==The corresponding data under {delete_data_name} has been deleted!")

    def smoothData(self, smooth_method_name, N):
        """Apply moving average (MA) smoothing for a specific method."""
        filtered_data = self.data[self.data["Method"] == smooth_method_name]
        if filtered_data.empty:
            print(f"==No data found for method: {smooth_method_name}")
            return

        self.data.loc[self.data["Method"] == smooth_method_name, "episode reward"] = (
            self.smooth(filtered_data["episode reward"].values, N)
        )
        print(f"==Finished smoothing {smooth_method_name} data with window size {N}!")

    @staticmethod
    def smooth(data, N=11):
        """Apply moving average smoothing."""
        if N < 2:
            return data

        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i < n:
                res[i] = np.mean(data[: i + n + 1])
            elif i >= len(data) - n:
                res[i] = np.mean(data[i - n :])
            else:
                res[i] = np.mean(data[i - n : i + n + 1])
        return res
