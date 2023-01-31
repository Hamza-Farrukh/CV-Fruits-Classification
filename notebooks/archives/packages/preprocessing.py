import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class Preprocessing:
    """Class for preprocessing"""

    def __init__(self, size=(100, 100)):
        self.size = size

    def get_size(self, df):
        sizes = []
        for i in df:
            img = plt.imread(i)
            sizes.append(img.shape)
        return sizes

    def image_resize(self, df):
        new_df = []
        for i in df:
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size[0], self.size[1]))
            new_path = i.replace('raw', 'processed')
            if os.path.exists(new_path[0:new_path.rfind('/')]) is False:
                os.mkdir(new_path[0:new_path.rfind('/')])
            plt.imsave(new_path, img)
            new_df.append(new_path)

        return pd.DataFrame({'Image': new_df})

    def fit(self, df):
        df['Image'] = self.image_resize(df['Image'])

        sizes = self.get_size(df['Image'])

        widths = [i[0] for i in sizes]
        heights = [i[1] for i in sizes]
        df['Width'] = widths
        df['Height'] = heights

        return df
