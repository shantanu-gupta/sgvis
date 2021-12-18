#!/usr/bin/env python

import numpy as np
import pandas as pd
from SGVis import SGVis

RNG = np.random.default_rng(72)

DATA_PATH = './data/atus_data.csv'
data = pd.read_csv(DATA_PATH)

data = data[~data['FAMINCOME'].isin([996, 997, 998])]
# group FAMINCOME further
i1 = data['FAMINCOME'].isin([1, 2, 3, 4, 5, 6, 7, 8])
i2 = data['FAMINCOME'].isin([9, 10, 11])
i3 = data['FAMINCOME'].isin([12, 13, 14])
i4 = data['FAMINCOME'].isin([15, 16])
data['FAMINCOME2'] = 0
data.loc[i1, 'FAMINCOME2'] = 1
data.loc[i2, 'FAMINCOME2'] = 2
data.loc[i3, 'FAMINCOME2'] = 3
data.loc[i4, 'FAMINCOME2'] = 4

data = data[data['EMPSTAT'] != 99]
# group EMPSTAT
i1 = data['EMPSTAT'].isin([1, 2])
i2 = data['EMPSTAT'].isin([3, 4])
i3 = data['EMPSTAT'].isin([5])
data['EMPSTAT2'] = 0
data.loc[i1, 'EMPSTAT2'] = 1
data.loc[i2, 'EMPSTAT2'] = 2
data.loc[i3, 'EMPSTAT2'] = 3

# None means count is plotted
score_attr = None
attr_cat_map = {'SEX': [1, 2],
                'REGION': [1, 2, 3, 4],
                'FAMINCOME2': [1, 2, 3, 4],
                'EMPSTAT2': [1, 2, 3]}
sgv = SGVis(data, group_attrs_categories=attr_cat_map,
            score_attr=score_attr)
sgv.logscale_score = True
sgv.plot_group_scores()

