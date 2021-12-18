#!/usr/bin/env python

import numpy as np
import pandas as pd
from SGVis import SGVis

RNG = np.random.default_rng(72)

def dummy_data(N, num_attr, num_cats):
    data = RNG.integers(0, high=num_cats, size=(N, num_attr))
    df = pd.DataFrame(data=data, columns=['A{}'.format(a)
                                        for a in range(1, 1+num_attr)])
    return df

NUM_ATTR = 10
NUM_CATS = 4
NUM_PTS = 100000
data = dummy_data(NUM_PTS, NUM_ATTR, NUM_CATS)
group_attrs = ['A1', 'A2', 'A3', 'A4']
# None means count is plotted
score_attr = None
sgv = SGVis(data, group_attrs_categories={ga: range(NUM_CATS)
                                            for ga in group_attrs},
                score_attr=score_attr)
sgv.plot_group_scores()

