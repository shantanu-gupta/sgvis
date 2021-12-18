""" SGFamilyTree.py

    Sub-group family tree extraction-related code. No visualization is done
    here.
"""

import numpy as np
import pandas as pd
import itertools
import copy

class SGFamilyTree:
    def __init__(self, attrs_categories_map, selected=None):
        self.attrs_categories_map = attrs_categories_map
        self.num_attr = len(attrs_categories_map)
        self.set_selected(selected)

    def set_selected(self, selected):
        self.selected = selected
        self.parents = []
        self.siblings = []
        self.children = []
        self._set_relatives()
        return

    def _set_relatives(self):
        if self.selected is not None:
            # parents
            self.parents = []
            sel_attrs = self.selected.keys()
            for c in range(1, len(self.selected)):
                parents = []
                for attrs in itertools.combinations(sel_attrs, c):
                    parents.append({a: self.selected[a] for a in attrs})
                self.parents.append(parents)
            # siblings
            self.siblings = []
            for attr in sel_attrs:
                # replace this with one of the remaining possible values
                for v in self.attrs_categories_map[attr]:
                    if v == self.selected[attr]:
                        continue
                    else:
                        s = copy.deepcopy(self.selected)
                        s[attr] = v
                    self.siblings.append(s)
                # endfor
            # endfor

            # children
            self.children = []
            # set difference
            rem_attrs = self.attrs_categories_map.keys() - sel_attrs
            for c in range(1, 1 + self.num_attr - len(self.selected)):
                children = []
                for attrs in itertools.combinations(rem_attrs, c):
                    new_parts = {a: self.attrs_categories_map[a]
                                for a in attrs}
                    children.append(new_parts)
                self.children.append(children)
            # endfor
            return

