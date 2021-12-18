""" SGVis.py
"""

import itertools
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy

from SGFamilyTree import SGFamilyTree

class SGVis:
    UNSELECTED = 1
    SELECTED_PARENT = 2
    SELECTED_SIBLING = 3
    SELECTED_ANCHOR = 4
    SELECTED_CHILD = 5

    def __init__(self, data, group_attrs_categories=None, score_attr=None):
        self.data = data
        self.group_attrs_categories = group_attrs_categories
        self.num_attr = len(group_attrs_categories)
        # None means count is plotted
        self.score_attr = score_attr
        self.COUNT_FIELD_NAME = 'subgroup_count'
        assert self.COUNT_FIELD_NAME not in self.data.columns
        self._calc_group_scores()

        self.sel_sg = None
        self.sel_ftree = SGFamilyTree(group_attrs_categories)
        # Set up the initial "family" states for the subgroups
        self.SELECTED_STATE_NAME = 'selstate'
        assert self.SELECTED_STATE_NAME not in self.data.columns
        self._clear_family_states()

        # can do layout in constructor since this doesn't change with
        # interaction
        self.RNG = np.random.default_rng(27)
        self.SIGMA0 = 0.3
        self.GAP_FACTOR = 1.5
        self.base_markersize = 10
        self.detail_highlight_sizemult = 10
        self.color_normalize \
                = matplotlib.colors.Normalize(vmin=SGVis.UNSELECTED,
                                            vmax=SGVis.SELECTED_CHILD)
        self.SCORE_FIELD_NAME = (self.score_attr if self.score_attr is not None
                                                else self.COUNT_FIELD_NAME)
        self._setup_figure()
        self._setup_xy_for_plot() 

        self.logscale_score = False
        self.hover_annot = None

        self.SEL_ANN_BBOX_COLOR = '#10e040c0'
        self.HOV_ANN_BBOX_COLOR = '#10e040c0'

        return

    def _calc_group_scores(self):
        # form groups of all possible complexities
        # count is always calculated, and other scores if score_attr is set
        self.groups = []
        attrs = self.group_attrs_categories.keys()
        for c in range(1, 1+self.num_attr):
            groups_c = []
            for attrs_selected in itertools.combinations(attrs, c):
                g = self.data.groupby(list(attrs_selected))
                gs = g.size().to_frame(self.COUNT_FIELD_NAME)
                if self.score_attr is not None:
                    # assume we want the mean
                    # TODO: maybe expand to other aggregations later
                    gs = gs.join(g[self.score_attr].mean())
                # make sure that all theoretically possible combinations are
                # considered
                if c > 1:
                    cats = [self.group_attrs_categories[a]
                            for a in attrs_selected]
                    gs = gs.reindex(pd.MultiIndex.from_product(cats,
                                                        names=attrs_selected))
                else:
                    gs = gs.reindex(
                            pd.Index(
                                self.group_attrs_categories[attrs_selected[0]],
                                name=attrs_selected[0]))
                gs[self.COUNT_FIELD_NAME].fillna(0, inplace=True)
                # NaNs added when reindexing means that count also becomes float
                # but I restore int (for no great reason).
                gs = gs.astype({self.COUNT_FIELD_NAME: np.int64})
                groups_c.append(gs)
            self.groups.append(groups_c)
        # endfor
        return

    def _clear_family_states(self):
        for Gc in self.groups:
            for gs in Gc:
                if self.SELECTED_STATE_NAME in gs.columns:
                    gs.loc[:,self.SELECTED_STATE_NAME] = SGVis.UNSELECTED
                else:
                    gs[self.SELECTED_STATE_NAME] = SGVis.UNSELECTED
                # endif
            # endfor
        # endfor

    def _set_family_states(self):
        if self.sel_sg is not None:
            sel_level = len(self.sel_sg) - 1
            # set parents
            for c in range(sel_level):
                Gc = self.groups[c]
                Pc = self.sel_ftree.parents[c]
                for gs in Gc:
                    # have multiple parents (all subsets of chosen attributes)
                    for pc in Pc:
                        if all([name in pc for name in gs.index.names]):
                            # set SELECTED_PARENT state for the right one
                            # necessarily have one value for each attribute
                            if c == 0:
                                ind = pc[gs.index.names[0]]
                            else:
                                ind = tuple([pc[name]
                                            for name in gs.index.names])
                            gs.loc[ind, self.SELECTED_STATE_NAME] \
                                    = SGVis.SELECTED_PARENT
                        # endif
                    # endfor
                # endfor
            #endfor
            # set selected itself
            # and siblings
            Gc = self.groups[sel_level]
            for i, gs in enumerate(Gc):
                if all([name in self.sel_sg
                        for name in gs.index.names]):
                    # set SELECTED_ANCHOR state for the right one
                    # necessarily have one value for each attribute
                    ind = tuple([self.sel_sg[name]
                                    for name in gs.index.names])
                    gs.loc[ind, self.SELECTED_STATE_NAME] \
                            = SGVis.SELECTED_ANCHOR
                # endif
                for sc in self.sel_ftree.siblings:
                    if all([name in sc for name in gs.index.names]):
                        # set SELECTED_SIBLING for the right ones
                        ind = tuple([sc[name] for name in gs.index.names])
                        gs.loc[ind, self.SELECTED_STATE_NAME] \
                                = SGVis.SELECTED_SIBLING
                    # endif
                # endfor
            # endfor
            # set children
            for c in range(1+sel_level, self.num_attr):
                Gc = self.groups[c]
                Cc = self.sel_ftree.children[c-1-sel_level]
                for i, gs in enumerate(Gc):
                    # have multiple children (all combinations of chosen attributes)
                    for cc in Cc:
                        if all([((name in cc) 
                                    or (name in self.sel_sg))
                                for name in gs.index.names]):
                            # set SELECTED_CHILDREN state for the right ones
                            # children have many possible values for new
                            # attributes (not set by sel_sg)
                            #
                            # idea from https://stackoverflow.com/a/61335465
                            # make 1-element lists for sel_sg
                            d0 = {**{k: [v]
                                    for k, v in self.sel_sg.items()},
                                **cc}
                            k, v = zip(*d0.items())
                            pd0 = [dict(zip(k, vv))
                                    for vv in itertools.product(*v)]
                            inds = [tuple([pd[name] for name in gs.index.names])
                                    for pd in pd0]
                            gs.loc[inds, self.SELECTED_STATE_NAME] \
                                    = SGVis.SELECTED_CHILD
                        # endif
                    # endfor
                # endfor
            #endfor
        # endif
        return

    def _setup_figure(self):
        self.figure = plt.figure(constrained_layout=True)
        self.gridspec = GridSpec(self.num_attr, 2, figure=self.figure,
                                width_ratios=[1, 10])
        self.axes_mini = []
        for c in range(self.num_attr):
            self.axes_mini.append(self.figure.add_subplot(self.gridspec[c,0]))
        self.ax_detail = self.figure.add_subplot(self.gridspec[:,1])
        self.figure.canvas.mpl_connect('pick_event',
                                        self._select_subgroup_onpick)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                        self._show_subgroup_onhover)
        return

    def _setup_xy_for_plot(self):
        self.Xvals_mini = []
        self.Yvals_mini = []
        self.Xvals_detail = []
        self.Yvals_detail = []
        # this is needed for axis ticklabels later
        self.attrs_selected_names = []
        # just logging
        self.crowding_sigmas = []

        for c in range(len(self.groups)):
            gattrs = []
            Xvals = []
            Sigmas = []
            Yvals_mini = []
            Yvals_detail = []
            for i, g in enumerate(self.groups[c]):
                gattrs.append(g.index.name if c == 0 else g.index.names)
                xvals = g[self.SCORE_FIELD_NAME].to_numpy()
                sigma = (SGVis._sigma_kde(xvals) 
                            * (1 / len(self.groups[c]))
                            * self.SIGMA0)
                Xvals.append(xvals)
                Sigmas.append(sigma)

                yvals_mini = np.empty(xvals.shape)
                yvals_mini.fill(i)
                yvals_mini = yvals_mini + self.RNG.normal(scale=sigma)
                Yvals_mini.append(yvals_mini)

            xmin = np.min([x.min() for x in Xvals])
            xmax = np.max([x.max() for x in Xvals])
            Xvals_detail = [(x - xmin) / (xmax - xmin) for x in Xvals]
            
            ymin = np.min([y.min() for y in Yvals_mini])
            ymax = np.max([y.max() for y in Yvals_mini])
            # going up to down so num_attr - <>
            Yvals_detail = [(y - ymin) / (ymax - ymin) 
                            + self.GAP_FACTOR * (self.num_attr - c)
                            for y in Yvals_mini]

            self.Xvals_mini.append(Xvals)
            self.Xvals_detail.append(Xvals_detail)
            self.Yvals_mini.append(Yvals_mini)
            self.Yvals_detail.append(Yvals_detail)
            self.attrs_selected_names.append(gattrs)
            self.crowding_sigmas.append(Sigmas)
        # endfor
        return

    def _setup_sc_for_plot(self):
        self.sizes_detail = []
        self.colors_detail = []
        for c in range(len(self.groups)):
            Colors = []
            Sizes = []
            for i, g in enumerate(self.groups[c]):
                colors = g[self.SELECTED_STATE_NAME].to_numpy()
                Colors.append(colors)

                sizes = self.base_markersize * np.ones(colors.shape)
                sizes[colors != SGVis.UNSELECTED] \
                        *= self.detail_highlight_sizemult
                Sizes.append(sizes)
            # endfor
            self.colors_detail.append(Colors)
            self.sizes_detail.append(Sizes)
        # endfor
        return

    def _set_sel_sg(self, sg):
        self.sel_sg = sg
        self.sel_ftree.set_selected(sg)
        self._clear_family_states()
        self._set_family_states()
        return

    def _lookup_subgroup_index(self, sel_level, ind_data):
        # find subgroup corresponding to ind_data
        Gsel = self.groups[sel_level]
        n = 0
        for g in Gsel:
            if n + len(g) <= ind_data:
                n = n + len(g)
                continue
            else:
                num_idx = ind_data - n
                sel_attr_vals = g.index[num_idx]
                if sel_level > 0:
                    sel_sg = {name: val 
                            for name, val in zip(g.index.names,
                                                sel_attr_vals)}
                else:
                    sel_sg = {g.index.name: sel_attr_vals}
                break
            #endif
        #endfor
        return sel_sg
    
    def _select_subgroup_onpick(self, event):
        ind = event.ind[0]
        artist = event.artist
        xy = artist.get_offsets()
        xy = xy[ind]
        sel_level = self.detail_scatters.index(artist)
        ind_data = self.detail_sortorder[sel_level][ind]
        self._set_sel_sg(self._lookup_subgroup_index(sel_level, ind_data))
        self.sel_xy = xy
        self.plot_group_scores()
        return

    def _show_subgroup_onhover(self, event):
        # from https://stackoverflow.com/a/47166787
        if event.inaxes == self.ax_detail:
            for i, sc in enumerate(self.detail_scatters):
                order = self.detail_sortorder[i]
                cont, ind = sc.contains(event)
                if cont:
                    # re-create annotation
                    if self.hover_annot is not None:
                        self.hover_annot.remove()
                    self.hovered_xy = sc.get_offsets()[ind['ind'][0]]
                    self.hovered_xy = (self.hovered_xy[0],
                                        self.hovered_xy[1] - 0.2)
                    self.hover_sgs = []
                    for ind_d in ind['ind']:
                        ind_data = order[ind_d]
                        self.hover_sgs.append(
                                self._lookup_subgroup_index(
                                                i, ind_data))
                    annot_text = '\n'.join(
                                    [str(sg) for sg in self.hover_sgs])
                    self.hover_annot = self.ax_detail.annotate(
                                            annot_text,
                                            self.hovered_xy,
                                            bbox=dict(boxstyle='square',
                                                    lw=2,
                                                    fc=self.HOV_ANN_BBOX_COLOR))
                    break
            # this is the fancy for-else
            else:
                if self.hover_annot is not None:
                    self.hover_annot.remove()
                    self.hover_annot = None
                # endif
            # endfor
            plt.draw()
        # endif
        return

    def _plot_mini(self):
        for c in range(self.num_attr):
            X = np.array([x for xvals in self.Xvals_mini[c] for x in xvals])
            Y = np.array([y for yvals in self.Yvals_mini[c] for y in yvals])
            self.axes_mini[c].scatter(X, Y, s=self.base_markersize)
            self.axes_mini[c].set_prop_cycle(None)
            self.axes_mini[c].set_yticks(range(len(self.groups[c])))
            self.axes_mini[c].set_yticklabels(self.attrs_selected_names[c])
            if self.logscale_score:
                self.axes_mini[c].set_xscale('log')
            else:
                self.axes_mini[c].set_xlim(left=0)
        self.axes_mini[0].set_title('Global view')
        return

    def _plot_detail(self):
        self.detail_scatters = []
        self.detail_sortorder = []
        for c in range(self.num_attr):
            X = np.array([x for xvals in self.Xvals_detail[c] for x in xvals])
            Y = np.array([y for yvals in self.Yvals_detail[c] for y in yvals])
            S = np.array([s for svals in self.sizes_detail[c] for s in svals])
            C = np.array([color for cvals in self.colors_detail[c]
                                for color in cvals])
            if self.logscale_score:
                # 0 is a bad value for logscale
                X += 1e-3
            # order by C to make unselected draw first
            # idea from https://stackoverflow.com/a/55929839
            order = np.argsort(C)
            # storing because we will want it later
            self.detail_sortorder.append(order)

            sc = self.ax_detail.scatter(X[order], Y[order],
                                        s=S[order], c=C[order],
                                        norm=self.color_normalize,
                                        picker=True)
            self.detail_scatters.append(sc)
        self.ax_detail.set_title('Detailed view -- click to select a group')
        if self.logscale_score:
            self.ax_detail.set_xscale('log')
        self.ax_detail.tick_params('y', labelleft=False)
        xmin = 0 if not self.logscale_score else 1e-3
        self.ax_detail.set_xticks([xmin, 1])
        self.ax_detail.set_xticklabels(['min', 'max'])
        self.ax_detail.set_xlabel(self.SCORE_FIELD_NAME)
        return

    def _annotate_selected(self):
        if self.sel_sg is not None:
            # need x and y
            # i guess the user gives us that (the thing to do would be to set
            # sel_sg based on them, rather)
            ann_x, ann_y = self.sel_xy
            if ann_y % 1 < 0.5:
                ann_y = ann_y + 0.3
            else:
                ann_y = ann_y - 0.3
            self.ax_detail.annotate(str(self.sel_sg),
                                    (ann_x, ann_y),
                                    bbox=dict(boxstyle="square",lw=2,
                                                fc=self.SEL_ANN_BBOX_COLOR))
        # endif
        return

    def plot_group_scores(self):
        # size and color depend on selected subgroup
        self._setup_sc_for_plot()
        for ax in self.axes_mini:
            ax.clear()
        # endfor
        self.ax_detail.clear()
        self._plot_mini()
        self._plot_detail()
        self.figure.suptitle('{} variation across {}'\
                                .format(self.SCORE_FIELD_NAME,
                                        list(self.group_attrs_categories.keys())),
                            fontweight='bold')
        self._annotate_selected()
        plt.show()

    @staticmethod
    def _sigma_kde(v):
        h, edges = np.histogram(v, bins=5, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        f = interpolate.interp1d(centers, h, fill_value='extrapolate')
        hmax = h.max()
        p = np.maximum(0, np.minimum(hmax, f(v))) / hmax
        return p


