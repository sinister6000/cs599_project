# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

__metaclass__ = type

import codecs
import webbrowser
import os
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import prettyplotlib as ppl
from gensim import matutils, corpora, models
import numpy as np
from sklearn.manifold import MDS
import pyLDAvis
import pyLDAvis.gensim as pg

from prepdata.myCorpus import MyCorpus, tokenize
import prepdata.sqliteQueries as sq


_SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64
ISO2DAY = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
FL = u'░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓▓▓█████████'
FR = u'█████████▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░'
HR = (FL + FR)*2


class LdaDriver(object):
    """
    Class to create corpora, LDA models, and perform queries on the models.

    Keeps track of all model parameters,
    """
    def __init__(self, **kwargs):
        """
        Initialize LdaDriver object.

        LdaDriver object has a corpus and lda model either created here or loaded from file.

        :return: self
        :rtype: LdaDriver
        """
        self.corpus_type = kwargs['corpus_type']
        self.num_topics = kwargs['num_topics']
        self.num_passes = kwargs['num_passes']
        self.alpha = kwargs['alpha']

        # if make_corpus flag is True, create a new corpus from files in ../../data/ven_
        if kwargs['make_corpus']:
            self.cor = MyCorpus('../../data/ven_', self.corpus_type)
            self.cor.dictionary.save('../../data/save/driver/{}_dictionary.dict'.format(self.corpus_type))
            # save new corpus to file
            corpora.MmCorpus.serialize('../../data/save/driver/{}_corpus.mm'.format(self.corpus_type),
                                       self.cor,
                                       id2word=self.cor.dictionary,
                                       index_fname='../../data/save/driver/{}_corpus.mm.index'.format(self.corpus_type),
                                       progress_cnt=1000)

        # If make_lda flag is True, train a new LDA.
        if kwargs['make_lda']:
            if self.alpha is 'auto':
                self.lda = models.LdaModel(self.cor,
                                           num_topics=self.num_topics,
                                           id2word=self.cor.dictionary,
                                           passes=self.num_passes,
                                           alpha=self.alpha,
                                           eval_every=10,
                                           iterations=50)
            # symmetric alpha uses multicore algorithm, much faster
            elif self.alpha is 'symmetric':
                self.lda = models.LdaMulticore(self.cor,
                                               num_topics=self.num_topics,
                                               id2word=self.cor.dictionary,
                                               passes=self.num_passes,
                                               alpha=self.alpha,
                                               batch=True,
                                               eval_every=10,
                                               iterations=50)

            # Save LDA model
            self.lda.save('../../data/save/driver/{}_lda_{}t_{}p_{}.model'
                          .format(self.corpus_type, self.num_topics, self.num_passes, self.alpha))

        # load corpus from file
        self.cor = corpora.MmCorpus('../../data/save/driver/{}_corpus.mm'.format(self.corpus_type))
        self.cor.dictionary = corpora.Dictionary.load('../../data/save/driver/{}_dictionary.dict'.format(self.corpus_type))

        # Load LDA model
        self.lda = models.LdaMulticore.load('../../data/save/driver/{}_lda_{}t_{}p_{}.model'
                                            .format(self.corpus_type, self.num_topics, self.num_passes, self.alpha))

        # Load venue index
        self.ven_id2i = {}
        with codecs.open('../../data/ven_id2i.txt', 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.split()
                self.ven_id2i[line[0]] = int(line[1])

        # Load venues for comparison
        self.vens = sq.topn_venues(1200)
        self.dist_matrix = self.compare_venues(self.vens[:30])

        ven_offsets = [self.ven_id2i[ve.id] for ve in self.vens]
        ven_p_dists = [self.lda.get_document_topics(self.cor[i]) for i in ven_offsets]
        self.ven_p_dists_dense = matutils.corpus2dense(ven_p_dists, int(self.num_topics), int(len(ven_offsets)))


    def __str__(self):
        """
        Print out settings info.

        :return: str repr of the LdaDriver
        :rtype: str
        """
        out = u'\nCorpus:\n'
        out += u'   Tokenizer: {}\n'.format(self.corpus_type)
        out += u'   # of Shouts: {}\n'.format(sq.checkin_count())
        out += u'   # of Venues: {}\n'.format(sq.venue_count())
        out += u'   # of Users: {}\n'.format(sq.user_count())

        out += u'\nTopic Model: LDA\n'
        out += u'   # of Topics: {}\n'.format(self.num_topics)
        out += u'   # of Passes: {}\n'.format(self.num_passes)
        out += u'   Alpha: {}\n'.format(self.alpha)
        return out


    def nn(self, vec, n=10):
        """
        Gets the n nearest neighbors to vec.

        :param vec: topic distribution representation of a text item
        :type vec: gensim representation [(topic_id, prob)]
        :param n: how many neighbors to return
        :type n: int
        :return: list of 2-tuples (venue, distance)
        :rtype: [(venue, distance)]
        """
        td_dense = matutils.sparse2full(vec, self.num_topics)
        td_distances = [(i, hellinger_distance(td_dense, ven_dense))
                        for (i, ven_dense) in enumerate(self.ven_p_dists_dense.T)]
        nn = sorted(td_distances, key=lambda x: x[1])[:n]
        return [(self.vens[neighbor_index], distance) for (neighbor_index, distance) in nn]


    def print_nn(self, vec, n=10):
        """
        Prints n nearest neighbors to vec.

        :param vec: topic distribution representation of a text item
        :type vec: gensim representation [(topic_id, prob)]
        :param n: how many neighbors to return
        :type n: int
        :return: None
        :rtype: None
        """
        neighbors = self.nn(vec, n)
        print(u'  {} nearest neighbors'.format(n))
        for (venue, distance) in neighbors:
            print(u'    {:1.3f}  {}'.format(distance, venue.name))
        print u'\n', HR, u'\n'

    def print_topics(self, n=10):
        """
        Prints the top n words and probabilities from each topic.

        :param n: number of words to print per topic
        :type n: int
        :return: None
        :rtype: None
        """
        print(u'(¯`·._.·(¯`·._.· {}, {} topics, {} passes, {} ·._.·´¯)·._.·´¯)'
              .format(self.corpus_type, self.num_topics, self.num_passes, self.alpha))
        for i, topic in enumerate(self.lda.show_topics(self.num_topics, n, log=True, formatted=False)):
            print(u'\nTOPIC {0}:'.format(i))
            for (prob, word) in topic:
                print(u'   {:1.5f}   {}'.format(prob, word))
        print u'\n', HR, u'\n'

    def compare_venues(self, venues):
        """
        Calculates distance matrix for venues.

        :param venues: venues to compare
        :type venues: [Venue database objects]
        :return: distance matrix
        :rtype: 2d np.ndarray
        """
        ven_offsets = [int(self.ven_id2i[v.id]) for v in venues]
        ven_p_dists = [self.lda.get_document_topics(self.cor[i]) for i in ven_offsets]
        ven_p_dists_dense = matutils.corpus2dense(ven_p_dists, int(self.num_topics), int(len(ven_offsets)))
        return hellinger_matrix(ven_p_dists_dense)

    def docbows_to_hellinger_matrix(self, bow_corpus):
        """
        Creates distance matrix from docbows in bow_corpus.

        :param bow_corpus: set of documents in BOW representation
        :type bow_corpus: list of BOWs
        :return: matrix of Hellinger distance measures. Mat(i,j) = dist(doc i, doc j)
        :rtype: 2d array of numpy.float64
        """
        size = len(bow_corpus)
        lda_cor = [self.lda.get_document_topics(doc_bow) for doc_bow in bow_corpus]
        lda_cor_matrix = matutils.corpus2dense(lda_cor, self.num_topics, size)
        return hellinger_matrix(lda_cor_matrix)


    def print_dist_matrix(self):
        """
        Prints venue distance matrix.

        :return: None
        :rtype: None
        """
        print(u"""~-'`'-.,.-'`'.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'~""")
        print(u'                Distance Matrix for Venues')
        print(u"""~'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.,.-'`'-.~""")
        ven_list = [self.ven_id2i[ven.id] for ven in self.vens]
        print('       | ' + '  '.join('{:<5}'.format(num) for num in ven_list))
        print('-------+' + '----------'*len(ven_list))

        for i, row in enumerate(self.dist_matrix):
            print ('{:>6} | '.format(ven_list[i]) + '  '.join('{:>1.3f}'.format(val) for val in row[:i + 1]))
        print u'\n', HR, u'\n'


    def temporal_weekday_single_ven(self, ven_id):
        """
        Splits ven_id up into bins, returns bins in bow format.

        :param ven_id: ID of venue to split
        :type ven_id: str
        :return: 2-tuple (mean distance, standard deviation)
        :rtype: 2-tuple (float, float)
        """
        split_ven_topics = {}
        ven_weekdays = sq.split_weekdays(ven_id)

        # get inferred topic distribution for each split bin
        for iso_day in range(1, 8):
            try:
                # tokenize shouts, making a list of tokens for each split
                word_list = []
                for ven_shout in ven_weekdays[iso_day]:
                    word_list.extend(tokenize(ven_shout, self.corpus_type))
                # turn list of tokens into BOW format
                bow = self.cor.dictionary.doc2bow(word_list)
                # infer topic distribution for the split, store in split_ven_topics{}
                split_ven_topics[iso_day] = self.lda[bow]
            except KeyError:
                split_ven_topics[iso_day] = []

        ven_name = (sq.get_ven_by_id(ven_id)).name
        ven_names = [u'{} ({})'.format(ven_name, ISO2DAY[iso_day]) for iso_day in range(1, 8)] + [ven_name]

        # make np.array of Hellinger distances between venue and splits
        distances = []
        # get dense vector representation of venue
        ven_vec = self.lda.get_document_topics(self.cor[self.ven_id2i[ven_id]])
        ven_vec_dense = matutils.sparse2full(ven_vec, self.num_topics)
        # distances[0] = venue vs. venue
        distances.append(hellinger_distance(ven_vec_dense, ven_vec_dense))

        # print nearest neighbors for venue
        print(u'{} - nearest neighbors:'.format(ven_name))
        self.print_nn(ven_vec)

        # distances[i] = venue vs. iso_day i
        for key in range(1, 8):
            split_vec = split_ven_topics[key]
            split_vec_dense = matutils.sparse2full(split_vec, self.num_topics)
            distances.append(hellinger_distance(ven_vec_dense, split_vec_dense))
            # print nearest neighbors for split
            print(u'{} ({}) nearest neighbors:'.format(ven_name, ISO2DAY[key]))
            self.print_nn(split_vec)
        # convert distances into numpy array
        distances = np.asarray(distances)
        self.vis_time_bars(distances, ven_name)

        # mean distance and SD
        dists = distances[1:]
        return np.mean(dists), np.std(dists)


    @staticmethod
    def vis_heatmap(dmatrix, ven_names, fout=None):
        """
        Create heatmap from a distance matrix.

        Saves as pdf if fout is given, otherwise displays graph.

        :param dmatrix: distance matrix
        :type dmatrix: 2d np.ndarray
        :param ven_names: names of venues in matrix
        :type ven_names: [str]
        :param fout: save graph to file
        :type fout: str
        :return: None
        :rtype: None
        """
        data = dmatrix
        labels = ven_names

        with plt.style.context('ggplot'):
            # setup plot figure
            fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
            ax.grid(color='white', linestyle='solid')
            fig = plt.gcf()
            fig.set_size_inches(10, 8, forward=True)
            fig.subplots_adjust(top=0.63, bottom=0.03, left=0.30, right=0.93)

            # set the colormap & norm
            norm = mplcolors.Normalize(vmin=0.0, vmax=1.0)
            cmap = plt.get_cmap('bone')

            # plot heatmap
            heatmap = ax.pcolor(data, cmap=cmap, norm=norm, edgecolor='gray')

            # turn off the frame
            ax.set_frame_on(False)

            # set axes ticks/labels
            ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
            ax.set_xticklabels(labels, minor=False, size='x-small', color='black')
            ax.set_yticklabels(labels, minor=False, size='x-small', color='black')

            # flip axes
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            # rotate xlabels
            plt.xticks(rotation=90)
            ax.grid(False)

            # turn off ticks
            ax = plt.gca()
            for t in ax.xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in ax.yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False

            # draw colorbar
            cb1 = fig.colorbar(heatmap)
            cb1.set_label('Distance')
            # fig.subplots_adjust(top=0.70, bottom=0.03, left=0.22, right=0.88)

            if fout is None:
                plt.show()
            else:
                fig.savefig(fout)

    def vis_MDS(self, dmatrix, ven_names, fout=None):
        """
        Displays MDS graph of venues.

        :param dmatrix: distance matrix
        :type dmatrix: numpy.ndarray
        :param ven_names: names of venues in matrix
        :type ven_names: list
        :param fout: save graph to file
        :type fout: str
        :return: None
        :rtype: None
        """
        # setup plot figure
        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
            ax.grid(color='white', linestyle='solid', linewidth=2)
            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches((8.0, 8.0), forward=True)
            plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95)
            plt.gca().grid(True)
            plt.axis([-1.0, 1.0, -1.0, 1.0])

            ttl = plt.title('MDS: Top 30 Venues')

            # get MDS coordinates for venues
            myMDS = MDS(2, verbose=0, n_jobs=-1, dissimilarity='precomputed')
            myMDS.fit(dmatrix)
            points = myMDS.embedding_

            # add column to points to hold venue categories
            points = np.c_[points, np.zeros((len(points), 2))]

            # create high-level categories and manually categorize top 30 venues
            # TODO: use Foursquare's categories json to get higher-level categories for venues
            airports = [0, 1, 5, 6, 11, 12, 28]
            conventions = [15, 18, 19, 29]
            theme_parks = [2, 8, 17]
            stadiums = [3, 4, 9, 10, 14, 24]
            others = [7, 13, 16, 20, 21, 22, 23, 25, 26, 27]

            for i, point in enumerate(points):
                point[3] = 20 * np.sqrt(self.vens[i].shout_count)

            airports_pts = np.stack([points[ind] for ind in airports])
            conventions_pts = np.stack([points[ind] for ind in conventions])
            theme_parks_pts = np.stack([points[ind] for ind in theme_parks])
            stadiums_pts = np.stack([points[ind] for ind in stadiums])
            others_pts = np.stack([points[ind] for ind in others])

            air = plt.scatter(airports_pts[:, 0], airports_pts[:, 1], marker='o', color='khaki',
                              s=airports_pts[:, 3], edgecolor='#303030', linewidth=0.5, alpha=0.6)
            con = plt.scatter(conventions_pts[:, 0], conventions_pts[:, 1], marker='o', color='cornflowerblue',
                              s=conventions_pts[:, 3], edgecolor='#303030', linewidth=0.5, alpha=0.6)
            theme = plt.scatter(theme_parks_pts[:, 0], theme_parks_pts[:, 1], marker='o', color='orangered',
                                s=theme_parks_pts[:, 3], edgecolor='#303030', linewidth=0.5, alpha=0.6)
            sta = plt.scatter(stadiums_pts[:, 0], stadiums_pts[:, 1], marker='o', color='gray', s=stadiums_pts[:, 3],
                              edgecolor='#303030', linewidth=0.5, alpha=0.6)
            oth = plt.scatter(others_pts[:, 0], others_pts[:, 1], marker='o', color='oldlace', s=others_pts[:, 3],
                              edgecolor='#303030', linewidth=0.5, alpha=0.6)

            # make legend
            legend = plt.legend((air, con, theme, sta, oth),
                                ('Airports', 'Conventions', 'Theme Parks', 'Stadiums', 'Other'),
                                scatterpoints=1, loc='lower left', ncol=5, fontsize=8, borderpad=1.5,
                                borderaxespad=1, shadow=True, labelspacing=1.7)
            frame = legend.get_frame()
            frame.set_facecolor('#FFFAFA')
            frame.set_edgecolor('#909090')

            # make labels as annotations
            for label, x, y in zip(ven_names, points[:, 0], points[:, 1]):
                plt.annotate(
                    label,
                    xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    size='xx-small')

            # adjust tick labels
            plt.tick_params(axis='both', which='major', labelsize=6, color='gray')
            plt.tick_params(axis='both', which='minor', labelsize=6, color='gray')

            # turn off ticks
            ax = plt.gca()
            for t in ax.xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in ax.yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False

            if fout is None:
                plt.show()
            else:
                fig.savefig(fout)

    def vis_super_MDS(self, dmatrix, fout=None):
        """
        Displays MDS graph of lots of venues (600)

        :param dmatrix: distance matrix
        :type dmatrix: numpy.ndarray
        :param fout: save graph to file
        :type fout: str
        :return: None
        :rtype: None
        """
        with plt.style.context('ggplot'):
            # setup plot figure
            fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
            ax.grid(color='white', linestyle='solid', linewidth=2)
            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches((8.0, 8.0), forward=True)
            plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95)
            plt.gca().grid(True)
            plt.axis([-1.0, 1.0, -1.0, 1.0])

            ttl = plt.title('MDS: Top 600 Venues')
            plt.xlabel('Click on dots for more info.\nDot size depends on total # of shouts.')

            # get MDS coordinates for venues
            myMDS = MDS(2, verbose=0, n_jobs=-1, dissimilarity='precomputed')
            myMDS.fit(dmatrix[:600, :600])
            points = myMDS.embedding_

            # add column to points to hold venue categories
            # TODO: higher-level categories from Foursquare's categories json
            points = np.c_[points, np.zeros(len(points))]
            categories = sq.get_categories(9)
            cat2num = {}
            for ind, cat in enumerate(categories):
                cat2num[cat] = ind
            for i, point in enumerate(points):
                point[2] = cat2num[self.vens[i].cat_name]

            # plot the points
            x_vals = points[:, 0]
            y_vals = points[:, 1]
            annotes = [u'{name}\n{category}'.format(name=v.name, category=v.cat_name) for v in self.vens]

            sizes = [(20 * np.sqrt(ven.shout_count)) for ven in self.vens]

            plt.scatter(x_vals, y_vals, marker='o', facecolor=points[:, 2], s=sizes,
                        cmap=plt.get_cmap('flag'), edgecolor='#303030', linewidth=0.5, alpha=0.4)

            af = AnnoteFinder(x_vals, y_vals, annotes)
            plt.connect('button_press_event', af)

            # adjust tick labels
            plt.tick_params(axis='both', which='major', labelsize=7, color='gray')
            plt.tick_params(axis='both', which='minor', labelsize=7, color='gray')

            # turn off ticks
            ax = plt.gca()
            for t in ax.xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in ax.yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False

            if fout is None:
                plt.show()
            else:
                fig.savefig(fout)


    def vis_super_cat_MDS(self, n=5, fout=None):
        """
        Displays MDS graph of venues in the top n categories

        :param n: number of top categories to include
        :type n: int
        :param fout: save graph to file
        :type fout: str
        :return: None
        :rtype: None
        """
        category_list, venue_list = sq.venues_from_top_n_categories(200/n, n)
        ven_dmatrix = self.compare_venues(venue_list)

        # make dict to assign numbers to category names for coloring in graph
        category_name2num = {}
        for i, category_name in enumerate(category_list):
            category_name2num[category_name] = i

        with plt.style.context('ggplot'):
            # setup plot figure
            fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
            ax.grid(color='white', linestyle='solid', linewidth=2)
            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches((8.0, 8.0), forward=True)
            plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95)
            plt.gca().grid(True)
            plt.axis([-1.0, 1.0, -1.0, 1.0])

            ttl = plt.title('MDS: Venues from Top {} Categories'.format(n))
            plt.xlabel('Click on dots to toggle labels.\nDot size depends on total # of shouts.')

            # get MDS coordinates for venues
            myMDS = MDS(2, verbose=0, n_jobs=-1, dissimilarity='precomputed')
            myMDS.fit(ven_dmatrix)
            points = myMDS.embedding_

            # add column to points to hold venue categories
            # TODO: higher-level categories from Foursquare's categories json
            points = np.c_[points, np.zeros(len(points))]

            # for each venue, normalize category number between 0-1 and store in new column.
            for i, point in enumerate(points):
                point[2] = float(category_name2num[venue_list[i].cat_name]) / len(category_list)

            # plot the points
            x_vals = points[:, 0]
            y_vals = points[:, 1]
            point_colors = points[:, 2]
            sizes = [(30*np.sqrt(ven.shout_count)) for ven in venue_list]
            annotes = [u'{name}\n{category}'.format(name=v.name, category=v.cat_name) for v in venue_list]

            my_plot = plt.scatter(x_vals, y_vals, marker='o', facecolor=point_colors, s=sizes,
                                  cmap=plt.get_cmap('prism'), edgecolor='#303030', linewidth=0.5, alpha=0.45)

            # connect mouse-click action
            af = AnnoteFinder(x_vals, y_vals, annotes)
            plt.connect('button_press_event', af)

            # adjust tick labels
            plt.tick_params(axis='both', which='major', labelsize=7, color='gray')
            plt.tick_params(axis='both', which='minor', labelsize=7, color='gray')

            # turn off ticks
            ax = plt.gca()
            for t in ax.xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in ax.yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False

            if fout is None:
                plt.show()
            else:
                fig.savefig(fout)


    def vis_ldavis(self):
        """
        Produces LDAvis visualization.

        Opens a web browser page with javascript topic viewer.
        """
        lda_vis_data = pg.prepare(self.lda, self.cor, self.cor.dictionary)
        pyLDAvis.save_html(lda_vis_data, '../../data/ldavis.html')
        vis_path = os.path.realpath('../../data/ldavis.html')
        webbrowser.open('file://{}'.format(vis_path), new=2)

    @staticmethod
    def vis_time_bars(ven_vs_split, ven_name, fout=None):
        """
        Displays a bar chart representing Hellinger distances between venue and temporal subsets.

        :param ven_vs_split: array of hellinger distances between venue and its splits
        :type ven_vs_split: numpy.ndarray
        :param ven_name: name of venue
        :type ven_name: str
        :param fout: save graph to file
        :type fout: str
        :return: None
        :rtype: None
        """
        # TODO: make more flexible to take in all sorts of temporal slices, not just weekdays.
        xtlabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # x-axis values
        x = range(7)
        # y-axis values
        y = ven_vs_split[1:]

        # create graph
        fig, ax = plt.subplots(1)
        ppl.bar(ax, x, y, xticklabels=xtlabels, facecolor='lightslategrey')
        ax.yaxis.grid(True, linestyle='-', linewidth=1.5, c='white', zorder=3)

        # graph settings
        fig.patch.set_facecolor('white')
        plt.ylim(0, 1)
        plt.ylabel("Hellinger Distance", fontsize=16)

        # spines, axes, ticks
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('darkslategrey')
        ax.spines['left'].set_zorder(4)
        ax.xaxis.label.set_color('darkslategrey')
        ax.yaxis.label.set_color('darkslategrey')
        ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
        ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
        ax.set_frame_on(True)
        fig.tight_layout()

        # add venue name, mean, and sd
        ax.text(6.75, 0.92, ven_name, fontsize=24, ha='right', va='center')
        mean_sd = 'mean distance = {:1.4f}\nSD = {:1.4f}'.format(np.mean(y), np.std(y))
        ax.text(6.75, 0.82, mean_sd, ha='right', va='center', fontsize=16)

        if fout is None:
            plt.show()
        else:
            fig.savefig(fout)


# end class LdaDriver

class AnnoteFinder(object):
    """
    callback for matplotlib to display an annotation when points are clicked on. The
    point which is closest to the click and within xtol and ytol is identified.

    Register this function like this:

    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, axis=None, xtol=.04, ytol=.04):
        self.data = zip(xdata, ydata, annotes)
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None:
            self.axis = plt.gca()
        else:
            self.axis = axis
        self.drawnAnnotations = {}
        self.links = []


    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return math.hypot(x1 - x2, y1 - y2)


    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if self.axis is None or self.axis == event.inaxes:
                annotes = []
                for x, y, a in self.data:
                    if clickX - self.xtol < x < clickX + self.xtol and clickY - self.ytol < y < clickY + self.ytol:
                        annotes.append((self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)


    def drawAnnote(self, axis, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        else:
            t = axis.text(x+0.03, y+0.03, u"{0:s}".format(annote),
                          bbox=dict(facecolor='#cccccc', alpha=0.75, edgecolor='#404040', linewidth=0.5), fontsize=9)
            m = axis.scatter([x], [y], marker='x', c='black', s=20, alpha=.8, zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.axis.figure.canvas.draw()


    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.axis, x, y, a)

# end class AnnoteFinder


def hellinger_distance(p, q):
    """
    Calculates the Hellinger distance between two probability distributions.

    :param p: 1st probability distribution
    :type p: numpy.ndarray
    :param q: 2nd probability distribution
    :type q: numpy.ndarray
    :return: number between 0 & 1. Lower numbers indicate higher similarity.
    :rtype: numpy.float64
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))/_SQRT2


def hellinger_matrix(dense_matrix):
    """
    Calculates pairwise Hellinger distances for columns of dense_matrix

    :param dense_matrix: matrix of probability distributions. Each column is a venue.
    :type dense_matrix: numpy.ndarray
    :return: distance matrix
    :rtype: numpy.ndarray
    """
    dense_matrix = dense_matrix.T
    sqrt_dense_matrix = np.sqrt(dense_matrix)
    size = len(dense_matrix)
    dist_matrix = np.ones((size, size))

    for i in range(size):
        sqrt_i = sqrt_dense_matrix[i]
        for j in range(i, size):
            sqrt_j = sqrt_dense_matrix[j]
            dist_matrix[i, j] = np.sqrt(np.sum((sqrt_i - sqrt_j)**2))/_SQRT2
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


if __name__ == '__main__':
    driver_settings = {'corpus_type':'twokenize',
                       'num_topics':25,
                       'num_passes':20,
                       'alpha':'symmetric',
                       'make_corpus':False,
                       'make_lda':False}
    driver = LdaDriver(**driver_settings)
    # driver.print_dist_matrix()

    venue_names = [ven.name for ven in driver.vens]
    venue_names[0] = 'LAX'
    venue_names[1] = 'SAN'
    venue_names[5] = 'OAK'
    venue_names[6] = 'SNA'
    venue_names[11] = 'BUR'
    venue_names[12] = 'SJC'
    venue_names[28] = 'LGB'

    # driver.vis_heatmap(driver.dist_matrix, venue_names)
    driver.vis_MDS(driver.dist_matrix, venue_names)
    # # driver.vis_ldavis()
    # #
    super_dist_matrix = driver.compare_venues(driver.vens)
    driver.vis_super_MDS(super_dist_matrix[:600, :600])

    driver.vis_super_cat_MDS(5)

    # driver.temporal_weekday_single_ven('4c31476b213c2d7f93cc335d')
    # driver.temporal_weekday_single_ven('445e36bff964a520fb321fe3')
    # driver.temporal_weekday_single_ven('44ded2f5f964a520d2361fe3')
    # driver.temporal_weekday_single_ven('4a6e5d0df964a52093d41fe3')
    # driver.temporal_weekday_single_ven('41cf5080f964a520a61e1fe3')
    # driver.temporal_weekday_single_ven('49dbd532f964a520155f1fe3')



    # driver.vis_ldavis()

    # driver_settings = {'corpus_type':'tweet',
    #                    'num_topics':25,
    #                    'num_passes':20,
    #                    'alpha':'symmetric',
    #                    'make_corpus':False,
    #                    'make_lda':True}
    # driver = LdaDriver(**driver_settings)
    # # driver.print_dist_matrix()
    #
    # driver.vis_heatmap(driver.dist_matrix, venue_names)
    # driver.vis_MDS(driver.dist_matrix, venue_names)
    # # driver.vis_ldavis()
    #
    # driver_settings = {'corpus_type':'twokenize',
    #                    'num_topics':25,
    #                    'num_passes':20,
    #                    'alpha':'symmetric',
    #                    'make_corpus':False,
    #                    'make_lda':True}
    # driver = LdaDriver(**driver_settings)
    # # driver.print_dist_matrix()
    #
    # driver.vis_heatmap(driver.dist_matrix, venue_names)
    # driver.vis_MDS(driver.dist_matrix, venue_names)
    # driver.vis_ldavis()
