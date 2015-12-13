# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
__metaclass__ = type

import os
import re
import logging

from ldaDriver import LdaDriver
import prepdata.sqliteQueries as sq


LT = u'(¯`·._.·(¯`·._.·(¯`·._.·'
RT = u'·._.·´¯)·._.·´¯)·._.·´¯)'
FL = u'░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓▓▓█████████'
FR = u'█████████▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░'
HR = (FL + FR) * 2



class CLI(object):
    """
    Command line interface.

    """
    def __init__(self):
        print u'\n\n', HR
        print u'            ', FR, u'Welcome to the TOPIC MODEL INFO (TMI) tool', FL
        print HR, u'\n'
        self.driver = self.load_or_new_tmi()
        print u'░▒▓█ ■ Corpus and Model Successfully Loaded.\n\n', HR


    def load_or_new_tmi(self):
        """Prompt user to load corpus/model from file or make new.
        """
        print(u'\n■ Before you can use TMI, you must load a test environment (a corpus & a topic model).')
        print(u'■ Enter 1 or 2 to select (recommended: 2 - TMI comes with a LOT of these to try)\n')
        print(u'  (1) Create new TMI test environment')
        print(u'  (2) Display list of saved environments to load (TMI includes many examples).')
        print(u'  (3) ■ Go back to main menu ■\n')
        cmd = get_input('> ', int, 1, 3)
        if cmd is 1:
            return self.new_tmi()
        elif cmd is 2:
            return self.load_tmi()
        elif cmd is 3:
            self.main_menu()
        else:
            print u'unrecognized command'

    def new_tmi(self):
        """
        Create corpus/model from scratch

        :return: LdaDriver
        :rtype: LdaDriver
        """
        # find out what tokenizer
        print(u'\n░▒▓█ ■ Select tokenizer:\n')
        print(u"░▒▓█   (1) gensim's basic tokenizer")
        print(u'░▒▓█   (2) twokenizer from TweetNLP')
        print(u'░▒▓█   (3) TweetTokenizer from NLTK')
        print(u'░▒▓█   (4) ■ Go back to main menu ■\n')
        cmd = get_input('> ', int, 1, 3)
        if cmd is 1:
            tokenizer = 'gensim'
        elif cmd is 2:
            tokenizer = 'twokenize'
        elif cmd is 3:
            tokenizer = 'tweet'
        elif cmd is 4:
            self.main_menu()
        else:
            print u'unrecognized command'

        # find out how many topics
        print (u'\n░▒▓█ ■ How many topics do you want in the model? (recommended: 15-60)\n')
        num_topics = get_input('> ', int, 1, 300)
        # find out how many passes
        print(u'\n░▒▓█ ■ While training the model, how many passes over the corpus should be performed?')
        print(u'░▒▓█ ■ More passes helps ensure model convergence. (recommended: 5-30)\n')
        num_passes = get_input('> ', int, 1, 300)
        # find out alpha
        print(u'\n░▒▓█ ■ While training the model, which setting do you want for alpha?')
        print(u'░▒▓█ ■ Symmetric assigns the same alpha to all topics (faster).')
        print(u'░▒▓█ ■ Auto learns an asymmetric prior directly from the data (much slower).\n')
        print(u'░▒▓█   (1) symmetric')
        print(u'░▒▓█   (2) auto')
        print(u'░▒▓█   (3) ■ Go back to main menu ■\n')
        cmd = get_input('> ', int, 1, 3)
        if cmd is 1:
            alpha = 'symmetric'
        elif cmd is 2:
            alpha = 'auto'
        elif cmd is 3:
            self.main_menu()
        else:
            print u'unrecognized command'

        # confirm creation with settings
        print(u'\n░▒▓█ ■ You have chosen the {} tokenizer, {} topics, {} passes, and {} alpha.\n'
              .format(tokenizer, num_topics, num_passes, alpha))
        print(u'░▒▓█   (1) Build environment with these settings.')
        print(u'░▒▓█   (2) I need to change a parameter.')
        print(u'░▒▓█   (3) ■ Go back to main menu ■\n')
        cmd = get_input('> ', int, 1, 3)
        if cmd is 1:
            # create LdaDriver with above settings
            driver_settings = {'corpus_type':tokenizer,
                               'num_topics':num_topics,
                               'num_passes':num_passes,
                               'alpha':alpha,
                               'make_corpus':True,
                               'make_lda':True}
            # check if there is an existing model with these settings
            model_name_from_settings = ('{}_lda_{}t_{}p_{}.model'
                                        .format(driver_settings['corpus_type'],
                                                driver_settings['num_topics'],
                                                driver_settings['num_passes'],
                                                driver_settings['alpha']))
            if model_name_from_settings in os.listdir('../../data/save/driver'):
                print(u'\n░▒▓█ ■ Environment already exists with these settings.')
                print(u'░▒▓█ ■ Loading existing environment.\n')
                driver_settings['make_corpus'] = False
                driver_settings['make_lda'] = False
            return LdaDriver(**driver_settings)
        elif cmd is 2:
            self.new_tmi()
        elif cmd is 3:
            self.main_menu()
        else:
            print u'unrecognized command'

    def load_tmi(self):
        """Prompts for file selection. Loads lda_driver from selected file
        """
        saved_list = [fname for fname in sorted(os.listdir('../../data/save/driver')) if fname.endswith('model')]
        print(u'\n░▒▓█ ■ Filenames are formatted as <tokenizer>_lda_<# topics>t_<# passes>p_<alpha>.model\n')
        print(u'░▒▓█ ■ Select an environment.\n')
        for i, fname in enumerate(saved_list):
            print(u'░▒▓█   ({}) {}'.format(i, fname))
        print(u'░▒▓█   ({}) ■ Create new test environment ■'.format(len(saved_list)))

        cmd = get_input('> ', int, 0, len(saved_list))
        if cmd is len(saved_list):
            return self.new_tmi()
        else:
            f_load = saved_list[cmd]

            # use RE to parse f_load for LdaDriver settings
            m = re.match(r'([a-z]+)_lda_([0-9]+)t_([0-9]+)p_([a-z]+)\.model', f_load)
            tokenizer = m.group(1)
            num_topics = int(m.group(2))
            num_passes = int(m.group(3))
            alpha = m.group(4)
            driver_settings = {'corpus_type':tokenizer,
                               'num_topics':num_topics,
                               'num_passes':num_passes,
                               'alpha':alpha,
                               'make_corpus':False,
                               'make_lda':False}
            driver = LdaDriver(**driver_settings)
            return driver

    def main_menu(self):
        print u'\n{} TMI Main Menu {}\n'.format(FR, FL)
        print u'▓█ ■ Enter a selection:'
        print u'▓█ (1) Display current environment settings'
        print u'▓█ (2) Change environment'
        print u'▓█ (3) Topic exploration'
        print u'▓█ (4) Venue-related tools'
        print u'▓█ (5) ■ Quit ■\n'
        cmd = get_input('> ', int, 1, 5)
        if cmd is 1:
            print self.driver
            self.main_menu()
        elif cmd is 2:
            self.driver = self.load_or_new_tmi()
            self.main_menu()
        elif cmd is 3:
            self.topic_menu()
            # self.main_menu()
        elif cmd is 4:
            self.venue_menu()
            # self.main_menu()
        elif cmd is 5:
            print(u'Bye')
            exit(0)
        else:
            print(u'Command unrecognized.')
            self.main_menu()

    def topic_menu(self):
        """
        CLI menu for topic-related tools

        """
        print u'\n░▒▓█ ■ Select a topic-viewing method.\n'
        print u'░▒▓█   (1) top n lists - for each topic, display top n most probable words'
        print u'░▒▓█   (2) pyLDAvis interactive visualization'
        print u'░▒▓█   (3) ■ back to main menu ■\n'
        pick_vis = get_input('> ', int, 1, 3)
        if pick_vis is 1:
            print u'\n░░▒▒▓█ ■ How many words to display per topic (recommended: 5-15)?\n'
            n = get_input('> ', int, 1, 200)
            self.driver.print_topics(n)
            self.topic_menu()
        if pick_vis is 2:
            print u'\n░░▒▒▓█ ■ This will open in a browser window...\n'
            self.driver.vis_ldavis()
            self.topic_menu()
        if pick_vis is 3:
            self.main_menu()


    def venue_menu(self):
        """
        Menu for venue visualizations.
        """
        print(u'\n░▒▓█ ■ Select a visualization of the inter-venue semantic distances.\n')
        print(u'░▒▓█   (1) 30 Venue Heatmap - top 30 venues')
        print(u'░▒▓█   (2) 30 Venue MDS - top 30 venues')
        print(u'░▒▓█   (3) 600 Venue MDS - top 600 venues')
        print(u'░▒▓█   (4) MDS - top 40 venues from each of top 5 categories')
        print(u'░▒▓█   (5) Temporal view. - top 2 venues from top 5 categories')
        print(u'░▒▓█   (6) ■ back to main menu ■\n')
        pick_vis = get_input('> ', int, 1, 6)
        if pick_vis is 1:
            print u'\n░░▒▒▓█ ■ You must close graph in order to continue using program.'
            self.driver.vis_heatmap(self.driver.dist_matrix, [ven.name for ven in self.driver.vens])
            self.venue_menu()
        elif pick_vis is 2:
            print u'\n░░▒▒▓█ ■ You must close graph in order to continue using program.'
            self.driver.vis_MDS(self.driver.dist_matrix, [ven.name for ven in self.driver.vens])
            self.venue_menu()
        elif pick_vis is 3:
            print u'\n░░▒▒▓█ ■ You must close graph in order to continue using program.'
            super_dist_matrix = self.driver.compare_venues(self.driver.vens)
            self.driver.vis_super_MDS(super_dist_matrix)
            self.venue_menu()
        elif pick_vis is 4:
            print u'\n░░▒▒▓█ ■ You must close graph in order to continue using program.'
            self.driver.vis_super_cat_MDS(5)
            self.venue_menu()
        elif pick_vis is 5:
            cat_list, ven_list = sq.venues_from_top_n_categories(2, 5)
            for v in ven_list:
                self.driver.temporal_weekday_single_ven(v.id)
            self.venue_menu()
        elif pick_vis is 6:
            self.main_menu()


def get_input(prompt, type_=None, min_=None, max_=None, range_=None):
    """
    Creates a prompt and validates user input.

    :param prompt: message to show user
    :type prompt: str
    :param type_: indicates what type of input is expected
    :type type_: str
    :param min_: min value accepted
    :type min_: int
    :param max_: max value accepted
    :type max_: int
    :param range_: can be either a range with a start and stop, or can be a list of accepted values
    :type range_: dict{} or list[]
    """
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = raw_input(prompt)
        if ui is -1:
            exit(0)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) is 1:
                    print(template.format(*range_))
                else:
                    print(template.format(" or ".join((", ".join(map(str, range_[:-1])), str(range_[-1])))))
        else:
            return ui


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.CRITICAL)
    os.chdir('./prepdata')
    mycli = CLI()
    mycli.main_menu()
