# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
__metaclass__ = type

import codecs
import ujson
import datetime
import time
import os

import requests


class Twitter2foursquare(object):
    """
    This class handles conversion of tweets into foursquare checkins.
    """

    def __init__(self):
        self.a_token = "DTZ5HFUFJFV11VP2INOSGME3J02L0GFGBIA00V4K1PZEXSQO"

    def tweets_to_checkins(self, fin, fout):
        """
        Processes tweets.

        Uses swampapp reference to query foursquare's API for full checkin info. Foursquare API has a max 500 requests
        per hour.
        Checkins come in json format and are output to file.

        :param fin: Filename containing raw tweets.
        :param fout: Output filename of foursquare json checkins.
        """
        with codecs.open(fin, 'r', encoding='utf-8') as tweets_file, codecs.open(fout, 'a', encoding='utf-8') as ch_file:
            count = 1
            for line in tweets_file:
                try:
                    twt = ujson.loads(line)
                    temp_url = twt['entities']['urls'][0]['expanded_url']
                    # find index of last '/' in the url, use index to get shortID substring
                    truncate_at = temp_url.rfind('/')
                    shortID = temp_url[truncate_at+1:]
                    print('    shout: ' + shortID)

                    # try max 30 attempts to get a reponse from foursquare API.
                    for i in range(30):
                        try:
                            r = requests.get('https://api.foursquare.com/v2/checkins/resolve?shortId=' +
                                             '{sid}&oauth_token={at}'.format(sid=shortID, at=self.a_token) + '&v=20150603')
                        except requests.exceptions.ConnectionError, e:
                            print 'ConnectionError:', e
                            time.sleep(3)
                        else:
                            break

                    rJson = r.json()
                    ch_file.write(ujson.dumps(rJson['response']['checkin']) + '\n')
                    count += 1

                except IndexError, e:
                    print 'IndexError', e
                    if rJson['meta']['code'] == 403:
                        self.display_403(rJson['meta']['errorType'], count)
                    continue

                except KeyError, e:
                    print 'KeyError', e
                    if rJson['meta']['code'] == 403:
                        self.display_403(rJson['meta']['errorType'], count)
                    continue

                except ValueError, e:
                    print 'ValueError', e
                    pass


    def get_categories(self):
        r = requests.get('https://api.foursquare.com/v2/venues/categories?oauth_token={at}'
                         .format(at=self.a_token) + '&v=20150603')
        rJson = r.json()
        categories = rJson['response']['categories']
        with codecs.open('../../data/categories.json', 'w', encoding='utf-8') as fout:
            fout.write(ujson.dumps(categories))


    @staticmethod
    def display_403(error_type, count):
        """
        Outputs info about server response 403.
        :param error_type: details about the error
        :type error_type: str
        :param count: how many shouts were processed until error
        :type count: int
        :return: None
        :rtype: None
        """
        print error_type
        print datetime.datetime.now().time().isoformat()
        print('max # requests for this hour')
        print('processed ' + str(count) + ' shouts')
        time.sleep(3600)


    def countUniqueUsers(self):
        return len(set([self.checkins[c]['userID'] for c in self.checkins]))


if __name__ == '__main__':
    t2f = Twitter2foursquare()
    t2f.get_categories()

    os.chdir('../../data/CA_split/')
    split_file_list = sorted(os.listdir(os.getcwd()), key=lambda filename: int(filename[27:-4]))
    for i, fname in enumerate(split_file_list):
        print i, '\t', fname
    print '\n'

    fout = '../CA_shouts_30-34.dat'
    for f in split_file_list:
        print('processing ' + f)
        t2f.tweets_to_checkins(f, fout)
        print('done with ' + f + '\n')
        time.sleep(3600)
