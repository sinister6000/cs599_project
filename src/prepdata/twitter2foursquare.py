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
    This class handles conversion of tweets into foursquare checkins into shortened checkins.
    """

    def __init__(self):
        self.a_token = "DTZ5HFUFJFV11VP2INOSGME3J02L0GFGBIA00V4K1PZEXSQO"

    def tweets_to_checkins(self, fin, fout):
        """
        Processes tweets.  Retrieves Foursquare checkins associated with the tweets.

        Uses swampapp reference to query foursquare's API for full checkin info. Foursquare API has a max 500 requests
        per hour.
        Checkins come in json format and are output to file.

        :param fin: Filename containing raw tweets.
        :type fin: str
        :param fout: Output filename of foursquare json checkins.
        :type fout: str
        """
        with codecs.open(fin, 'r', encoding='utf-8') as tw_file, codecs.open(fout, 'a', encoding='utf-8') as ch_file:
            count = 1
            for line in tw_file:
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
                    ch_json = rJson['response']['checkin']
                    clean_ch_json = self.shorten_shout_json(ch_json)
                    if clean_ch_json != '':
                        ch_file.write(clean_ch_json)
                        ch_file.write('\n')
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
                    continue


    def get_categories(self):
        """
        Downloads foursquare's current categories list.

        :return: None
        :rtype: None
        """
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


    def batch_conversion(self):
        os.chdir('../../data/CA_split/')
        # split_file_list = sorted(os.listdir(os.getcwd()), key=lambda filename:int(filename[27:-4]))
        split_file_list = sorted(os.listdir(os.getcwd()), key=lambda filename:int(filename[21:-4]))
        for i, fname in enumerate(split_file_list):
            print i, '\t', fname
        print '\n'

        fout = '../CA_shouts_transformed.dat'
        for f in split_file_list:
            print('processing ' + f)
            self.tweets_to_checkins(f, fout)
            print('done with ' + f + '\n')
            time.sleep(3600)


    @staticmethod
    def shorten_shout_json(ch_json):
        """
        Edits a foursquare checkin json to include only the fields we need.

        :param ch_json: raw checkin json
        :type ch_json: dict{}
        :return: new json string with needed fields
        :rtype: str
        """
        result = {}
        try:
            result['id'] = ch_json['id']
            result['userid'] = ch_json['user']['id']
            result['userLast'] = ch_json['user'].get('lastName', None)
            result['userFirst'] = ch_json['user'].get('firstName', None)
            result['venue'] = ch_json['venue']['id']
            result['venueName'] = ch_json['venue'].get('name', None)
            result['venueCity'] = ch_json['venue']['location'].get('city', None)
            result['venueState'] = ch_json['venue']['location'].get('state', None)
            result['venueZip'] = ch_json['venue']['location'].get('postalCode', None)
            venCat = ch_json['venue'].get('categories', None)
            if not venCat:
                result['venueCatID'] = None
                result['venueCatName'] = None
            else:
                result['venueCatID'] = venCat[0]['id']
                result['venueCatName'] = venCat[0]['name']
            result['shout'] = ch_json['shout']
            when = datetime.datetime.utcfromtimestamp(int(ch_json['createdAt']) + 60*int(ch_json['timeZoneOffset']))
            result['date'] = when.date().strftime("%Y-%m-%d")
            result['time'] = when.time().strftime("%H:%M:%S")
            result['weekday'] = when.isoweekday()  # Monday = 1, Sunday = 7
        except KeyError, e:
            print 'KeyError:', e, 'while on checkin:', result['id']
            return ''
        except IndexError, e:
            print 'IndexError:', e, 'while on checkin:', result['id']
            pass
        return ujson.dumps(result)
# end class Twitter2foursquare

def main():
    t2f = Twitter2foursquare()
    t2f.get_categories()
    t2f.batch_conversion()



if __name__ == '__main__':
    main()
