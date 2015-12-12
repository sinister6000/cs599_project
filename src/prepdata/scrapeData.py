# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

__metaclass__ = type

import codecs
import time

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from requests.packages.urllib3.exceptions import ReadTimeoutError

ckey = 't8sInaObAeDOI1dFkPQpNwz2S'
csecret = 'j3aANT5svxmCc42JxrCflUieugM4tUci7hQ5iTOZWSaYwxtPJ0'
atoken = '3220113236-10h1ljtrXdeVAlk3jtt0iGmK2FoNqMEaQW0ux7m'
asecret = 'opSi4CkqeMPFO4NCxqAARsxebBmDqcFQLoSZE0Ol6FCDD'

dataFile = '../../data/twitData35.dat'


class Listener(StreamListener):
    """
    Simple class to write Twitter data to a file
    """

    def on_data(self, data):
        print data
        saveFile.write(data)
        return True

    def on_error(self, status):
        print status

def main():
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    lstn = Listener()
    with codecs.open(dataFile, 'a', encoding='utf-8') as saveFile:
        while True:
            try:
                twitterStream = Stream(auth, lstn)
                # only grab tweets that contain any of these keywords and written in English
                twitterStream.filter(track=["foursquare", "4sq", "swarmapp"], languages=["en"])
            except ReadTimeoutError:
                print('ReadTimeoutError...')
                time.sleep(1)
            except:
                time.sleep(1)


if __name__ == '__main__':
    main()
