# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

__metaclass__ = type

import filter
import twitter2foursquare
import sqliteModels
import sqliteQueries


def main():
    filter.main()
    twitter2foursquare.main()
    sqliteModels.main()
    sqliteQueries.main()

if __name__ == '__main__':
    main()
