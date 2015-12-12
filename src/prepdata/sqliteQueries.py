# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
__metaclass__ = type

import codecs
import math

import peewee as pw

from sqliteModels import User, Venue, Checkin

db = pw.SqliteDatabase('../../data/db/10-27.sqlite')


def update_shout_counts():
    """
    Counts total # of shouts for each user and each venue.

    :return: None
    :rtype: None
    """
    venue_subquery = Checkin.select(pw.fn.COUNT(Checkin.id)).where(Checkin.venue == Venue.id)
    venue_update = Venue.update(shout_count=venue_subquery)
    venue_update.execute()

    user_subquery = Checkin.select(pw.fn.COUNT(Checkin.id)).where(Checkin.user == User.id)
    user_update = User.update(shout_count=user_subquery)
    user_update.execute()


def venues_to_docs():
    """
    For each venue, creates a file of all related shouts, one shout per line.

    :return: None
    :rtype: None
    """
    for ven in Venue.select():
        with codecs.open('../../data/ven/{}.txt'.format(ven.id), 'w', encoding='utf-8') as ven_f:
            for checkin in ven.checkins:
                ven_f.write(checkin.shout)
                ven_f.write('\n')


def venues_to_doc(fname='../../data/allVenues.txt'):
    """
    Writes all venues to one file.

    :param fname: filename of output file
    :return: None
    :rtype: None
    """
    with codecs.open(fname, 'w', encoding='utf-8') as fout:
        for ven in Venue.select():
            fout.write(ven.id + ' ')
            checkin_str = u' '.join(ven.checkins)
            fout.write(checkin_str)
            fout.write('\n')


def topn_venues(n=30):
    """
    Finds the n venues with the most shouts.

    :param n: how many venues to find
    :type n: int
    :return: list of n venues (id, name)
    :rtype: []
    """
    venues_by_shoutcount = (Venue
                            .select()
                            .order_by(Venue.shout_count.desc())
                            .limit(n))
    return [ven for ven in venues_by_shoutcount]


def split_weekdays(ven_id):
    """
    Divides a venue's shouts into bins by day of the week.

    :param ven_id: venue id to be split
    :type ven_id: str
    :return: a dict where key=weekday, value=list of shouts
    :rtype: dict{}
    """
    ven_by_weekday = (Checkin
                      .select(Checkin.weekday, Checkin.shout)
                      .where(Checkin.venue == ven_id)
                      .order_by(Checkin.weekday))
    weekday_dict = {}
    for checkin in ven_by_weekday.execute():
        shouts_temp = weekday_dict.get(checkin.weekday, [])
        shouts_temp.append(checkin.shout)
        weekday_dict[checkin.weekday] = shouts_temp
    return weekday_dict


def get_ven_by_id(ven_id):
    """
    Get venue name for ven_id.

    :param ven_id: ID of venue to find name
    :type ven_id: str
    :return: Venue
    :rtype: database row
    """
    try:
        return Venue.get(Venue.id == ven_id)
    except Venue.DoesNotExist:
        print(u'Venue with that ID does not exist.')


def get_categories(n):
    """
    Gets set of categories for venues whose shout_count > n.

    :param n: shout_count > n
    :type n: int
    :return: set of category names
    :rtype: [str]
    """
    return [ven.cat_name for ven in Venue.raw('select distinct cat_name '
                                              'from venue '
                                              'where id in '
                                                 '(select venue.id '
                                                 'from venue '
                                                 'where shout_count > ?)', n)]


def venues_from_top_n_categories(n=15):
    """
    Gets the top n categories ordered by shout_count

    :param n:
    :type n:
    :return: 2 lists - one for categories, and one for venues
    :rtype: ([], [])
    """
    cats = Venue.raw('SELECT cat_id, cat_name '
                     'FROM venue '
                     'GROUP BY cat_id '
                     'ORDER BY sum(shout_count) DESC '
                     'LIMIT ?', n)
    cat_list = []
    venue_list = []
    for v in cats.execute():
        vens = [ven for ven in (Venue
                                .select()
                                .where(Venue.cat_id == v.cat_id)
                                .order_by(-Venue.shout_count)
                                .limit(math.floor(200/n)))]
        cat_list.append(v.cat_name)
        venue_list.extend(vens)
    return cat_list, venue_list


def venue_count():
    return pw.SelectQuery(Venue).count()

def user_count():
    return pw.SelectQuery(User).count()

def checkin_count():
    return pw.SelectQuery(Checkin).count()

if __name__ == '__main__':
    db.connect()
    # venues_to_docs()
    # update_shout_counts()
    # topn = topn_venues(50)
    # topnIDs = ['{}'.format(n.id) for n in topn]
    # print topnIDs
    # for c in get_categories(24):
    #     print c
    print venue_count()

    db.close()
