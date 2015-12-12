# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
__metaclass__ = type

import codecs
import ujson

import peewee as pw

db = pw.SqliteDatabase('c:\\docs\\python\\twitterscrape\\data\\db\\10-27.sqlite')


class BaseModel(pw.Model):
    """
    Parent database model for all database Models. Insures proper parameters in created Models.
    """
    class Meta:
        """Sets model parameters."""
        database = db


class User(BaseModel):
    """
    ORM model of the User table
    """
    id = pw.CharField(unique=True, primary_key=True)
    lastname = pw.CharField(null=True)
    firstname = pw.CharField(null=True)
    shout_count = pw.IntegerField()


class Venue(BaseModel):
    """
    ORM model of the Venue table
    """
    id = pw.CharField(unique=True, primary_key=True)
    name = pw.CharField()
    city = pw.CharField(null=True)
    state = pw.CharField(null=True)
    zip = pw.CharField(null=True)
    cat_id = pw.CharField(null=True)
    cat_name = pw.CharField(index=True, null=True)
    shout_count = pw.IntegerField()


class UserVenue(BaseModel):
    """
    Intermediate table for the many-to-many relationship between User and Venue.
    """
    user = pw.ForeignKeyField(User, on_delete='CASCADE', on_update='CASCADE')
    venue = pw.ForeignKeyField(Venue, on_delete='CASCADE', on_update='CASCADE')


class Checkin(BaseModel):
    """
    ORM model of the Checkin table
    """
    id = pw.CharField(unique=True, primary_key=True)
    shout = pw.TextField()
    date = pw.DateField()
    time = pw.TimeField()
    weekday = pw.IntegerField()
    user = pw.ForeignKeyField(User, related_name='checkins', on_delete='SET NULL', on_update='CASCADE')
    venue = pw.ForeignKeyField(Venue, related_name='checkins', on_delete='CASCADE', on_update='CASCADE')


def bulk_insert_ch(ch_json_file):
    """
    Create entries into the tables from data in json file.

    :param ch_json_file: transformed checkin json filename
    :type ch_json_file: str
    :return: 2-tuple of num_processed, num_loaded
    :rtype: 2-tuple of ints
    """
    with codecs.open(ch_json_file, 'r', encoding='utf-8') as fin:
        num_loaded = 0
        num_processed = 0
        with db.atomic():
            for line in fin:
                try:
                    ch = ujson.loads(line)

                    curr_user, created = User.get_or_create(
                        id=ch['userid'],
                        defaults={'lastname': ch['userLast'],
                                  'firstname': ch['userFirst'],
                                  'shout_count': 0})

                    curr_venue, created = Venue.get_or_create(
                        id=ch['venue'],
                        defaults={'name': ch['venueName'],
                                  'city': ch['venueCity'],
                                  'state': ch['venueState'],
                                  'zip': ch['venueZip'],
                                  'cat_id': ch['venueCatID'],
                                  'cat_name': ch['venueCatName'],
                                  'shout_count': 0})

                    UserVenue.get_or_create(
                        user=curr_user,
                        venue=curr_venue)

                    Checkin.create(
                        id=ch['id'],
                        user=curr_user,
                        venue=curr_venue,
                        date=ch['date'],
                        time=ch['time'],
                        weekday=ch['weekday'],
                        shout=ch['shout'])

                    num_loaded += 1

                except pw.IntegrityError, e:
                    print 'tried to load duplicate'
                    continue

                finally:
                    num_processed += 1

        print 'Processed {} checkins. Created {} new rows in checkin table.'.format(num_processed, num_loaded)
        return num_processed, num_loaded


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


def main():
    db.connect()
    db.create_tables([Checkin, User, Venue, UserVenue], safe=True)
    # fin = '../../data/CA_shouts_1-28.dat'
    fout = '../../data/CA_shouts_30-34_transformed.dat'
    # transform_shouts.transform_shouts(fin, fout)
    bulk_insert_ch(fout)
    fout = '../../data/CA_shouts_1-28_transformed.dat'
    bulk_insert_ch(fout)
    update_shout_counts()
    db.close()


if __name__ == '__main__':
    main()
