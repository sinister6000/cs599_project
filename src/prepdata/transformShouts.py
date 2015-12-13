# -*- coding: utf-8 -*-
__metaclass__ = type

import codecs
import ujson
import datetime


def prep_json(ch_json):
    """
    Transform a raw checkin json to include only the fields we need.

    :param ch_json: raw checkin json
    :type ch_json: dict{}
    :return: new json with needed fields
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
        result['weekday'] = when.isoweekday()              # Monday = 1, Sunday = 7
    except KeyError, e:
        print 'KeyError:', e, 'while on checkin:', result['id']
        return ''
    except IndexError, e:
        print 'IndexError:', e, 'while on checkin:', result['id']
        pass
    return ujson.dumps(result)


def transform_shouts(fin, fout):
    """
    Reads raw shouts in fin, extracts needed fields, then writes to fout.

    :param fin: filename for input file
    :type fin: str
    :param fout: fileneame for output file
    :type fout: str
    :return: None
    :rtype: None
    """
    with codecs.open(fin, 'r', encoding='utf-8') as f_in, codecs.open(fout, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                ch_json = ujson.loads(line)
                new_shout_json = prep_json(ch_json)
                if new_shout_json != '':
                    f_out.write(new_shout_json)
                    f_out.write('\n')
            except ValueError, e:
                print 'ValueError:', e
                continue


def main():
    fin = '../../data/CA_shouts_30-34.dat'
    fout = '../../data/CA_shouts_30-34_transformed.dat'
    transform_shouts(fin, fout)


if __name__ == '__main__':
    main()
