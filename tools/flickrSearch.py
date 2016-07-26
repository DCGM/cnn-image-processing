#!/usr/bin/env python

from __future__ import print_function
import flickrapi
import pprint
import argparse
import urllib2
import json
import os

api_key = u'5ea6c4c69dacd0c9eab3c285a8d226e6'
api_secret = u'520d57b564e636a8'

photoSizes = ['o', 'l', 'c']
sizeUrls = ['url_' + i for i in photoSizes]
sizeWidths = ['width_' + i for i in photoSizes]
sizeHeights = ['height_' + i for i in photoSizes]


def parseArgs():
    parser = argparse.ArgumentParser(description="Train the coef CNN.")
    parser.add_argument("-q", "--query", help="Text query",
                        type=str, required=True)
    parser.add_argument("-s", "--min_size", help="Min size in pixels (image area).",
                        type=int, default=100000)
    parser.add_argument("-c", "--image_count", help="Number of images to download.",
                        type=int, default=100)
    parser.add_argument("--commercial", action='store_const', const='4,5,6,7,8', default='1,2,3,4,5,6,7,8', help="Only images with commercial use allowed. (see flickr.photos.licenses.getInfo)")
    args = parser.parse_args()
    return args

class flickrAPI:
    def __init__(self, api_key, api_secret):
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    def queryPhotos(self, args, max_upload_date=None):
        query = {}
        query['text'] = args.query
        query['media'] = 'photos'
        query['license'] = args.commercial
        query['per_page'] = '1000'
        query['extras'] = 'description,license,date_upload,date_taken,owner_name,original_format,geo,tags,machine_tags,' + ','.join(sizeUrls),
        query['sort'] = 'date-posted-desc'
        if max_upload_date:
            query['max_upload_date'] = max_upload_date
        photos = self.flickr.photos.search(**query)

        return photos

def main():

    args = parseArgs()

    api = flickrAPI( api_key=api_key, api_secret=api_secret)
    outPath = '_'.join(args.query.split())
    try:
        os.makedirs(outPath)
    except OSError as exc:
        pass

    count = 0
    last=None
    while count < args.image_count:
        photos = api.queryPhotos(args, max_upload_date=last)
        print('Total', photos['photos']['total'])

        for photo in photos['photos']['photo']:

            last = photo['dateupload']
            size = 0
            url = None
            for u,w,h in zip(sizeUrls,sizeWidths,sizeHeights):
                if w in photo and h in photo:
                    size = max(size, int(photo[w]) * int(photo[h]))
                    url = u
                    break

            if size > args.min_size:
                try:
                    print(photo['dateupload'], photo['id'], size, 'DOWNLOADING', url)
                    name = "{}_{}_{}".format(photo['dateupload'], photo['owner'], photo['id'])
                    name = os.path.join(outPath, name)

                    t, ext = os.path.splitext(photo[url])
                    response = urllib2.urlopen(photo[url])
                    data = response.read()
                    with open(name+'.json', 'w') as f:
                        json.dump(photo, f)
                    with open(name+ext, 'w') as f:
                        f.write(data)
                    count += 1
                except Exception as e:
                    print('FAILED', photo['id'], e)
            else:
                print(photo['dateupload'], photo['id'], size, 'SKIPPING', url)

            if count > args.image_count:
                break

if __name__ == "__main__":
    main()
