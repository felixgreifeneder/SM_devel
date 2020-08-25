import ee
import time

def GEEtoGDrive(geds, name, dir, sampling, roi, timeout=True):

    file_exp = ee.batch.Export.image.toDrive(image=geds,
                                             description='fileexp' + name,
                                             folder=dir,
                                             fileNamePrefix=name,
                                             scale=sampling,
                                             region=roi.getInfo()['coordinates'],
                                             maxPixels=1000000000000)

    file_exp.start()

    start = time.time()
    success = 1

    while (file_exp.active() == True):
        time.sleep(2)
        if timeout == True and (time.time()-start) > 4800:
            success = 0
            break
    else:
        print('Export completed')

    return success

ee.Initialize()

minlon = -3.7
maxlon = -2.5
minlat = 36.88
maxlat = 37.27

roi = ee.Geometry.Polygon([[[minlon, minlat],
                            [minlon, maxlat],
                            [maxlon, maxlat],
                            [maxlon, minlat],
                            [minlon, minlat]]])

s3_collection = ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(roi) \
                                                        .map(lambda x: x.clip(roi))

s3_list = s3_collection.toList(1000)
llength = s3_list.length().getInfo()

for i in range(llength):
    cimage = ee.Image(s3_list.get(i))
    cimagename = cimage.get('system:index').getInfo()
    success = GEEtoGDrive(cimage.select(0,1,2,3,4,5,6,7,8,9,10,
                        11,12,13,14,15,16,17,18,19,20), cimagename, 'sentinel3', 300, roi)

