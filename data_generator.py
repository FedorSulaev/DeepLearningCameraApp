import os
import tarfile

import cv2
import numpy

data_folder = 'data/'

def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, force=False):
    if not os.path.exists(data_folder + "/bgs"):
        os.mkdir(data_folder + "/bgs")
    elif not force:
        print('/bgs already present - Skipping extraction of %s.' % (archive_name))
        return

    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))
        fname = data_folder + "bgs/{:08}.jpg".format(index)
        print(fname)
        rc = cv2.imwrite(fname, im)
        if not rc:
            raise Exception("Failed to write file {}".format(fname))
        index += 1