from collections import namedtuple

EdgeArea = namedtuple("EdgeArea", ("top", "left", "right", "bottom"))


def getInEdgeArea(x, y, boundingDistance, frameSize):
    top = y <= 0 + boundingDistance
    left = x <= 0 + boundingDistance
    right = x >= frameSize.width - boundingDistance
    bottom = y >= frameSize.height - boundingDistance
    return EdgeArea(top, left, right, bottom)

