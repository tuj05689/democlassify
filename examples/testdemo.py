from deepnet.deepclassify import SkiNet

obj = SkiNet(img="dog3.jpg")

obj.evalute()

print("Object is {}|||{}".format(obj.getName(),obj.getPercentage()))