from deepnet.deepclassify import SkiNet

obj = SkiNet(img="dog2.jpg")

obj.evalute()

print("Object is {}|||{}".format(obj.getName(),obj.getPercentage()))