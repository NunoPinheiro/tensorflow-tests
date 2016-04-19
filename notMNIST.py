from os import listdir
from random import shuffle, randint
import tensorflow as tf
from PIL import Image
from io import BytesIO

matchvalues = "ABCDEFGHIJ"
dataFolder = "notMNIST_small/"
learnSizePercentage = 0.7
batchPercentage = 0.01
validationSizePercentage = 0.1
testSizePercentage = 0.2
imgWidth = 28
imgSize = imgWidth * imgWidth
extraTestPercentage = 0

files = []
for i in matchvalues:
    folder = dataFolder + i + "/"
    letterFiles = listdir(folder)
    files.extend([{"file" : folder + file, "letter" : i} for file in letterFiles])
shuffle(files)

learnSize = int(round(learnSizePercentage*len(files)))
validationSize = int(round(validationSizePercentage*len(files)))
learnSet = files[0:learnSize]
validationSet = files[learnSize: learnSize + validationSize]
testSet = files[learnSize + validationSize:]

if extraTestPercentage > 0:
    for learnSetIndex in range(0 , learnSize):
        if randint(0, 100) < extraTestPercentage * 100:
            currentLearnImage = learnSet[learnSetIndex]
            transformation = "rotate" if randint(0, 1) == 0 else "shift"
            newLearnImage = {"file" : currentLearnImage['file'], "letter" : currentLearnImage['letter'], "transformation" : transformation}
            learnSet.append(newLearnImage)
            learnSize += 1
    #re shuffle test data to mix generated images with original images
    shuffle(learnSet)


def readFile(file):
        image = Image.open(file['file'])
        if "transformation" in file.keys():
            if file["transformation"] == "rotate":
                rotation = randint(-20, 20);
                image = image.rotate(rotation)
            if file["transformation"] == "shift":
                #move lose up to 2 bytes in each side
                lossMax = 5
                rightLoss = randint(0, lossMax)
                leftLoss = randint(0, lossMax)
                upLoss = randint(0, lossMax)
                downLoss = randint(0, lossMax)
                box = (rightLoss, upLoss, imgWidth - rightLoss - leftLoss, imgWidth - upLoss - leftLoss)
                crop = image.crop(box)
                image = Image.new("L", (imgWidth, imgWidth), color=0)
                image.paste(crop, box)
        return [int(byte.encode('hex'), 16) for byte in image.tobytes()]

batchSize = int(round(learnSize * batchPercentage))
batchCount = learnSize / batchSize
class BatchGenerator:
    def __init__(self):
        self.currentBatch = 0
    def generateBatch(self):
        currentIndex = self.currentBatch * batchSize + batchSize
        self.currentBatch +=1
        return getXsYs(learnSet[currentIndex - batchSize:currentIndex])

def normalize(xs):
    normalizedXs = []
    for i in range(0, len(xs)):
        normalizedXs.append( (128 - xs[i]) / 255.0)
    return normalizedXs

def getY_(letter):
    index = - (ord('A') - ord(letter))
    y_ = [0,0,0,0,0,0,0,0,0,0]
    y_[index] = 1
    return y_

def getLetter(yIndex):
    return unichr(ord('A') + yIndex)

def getXsYs(batch):
    xs = [normalize(readFile(file)) for file in batch]
    ys = [getY_(file['letter']) for file in batch]
    return (xs, ys)
###### Model First Iteration  ######
###### Direct feed from image to final softmax

x = tf.placeholder(tf.float32, shape=(None, imgSize))
W = tf.Variable(tf.zeros([imgSize, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, shape=(None, len(matchvalues)))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

batchGenerator = BatchGenerator()
for i in range(0, batchCount):
  batch_xs, batch_ys = batchGenerator.generateBatch()
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})




correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_xs, test_ys = getXsYs(validationSet)
print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

#Following code is used to manually compute the letter for a sample
#sample, sampleResults = getXsYs([{"file" : "notMNIST_small/A/QXJpYWwgQ0UgQm9sZC50dGY=.png", "letter" : "A"}])

#results = sess.run(tf.argmax(y,1), feed_dict={x:sample})
#humanizedResults = [getLetter(yIndex) for yIndex in results]
#print humanizedResults
