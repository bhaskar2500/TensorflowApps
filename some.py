import tensorflow as tf
import pygame
import scipy.io as sio
import numpy as np
def weights(shape):
    return tf.Variable(tf.random_normal(shape))

def bias(shape):
    return tf.Variable(tf.random_normal(shape))

def neural_network_model(data):
    #input data * weight) + biases
    total_nodes_hl1=400
    total_classes=10
    hidden_layer_1 = {'weights':weights([900,total_nodes_hl1]),'biases':bias([total_nodes_hl1]) }

    output_layer = {'weights':weights([total_nodes_hl1,total_classes]),'biases':bias([total_classes])}

    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
	#rectified linear as threshold function
    layer_1 = tf.nn.relu(layer_1)
    output = tf.matmul(layer_1,output_layer['weights']) + output_layer['biases']
    return output

inputTheta = sio.loadmat('newX.mat')
x=inputTheta['X']
epoch_x = tf.placeholder('float',[None,x.shape[1]])
epoch_y = tf.placeholder('float')


def train_neural_net(x):
    y= np.array(sio.loadmat('newY.mat')['y'])
    xTrain,xTest,yTrain,yTest=splitData(x,y)
    xTrain=np.float32(xTrain)
    a=np.array(yTrain,'int32')
    d=tf.pack(a)
    oneHotYTrain=tf.one_hot(d,10,1.0,0.0)

    prediction = neural_network_model(xTrain)
    
    oneHotYTrain=tf.reshape(oneHotYTrain,[2115,10])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,oneHotYTrain))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        hm_epochs = 15

        # for epoch in range(hm_epochs):
        #     epoch_loss = 0
        #     for _ in range(int(y.shape[0]/100)):

        #         _, c = sess.run([optimizer, cost], feed_dict= {epoch_x:xTrain, epoch_y:oneHotYTrain.eval()})
        #         epoch_loss += c
        #     print('Epoch', epoch, 'UNDERSCORE', _,'loss:',epoch_loss)

        correcct = tf.equal(tf.argmax(prediction,1),tf.argmax(oneHotYTrain,1))
        print(correcct.eval())
        accuracy = tf.reduce_mean(tf.cast(correcct,'float'))
        print(yTest[1])
        print('Accuracy:', accuracy.eval({epoch_x:xTest,epoch_y:yTest}))


def splitPredictionData(y):
    """Split sample data into training set (80%) and testing set (20%)"""

    size1 =int(y.shape[0] * 0.8)
    ytrain = np.zeros((size1,X.shape[1]))
    for i, v in enumerate(np.random.permutation(len(y))):
            #print(i, y[v], len(X[v]))
        try:
            ytrain[i] = y[v]
        except:
            continue
    return ytrain
def splitData(X, y):
    """Split sample data into training set (80%) and testing set (20%)"""

    size1 =int( X.shape[0] * 0.8)
    size2 = int(X.shape[0] * 0.2)
    Xtrain = np.zeros((size1,X.shape[1]))
    Xtest = np.zeros((size2+1,X.shape[1]))
    ytrain = np.zeros((size1,1))
    ytest = np.zeros((size2+1,1))

    for i, v in enumerate(np.random.permutation(len(y))):
        #print(i, y[v], len(X[v]))

        try:
            Xtrain[i] = X[v]
            ytrain[i] = y[v]
        except:
            Xtest[i-size1] = X[v]
            ytest[i-size1] = y[v]
    return Xtrain, Xtest, ytrain, ytest

if __name__ == "__main__":
    train_neural_net(x)
    #main















# def main():
#     """Main method. Draw interface"""

#     global screen
#     pygame.init()
#     screen = pygame.display.set_mode((730, 450))
#     pygame.display.set_caption("Handwriting recognition")

#     background = pygame.Surface((360,360))
#     background.fill((255, 255, 255))
#     background2 = pygame.Surface((360,360))
#     background2.fill((255, 255, 255))

#     clock = pygame.time.Clock()
#     keepGoing = True
#     lineStart = (0, 0)
#     drawColor = (255, 0, 0)
#     lineWidth = 15

#     inputTheta = sio.loadmat('scaledTheta.mat')
#     theta = inputTheta['t']
#     pygame.display.update()
#     image = None

#     while keepGoing:

#         clock.tick(30)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 keepGoing = False
#             elif event.type == pygame.MOUSEMOTION:
#                 lineEnd = pygame.mouse.get_pos()
#                 if pygame.mouse.get_pressed() == (1, 0, 0):
#                     pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
#                 lineStart = lineEnd
#             elif event.type == pygame.MOUSEBUTTONUP:
#                 screen.fill((0, 0, 0))
#                 screen.blit(background2, (370, 0))
#                 #w = threading.Thread(name='worker', target=worker)

#                 image = calculateImage(background, screen, Theta1, Theta2, lineWidth)

#             elif event.type == pygame.KEYDOWN:
#                 myData = (event, background, drawColor, lineWidth, keepGoing, screen, image)
#                 myData = checkKeys(myData)
#                 (event, background, drawColor, lineWidth, keepGoing) = myData


#         screen.blit(background, (0, 0))
#         pygame.display.flip()

# def checkKeys(myData):
#     """test for various keyboard inputs"""

#     (event, background, drawColor, lineWidth, keepGoing, screen, image) = myData

#     if event.key == pygame.K_q:
#         keepGoing = False
#     elif event.key == pygame.K_c:
#         background.fill((255, 255, 255))
#         drawPixelated(np.zeros((30,30)), screen)
