import matplotlib.pyplot as plt

figGT = plt.figure()

plt.plot([5,10,15],[0.4,0.47,0.48])
plt.plot([5,10,15],[0.45,0.53,0.53])
plt.plot([5,10,15],[0.42,0.53,0.56])

figGT.suptitle('Gyroscope Train Score')
plt.legend(['Hidden Layer = 1','Hidden Layer = 2','Hidden Layer = 3'],loc='upper left')
plt.xlabel('Hidden units')
plt.ylabel('F1 score')

figGT.savefig('GyroTrain.png')

figGC = plt.figure()

plt.plot([5,10,15],[0.32,0.42,0.41])
plt.plot([5,10,15],[0.37,0.46,0.44])
plt.plot([5,10,15],[0.35,0.33,0.38])

figGC.suptitle('Gyroscope Cross Validation Score')
plt.legend(['Hidden Layer = 1','Hidden Layer = 2','Hidden Layer = 3'],loc='upper left')
plt.xlabel('Hidden units')
plt.ylabel('F1 score')

figGC.savefig('GyroTest.png')

figAT = plt.figure()

plt.plot([5,10,15],[0.53,0.58,0.59])
plt.plot([5,10,15],[0.55,0.61,0.62])
plt.plot([5,10,15],[0.57,0.64,0.66])

figAT.suptitle('Accelerometer Train Score')
plt.legend(['Hidden Layer = 1','Hidden Layer = 2','Hidden Layer = 3'],loc='upper left')
plt.xlabel('Hidden units')
plt.ylabel('F1 score')

figAT.savefig('AccTrain.png')

figAC = plt.figure()

plt.plot([5,10,15],[0.4,0.4,0.38])
plt.plot([5,10,15],[0.18,0.36,0.4])
plt.plot([5,10,15],[0.44,0.4,0.43])

figAC.suptitle('Accelerometer Cross Validation Score')
plt.legend(['Hidden Layer = 1','Hidden Layer = 2','Hidden Layer = 3'],loc='lower right')
plt.xlabel('Hidden units')
plt.ylabel('F1 score')

figAC.savefig('AccTest.png')
