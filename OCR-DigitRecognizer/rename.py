import os
i=0
directory='C:/Users/Bhaskar Kaushal/DigitRecognition/src/full_data/train/cats'
for each in os.listdir(directory):
    i+=1
    os.rename(os.path.join(directory,each),os.path.join(directory,"cat0"+str(i)+'.jpg'))