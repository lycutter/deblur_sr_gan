# rename files
import os
path = 'D:/pythonWorkplace/Dataset/GFNDataSet/GOPR0884_11_00/blur'
count = 1
for file in os.listdir(path):
    os.rename(os.path.join(path,file),os.path.join(path,  'GOPR0884_11_00'+str(count)+".png"))

    count+=1