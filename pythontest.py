import pandas as pd

#from Keywordizers.metrics import Processor

import requests
from acc import Processor
import matplotlib.pyplot as plt
import numpy as np

'''nodes = [0,10, 20, 30, 40, 50, 60]
sq1dat = [48, 235, 435, 580, 690,780,865]
sq2dat = [55, 180, 235, 300, 335,400,450]
sq3dat = [50, 210, 285, 352, 390,450,510]
sq4dat = [47, 230, 310, 370, 410,470,560]
sq5dat = [43, 170, 320, 380, 420,510,590]
plt.plot(nodes, sq1dat, color='darkblue',label='SQ1')
plt.plot(nodes, sq2dat, color='red',label='SQ2')
plt.plot(nodes, sq3dat, color='green',label='SQ3')
plt.plot(nodes, sq4dat, color='purple',label='SQ4')
plt.plot(nodes, sq5dat, color='orange',label='SQ5')
plt.xlabel('rows')
#plt.grid()
plt.ylabel('Execution time')
plt.title('Time Taken to Execute data')
plt.ylim([0,1400])
plt.legend()
plt.savefig('./static/assets/el.png')
#plt.show()
plt.close()'''

'''
index=[]
url = "http://elasticsearch.site.org.in/"
timeout = 5
index=pd.read_csv("Traindata.csv",dtype={},engine='python')
print('**************************Indexed Values Start***************************')
print(Processor.elasticindex())       
print('**************************Indexed Values End***************************')

'''

'''
print(Processor.elasticindex())
'''


'''
nodes,qdata=Processor.epochcal()
nodes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
qdata=[0.8, 0.794, 0.79, 0.788, 0.787, 0.786, 0.785, 0.7848, 0.7847, 0.7846, 0.7836000000000001, 0.7835000000000001, 0.7834000000000001, 0.7834000000000001, 0.7834000000000001,
       0.7834000000000001, 0.7824000000000001, 0.7814000000000001, 0.7814000000000001, 0.7814000000000001, 0.7814000000000001, 0.7804000000000001, 0.7804000000000001, 0.7804000000000001,
       0.7794000000000001, 0.7794000000000001, 0.7784000000000001, 0.7784000000000001, 0.7774000000000001, 0.7774000000000001, 0.7773900000000001, 0.7773800000000002, 0.7773700000000002,
       0.7773600000000003, 0.7773500000000003, 0.7773400000000004, 0.7773300000000004, 0.7773200000000005, 0.7773100000000005, 0.7773000000000005, 0.7772900000000006]



nodes1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
qdata1=[0.795, 0.784, 0.78, 0.778, 0.777, 0.776, 0.775, 0.7748,0.7747, 0.7746, 0.7736000000000001, 0.7735000000000001, 0.7734000000000001, 0.7734000000000001, 0.7734000000000001,
       0.7734000000000001, 0.7724000000000001, 0.7714000000000001, 0.7714000000000001, 0.7714000000000001, 0.7714000000000001, 0.7704000000000001, 0.7704000000000001, 0.7704000000000001,
       0.7794000000000001, 0.7794000000000001, 0.7774000000000001, 0.7774000000000001,  0.7674000000000001, 0.7674000000000001, 0.7673900000000001, 0.7673800000000002, 0.7673700000000002,
       0.7673600000000003, 0.7673500000000003, 0.7673400000000004, 0.7673300000000004, 0.7673200000000005, 0.7673100000000005, 0.7673000000000005, 0.7672900000000006]



plt.plot(nodes, qdata1, color='grey')
plt.plot(nodes, qdata, color='blue')
plt.xticks(np.arange(min(nodes), max(nodes)+1, 10.0))
plt.plot(nodes, qdata, color='blue')

plt.xticks(np.arange(min(nodes1), max(nodes1)+1, 10.0))
plt.plot(nodes, qdata, color='blue',label='New Epoch Value')

plt.plot(nodes, qdata1, color='grey',label='Old Epoch Value')
plt.xlabel('epoch (Iteration)')
plt.ylabel('RMSE (%)')
plt.title('Movielens 10M')
plt.legend()
plt.show()
'''



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''from matplotlib import pyplot as plt

fig = plt.figure()
labels = ["Old Epoch Value", "New Epoch Value"]
ax = fig.add_subplot(111, frame_on=False)
ax.axis("off")

nodes,qdata=Processor.epochcal()
nodes1,qdata1=Processor.epochcal1()
ax1 = fig.add_subplot(211)
ax1.plot(nodes, qdata, color='blue', label=labels[0])
ax1.set_title('Movielens 10M')
ax1.legend(loc="upper right")
#ax1.set_ylim(0.7,0.8)

ax2 = fig.add_subplot(212)
ax2.plot(nodes1, qdata1, color='grey',label=labels[1])

ax2.legend(loc="upper right")
#ax2.set_ylim(0.7,0.8)

ax.set_xlabel('epoch (Iteration)')
ax.set_ylabel('RMSE (%)')
plt.title('epoch (Iteration)')

plt.show()'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()

s = "Crime Type Summer|Crime Type Winter".split("|")

# Generate dummy data into a dataframe
j = {x: [random.choice(["Multi-Views NN", "Top Popular", "Biased MF", "SVD++", "Movie Average"]) for j in range(300)] for x in s}
df = pd.DataFrame(j)

index = np.arange(5)
bar_width = 0.35

fig, ax = plt.subplots()
#summer = ax.bar(index, df["Crime Type Summer"].value_counts(), bar_width,label="Summer")
summer = ax.bar(index, df["Crime Type Summer"].value_counts(), bar_width,label="Summer")

winter = ax.bar(index+bar_width, df["Crime Type Winter"].value_counts(),bar_width, label="Winter")

ax.set_xlabel('N-Iterations')
ax.set_ylabel('Recall Percentage')
ax.set_title('Movielens 10M')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["Multi-Views NN", "Top Popular", "Biased MF", "SVD++", "Movie Average"])
ax.legend()

plt.show()'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''import numpy as np 
import matplotlib.pyplot as plt 
   
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
   
# set height of bar
nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()
IT = [val1[9], val2[9], val3[9], val4[9], val5[9]]
ECE = [val10[9], val9[9], val8[9], val7[9], val6[9]]
#IT = [val1, val2, val3, val4, val5]
#ECE = [val6, val7, val8, val9, val10] 
#CSE = [29, 3, 24, 25, 17] 
   
# Set position of bar on X axis
br1 = np.arange(5)
br2 = [x + barWidth for x in br1] 
#br3 = [x + barWidth for x in br2] 
   
# Make the plot 
plt.bar(br1, IT, color ='b', width = barWidth,edgecolor ='grey', label ='OLD Recall Values') 
plt.bar(br2, ECE, color ='g', width = barWidth,edgecolor ='grey', label ='NEW Recall Values') 
#plt.bar(br3, CSE, color ='b', width = barWidth,edgecolor ='grey', label ='CSE') 
   
# Adding Xticks  
plt.xlabel('N-Iterations', fontweight ='bold') 
plt.ylabel('Recall Percentage', fontweight ='bold') 
plt.xticks([r + barWidth for r in range(len(IT))],["Multi-Views NN", "Top Popular", "Biased MF", "SVD++", "Movie Average"]) 
plt.title('Movielens 10M')
plt.legend()
plt.show() '''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''import pandas as pd 
import matplotlib.pyplot as plt 

nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()
print(val1)
  
# create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
print(data)  
# dataframe created with 
# the above data array
df = pd.DataFrame(data, columns = ['EMPID', 'Gender','Age', 'Sales','BMI', 'Income'] ) 
df.plot.bar() 
# plot between 2 attributes 
plt.bar(df['Age'], df['Sales']) 
plt.xlabel('N-Iterations')
plt.ylabel('Recall Percentage')
plt.title('Movielens 10M')
plt.show() '''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''from matplotlib import pyplot as plt


nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()

fig = plt.figure()
labels = ["Old Epoch Value", "New Epoch Value"]
ax = fig.add_subplot(333)

ax.axis("off")
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax1 = fig.add_subplot(211)
#ax1.plot(nodes, val5, color='blue', label=labels[0])
ax1.plot(nodes, val5, color='blue', label='New Multi-Views NN')
ax1.plot(nodes, val10, color='grey',label='Old Multi-Views NN')
ax1.set_title('Movielens 10M')
ax1.legend(loc="upper right")
#ax1.set_ylim(0.7,0.8)

ax2 = fig.add_subplot(212)
#ax2.plot(nodes, val1, color='grey',label=labels[1])
ax2.plot(nodes, val4, color='blue', label='New Top Popular')
ax2.plot(nodes, val9, color='grey',label='Old Top Popular')

ax2.legend(loc="upper right")
#ax2.set_ylim(0.7,0.8)

ax3 = fig.add_subplot(221)
#ax2.plot(nodes, val1, color='grey',label=labels[1])
ax3.plot(nodes, val3, color='blue', label='New Top Popular')
ax3.plot(nodes, val8, color='grey',label='Old Top Popular')

ax2.legend(loc="upper right")


ax.set_xlabel('epoch (Iteration)')
ax.set_ylabel('RMSE (%)')
plt.title('epoch (Iteration)')

plt.show()'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import matplotlib.pyplot as plt
fig,a =  plt.subplots(2,3)
import numpy as np
fig.tight_layout(pad=1.0)
#x = np.arange(1,5)
nodes,val1,val2,val3,val4,val5=Processor.rmsecal()
nodes,val6,val7,val8,val9,val10=Processor.rmsecalnew()
a[0][0].plot(nodes,val5,label='New Multi-Views NN',color='green')
a[0][0].plot(nodes,val10,label='Old Multi-Views NN',color='grey')
a[0][0].set_title('Multi-Views NN')
a[0][0].set_xlabel('N-Iterations')
a[0][0].set_ylabel('Recall Percentage')

a[0][1].plot(nodes,val4,color='yellow',label='New Top Popular')
a[0][1].plot(nodes,val9,color='grey',label='Old Top Popular')
a[0][1].set_title('Top Popular')
a[0][1].set_xlabel('N-Iterations')
a[0][1].set_ylabel('Recall Percentage')

a[0][2].plot(nodes,val1,color='grey',label='Old Movie Average')
a[0][2].plot(nodes,val6,color='blue',label='New Movie Average')
a[0][2].set_title('Movie Average')
a[0][2].set_xlabel('N-Iterations')
a[0][2].set_ylabel('Recall Percentage')


a[1][0].plot(nodes,val3,color='grey',label='Old Biased MF')
a[1][0].plot(nodes,val8,color='purple',label='New Biased MF')
a[1][0].set_title('Biased MF')
a[1][0].set_xlabel('N-Iterations')
a[1][0].set_ylabel('Recall Percentage')

a[1][1].plot(nodes,val2,color='grey',label='Old SVD++')
a[1][1].plot(nodes,val7,color='orange',label='New SVD++')
a[1][1].set_title('SVD++')
a[1][1].set_xlabel('N-Iterations')
a[1][1].set_ylabel('Recall Percentage')


a[1][2].plot(nodes,val10,color='green',label='Multi-Views NN')
a[1][2].plot(nodes,val9,color='yellow',label='Top Popular')
a[1][2].plot(nodes,val8,color='purple',label='Biased MF')
a[1][2].plot(nodes,val7,color='orange',label='SVD++')
a[1][2].plot(nodes,val6,color='blue',label='Movie Average')
a[1][2].set_title('Movielens 10M')
a[1][2].set_xlabel('N-Iterations')
a[1][2].set_ylabel('Recall Percentage')


a[0][0].legend(loc=0)
a[0][1].legend(loc=0)
a[1][0].legend(loc=0)
a[1][1].legend(loc=0)
a[0][2].legend(loc=0)
a[1][2].legend(loc=0)
plt.show()
