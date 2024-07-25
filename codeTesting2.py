import codeTesting as cT

# cT.createResultCSV()
cT.load_saved_artifacts()

############################################################ Getting Output by asking Model
# THE LINE TO GENERATE OUPUT
#result=class_number_to_name(__model.predict(final)[0])
# here final means each array [0 1 2 122 0 301...] inside X



import numpy as np
import csvLibrary as cl
X=[]
y=[]

      
path = "./currentOutput.csv"
# print(path[15])
csvData = cl.dread(path)

for rowEntry in csvData:
    # print(rowEntry)
    tempRow = []
    for handPoint, coords in rowEntry.items():
        if coords != '':
            coords = list(coords[1:-1].split(","))
            # print(float(handPoint), int(coords[0]),int(coords[1]))
            tempRow.extend([float(handPoint), int(coords[0]),int(coords[1])])
        else:
            tempRow.extend([float(handPoint),-600,-600])
    X.append(tempRow)
    final = np.array(tempRow).reshape(1,126) #1 row and 3*42 entries
# tempRow
# final
    y.append(cT.__model.predict(final)[0])

# print("X[0]: ",X[0])
# print("\n\nY: ",y,len(y))
count = {}
for i in y:
    count[i]=y.count(i)
print([key for key in count if all(count[temp] <= count[key] for temp in count)])
