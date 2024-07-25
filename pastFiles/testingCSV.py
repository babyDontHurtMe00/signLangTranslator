import sys

# adding Folder_2/subfolder to the system path
sys.path.insert(0, 'D:\Projects 2\mediaPipeModel')

import csvLibrary

# csvLibrary.info()
# csvLibrary.lwrite(["field1","field2","field3"],[[1,2,3],[4,5,6],[7,8,9]],"./trainingData/test0.csv")

# csvLibrary.dwrite(["field1","field2","field3"],[{"field1":1,"field2":2,"field3":3},{"field1":4,"field2":5,"field3":6},{"field1":7,"field2":8,"field3":9}],"./trainingData/test3.csv")
csvLibrary.dappend(["field1","field2","field3"],[{"field1":1,"field2":2,"field3":3},{"field1":4,"field3":6},{"field1":7,"field2":8,"field3":9}],"./trainingData/test3.csv")
csvLibrary.dread("./trainingData/test3.csv")