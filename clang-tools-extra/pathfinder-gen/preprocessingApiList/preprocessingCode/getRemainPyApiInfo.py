atFunctionList=open('../pyApiFile/py_api_list','r')
differenceSet_pyApi=open('../differenceSet_pyApi','r')
w=open('../remainPyApiInfo','w')
 
#파일에서 읽은 라인들을 리스트로 읽어들임
atFunctionLines = atFunctionList.readlines()
remainPyApiLines = differenceSet_pyApi.readlines()

libraryPath = "temp"
currentLibraryCount = 0
outputLines = []

for line in atFunctionLines:
    currentLineList = line.split('.')
    if currentLineList[-2] + '\n' in remainPyApiLines:
        # get library path
        currentLibraryPath = currentLineList[0]
        for word in currentLineList[1:-2]:
            currentLibraryPath = currentLibraryPath + "."
            currentLibraryPath = currentLibraryPath + word
        if libraryPath != currentLibraryPath:
            if libraryPath != "temp":
                outputLines.append('----------------------------------------------------------------------\n')
                outputLines.append("Path: " + libraryPath + ", Count: " + str(currentLibraryCount) + '\n')
                outputLines.append('----------------------------------------------------------------------\n')
                outputLines.append('\n')
            libraryPath = currentLibraryPath
            currentLibraryCount = 0
        outputLines.append(line)
        currentLibraryCount = currentLibraryCount + 1

outputLines.append('----------------------------------------------------------------------\n')
outputLines.append("Path: " + libraryPath + ", Count: " + str(currentLibraryCount) + '\n')
outputLines.append('----------------------------------------------------------------------\n')
outputLines.append('\n')

for line in outputLines:
    w.write(line)


#파일 닫기
atFunctionList.close()
differenceSet_pyApi.close()
w.close()