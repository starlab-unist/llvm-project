from enum import Enum

class getApiList:
    def __init__(self, pyApiListPath):
        self.pyApiDict = self.preprocessPythonApi(pyApiListPath)
        self.pyPathList = self.pyApiDict.keys()
    
    def preprocessPythonApi(self, pyApiListPath):
        listFile = open(pyApiListPath, 'r')
        apiLines = listFile.readlines()
            
        result = dict()
            
        for line in apiLines:
            splitLine = line.split('.')
            currentPath = ""
            for index, word in enumerate(splitLine):
                currentPath = currentPath + word
                if splitLine[index+2] == "json\n":
                    if currentPath not in result:
                        result[currentPath] = [splitLine[index+1]]
                    else:
                        result[currentPath].append(splitLine[index+1])
                    break
                else:
                    currentPath = currentPath + "."
        result = dict(sorted(result.items()))
        
        for key in result:
            result[key].sort()
            
        listFile.close()

        return result
    
    def getIntersectionApiList(self, apiNamespace ,apiListPath):
        intersectionApiList = []
        preprocessedApiList = self.preprocessApiList(apiNamespace, apiListPath)
        
        if apiNamespace == "at" or apiNamespace == "at::native":
            for key in self.pyApiDict:
                intersectionApiList.extend(self.getIntersectionList(preprocessedApiList, self.pyApiDict[key]))
        else:
            convertedApiNameSpace = apiNamespace.replace("::", ".")
            intersectionApiList.extend(self.getIntersectionList(preprocessedApiList, self.pyApiDict[convertedApiNameSpace]))
        
        intersectionApiList = list(set(intersectionApiList))
        intersectionApiList.sort()
                
        self.makeOutputFile("../IntersectionList/intersection_" + apiNamespace, intersectionApiList, apiNamespace + "::")
        
        return intersectionApiList
        
    def preprocessApiList(self, apiNamespace ,apiListPath):
        if apiNamespace == "at":
            return self.preprocessAtApi(apiListPath, 4)           
        elif apiNamespace == "at::native":
            return self.preprocessAtApi(apiListPath, 12)
        else:
            return self.preprocessAnotherApi(apiNamespace, apiListPath)
        
    def getIntersectionList(self, firstApiList, secondApiList):
        intersectionList = list(set(firstApiList) & set(secondApiList))
        intersectionList.sort()
        return intersectionList
    
    def makeOutputFile(self, fileName, lineList, additionalStr = ""):
        outputFile = open(fileName, "w")
        for line in lineList:
            outputFile.write(additionalStr + line + '\n')
        outputFile.close()
    
    def preprocessAtApi(self, apiListPath, removeLength):
        listFile = open(apiListPath, 'r')
        apiLines = listFile.readlines()
        
        result = []
        for line in apiLines:
            splitLine = line[removeLength:]
            if splitLine.find('(') != -1:
                splitLine = splitLine[:splitLine.find('(')]
            else:
                splitLine = splitLine[:-1]
            result.append(splitLine)
        result = list(set(result))
        result.sort()
        
        if removeLength == 4:
            self.makeOutputFile("../CppLibraryList/at_function_list_name", result)
        elif removeLength == 12:
            self.makeOutputFile("../CppLibraryList/at::native_function_list_name", result)
        
        listFile.close()
        
        return result
    

    def preprocessAnotherApi(self, apiNamespace, apiListPath):
        #print(type(apiListPath))
        listFile = open(apiListPath, 'r')
        apiLines = listFile.readlines()
        
        result = []
        for line in apiLines:
            if line == '\n':
                continue
            if line.find('(') != -1:
                splitLine = line[:line.find('(')]
            else:
                splitLine = line[:-1]
            splitLine = splitLine.split(" ")[-1]
            splitLine = splitLine.replace((apiNamespace + "::"), "")
            if apiListPath == "torch::nn" and "Options" in splitLine:
                continue
            
            result.append(splitLine)

        result = list(set(result))
        result.sort()
        
        self.makeOutputFile("../CppLibraryList/" + apiListPath + "_name", result)
        
        listFile.close()
        
        return result