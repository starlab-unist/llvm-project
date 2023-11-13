atFunctionList=open('../atLibraryFile/at_function_list','r')
w=open('../atLibraryFile/at_function_names','w')

#파일에서 읽은 라인들을 리스트로 읽어들임
atFunctionLines = atFunctionList.readlines()

atFunctionPreprocessed = []

for line in atFunctionLines:
    currentLine = line[4:]
    if currentLine.find('(') != -1:
        currentLine = currentLine[:currentLine.find('(')]
    else:
        currentLine = currentLine[:-1]
    atFunctionPreprocessed.append(currentLine)

atFunctionPreprocessed = list(set(atFunctionPreprocessed))
atFunctionPreprocessed.sort()

#중복제거한 리스트 파일 쓰기
for line in atFunctionPreprocessed:
    w.write(line + '\n')
    
 
#파일 닫기
atFunctionList.close()
w.close()