atApiName=open('../atLibraryFile/at_function_names','r')
pyApiName=open('../pyApiFile/py_api_names', 'r')
intersection=open('../intersection_at_pyApi','w')
differenceSet_pyApi=open('../differenceSet_pyApi','w')
differenceSet_at=open('../differenceSet_at','w')
 
#파일에서 읽은 라인들을 리스트로 읽어들임
atNamesLine = atApiName.readlines()
pyNamesLine = pyApiName.readlines()

pyNamesLine = list(set(pyNamesLine))



# 교집합
intersectionLine = list(set(pyNamesLine) & set(atNamesLine))
intersectionLine.sort()

# 차집합
differenceSetAt = list(set(atNamesLine) - set(pyNamesLine))
differenceSetAt.sort()
differenceSetPy = list(set(pyNamesLine) - set(atNamesLine))
differenceSetPy.sort()

for line in intersectionLine:
    intersection.write("at::" + line)
for line in differenceSetAt:
    differenceSet_at.write("at::" + line)
for line in differenceSetPy:
    differenceSet_pyApi.write(line)

#파일 닫기
atApiName.close()
pyApiName.close()
intersection.close()
differenceSet_pyApi.close()
differenceSet_at.close()
