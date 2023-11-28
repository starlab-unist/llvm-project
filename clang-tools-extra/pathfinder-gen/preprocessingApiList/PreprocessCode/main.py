from getApiList import getApiList

apiPathExtractor = getApiList("../pyApiFile/py_api_list")

# Get python api list's namespace and api
outputFile = open("../IntersectionList/pyApiPath", "w")
outputFile2 = open("../IntersectionList/pyApiPath2", "w")

for key in apiPathExtractor.pyApiDict:
    outputFile.write("Path: " + key + '\n')
    for api in apiPathExtractor.pyApiDict[key]:
        outputFile.write("  " + api + '\n')
    outputFile.write('\n')

for line in apiPathExtractor.pyPathList:
    outputFile2.write(line + '\n')
    
outputFile.close()
outputFile2.close()

# Get intersection Api list
apiPathExtractor.getIntersectionApiList("at::Tensor", "../CppLibraryList/at::Tensor_function_list")
apiPathExtractor.getIntersectionApiList("at", "../CppLibraryList/at_function_list")
apiPathExtractor.getIntersectionApiList("at::native", "../CppLibraryList/at::native_function_list")
apiPathExtractor.getIntersectionApiList("torch", "../CppLibraryList/torch_function_list")
apiPathExtractor.getIntersectionApiList("torch::fft", "../CppLibraryList/torch::fft_function_list")
apiPathExtractor.getIntersectionApiList("torch::jit", "../CppLibraryList/torch::jit_function_list")
apiPathExtractor.getIntersectionApiList("torch::linalg", "../CppLibraryList/torch::linalg_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn", "../CppLibraryList/torch::nn_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn::functional", "../CppLibraryList/torch::nn::functional_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn::init", "../CppLibraryList/torch::nn::init_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn::parallel", "../CppLibraryList/torch::nn::parallel_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn::utils", "../CppLibraryList/torch::nn::utils_function_list")
apiPathExtractor.getIntersectionApiList("torch::nn::utils::rnn", "../CppLibraryList/torch::nn::utils::rnn_function_list")
apiPathExtractor.getIntersectionApiList("torch::optim", "../CppLibraryList/torch::optim_function_list")
apiPathExtractor.getIntersectionApiList("torch::special", "../CppLibraryList/torch::special_function_list")


'''
at
at::native
at::Tensor
torch
torch::fft
torch::jit
torch::linalg
torch::nn
torch::nn::functional
torch::nn::init
torch::nn::parallel
torch::nn::utils
torch::nn::utils::rnn
torch::optim
torch::sparse // x
torch::special
'''


