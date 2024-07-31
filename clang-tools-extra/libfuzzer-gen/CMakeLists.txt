set(LLVM_LINK_COMPONENTS support)

add_clang_executable(libfuzzer-gen
  libfuzzerGen.cpp
  extract.cpp
  api.cpp
  param.cpp
  utils.cpp
  )
target_link_libraries(libfuzzer-gen
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )