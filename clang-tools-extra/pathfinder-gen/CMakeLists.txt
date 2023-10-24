set(LLVM_LINK_COMPONENTS support)

add_clang_executable(pathfinder-gen
  pathfinderGen.cpp
  extract.cpp
  api.cpp
  param.cpp
  utils.cpp
  )
target_link_libraries(pathfinder-gen
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )