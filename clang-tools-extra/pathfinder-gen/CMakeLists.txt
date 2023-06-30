set(LLVM_LINK_COMPONENTS support)

add_clang_executable(pathfinder-gen
  pathfinderGen.cpp
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