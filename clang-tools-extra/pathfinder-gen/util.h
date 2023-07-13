#ifndef PATHFINDER_GEN_UTIL
#define PATHFINDER_GEN_UTIL

#include <sys/stat.h>
#include <sys/types.h>
#include <string>

bool include(std::vector<std::string>& vec, std::string name) {
  for (auto n: vec) {
    if (n == name)
      return true;
  }

  return false;
}

bool include(std::vector<std::pair<std::string,std::unique_ptr<Param>>>& vec, std::string name) {
  for (auto&& p: vec) {
    if (p.first == name)
      return true;
  }

  return false;
}

void push_back_unique(std::vector<std::pair<std::string,std::unique_ptr<Param>>>& vec, std::string name, std::unique_ptr<Param> param) {
  if (!include(vec, name))
    vec.push_back({name, std::move(param)});
}

std::string strip_ext(std::string filename) {
  return filename.substr(0,filename.find_last_of("."));
}

#endif