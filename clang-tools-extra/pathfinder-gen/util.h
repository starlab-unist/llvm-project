#ifndef PATHFINDER_GEN_UTIL
#define PATHFINDER_GEN_UTIL

#include <sys/stat.h>
#include <sys/types.h>
#include <string>

const dir_id = 0;

void mk_fuzz_target_dir() {
  std::string dir_name = 

  struct stat St;
  if (stat(Path.c_str(), &St))
    return false;
  return S_ISDIR(St.st_mode);
}

#endif