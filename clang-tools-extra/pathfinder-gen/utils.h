#ifndef PATHFINDER_GEN_UTILS
#define PATHFINDER_GEN_UTILS

#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <dirent.h>
#include <fstream>

bool startswith(std::string base, std::string prefix);
bool endswith(std::string base, std::string suffix);
bool include(std::vector<std::string>& vec, std::string name);
std::string strip_ext(std::string filename);
std::string torch_module_list_file_name();
void init_torch_api_list();
const std::map<std::string, std::set<std::string>>& get_torch_function_list();
const std::set<std::string>& get_torch_module_list();
void write_recursive(const std::map<std::string, std::map<std::string, std::string>>& contents);

#endif