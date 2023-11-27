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
#include <memory>

#define PFGEN_CHECK(cond, msg)     \
  if (!(cond)) {                   \
    std::cerr << msg << std::endl; \
    exit(0);                       \
  }

static const std::string space = " ";
static const std::string gte = " >= ";
static const std::string assign = " = ";
static const std::string comma = ", ";
static const std::string semicolon = ";";
static const std::string newline = "\n";

bool startswith(std::string base, std::string prefix);
bool endswith(std::string base, std::string suffix);
bool include(std::vector<std::string>& vec, std::string name);
std::string strip_ext(std::string filename);
std::string torch_module_list_file_name();
void init_torch_api_list();
const std::map<std::string, std::set<std::string>>& get_torch_function_list();
const std::set<std::string>& get_torch_module_list();
void make_dir(std::string dir);
std::string unique_name(std::string name, std::set<std::string>& names_seen);

std::string quoted(std::string param_name);
std::string sq_quoted(std::string param_name);
std::string bracket(std::string str="");
std::string square(std::string str);
std::string curly(std::string str);
std::string join_strs(const std::vector<std::string>& strs, std::string sep=comma);
void concat(
  std::vector<std::string>& left,
  const std::string& prefix,
  const std::vector<std::string>& right,
  const std::string& postfix="");

template<typename T>
void concat(std::vector<T>& left, const std::vector<T>& right) {
  for (auto& elem: right)
    left.push_back(elem);
}

template<typename T>
std::string to_string(const std::vector<std::unique_ptr<T>>& params) {
  std::string str;
  for (size_t i = 0; i < params.size(); i++) {
    str += params[i]->expr();
    if (i != params.size() - 1)
      str += comma;
  }
  return curly(str);
}

#endif