#ifndef PATHFINDER_GEN
#define PATHFINDER_GEN

#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace clang;

enum ParamType {
  INT,
  BOOL,
  ENUM,
  TENSOR,
  EXPENDINGARRAY,
  VARIANT,
  API_OPTION,
};

class Param {
  public:
    Param(ParamType ptype_): ptype(ptype_) {
      assert(ptype == INT || ptype == BOOL || ptype == TENSOR);
    }
    Param(ParamType ptype_, std::string enum_name_)
      : ptype(ptype_), enum_name(enum_name_)
    { assert(ptype == ENUM); }
    Param(ParamType ptype_, long expendingarray_size_)
      : ptype(ptype_), expendingarray_size(expendingarray_size_)
    { assert(ptype == EXPENDINGARRAY); }
    Param(ParamType ptype_, std::vector<std::unique_ptr<Param>> variant_types_)
      : ptype(ptype_), variant_types(std::move(variant_types_)) { assert(ptype == VARIANT); }
    Param(
      ParamType ptype_,
      std::string api_option_name_,
      std::map<std::string,std::unique_ptr<Param>> api_option_types_)
      : ptype(ptype_), api_option_name(api_option_name_), api_option_types(std::move(api_option_types_))
    { assert(ptype == API_OPTION); }
    std::string to_string(int depth=0);
    void set_default(std::string param_name, CXXRecordDecl* cdecl);
  private:
    ParamType ptype;
    std::string enum_name;
    long expendingarray_size;
    std::vector<std::unique_ptr<Param>> variant_types;
    std::string api_option_name;
    std::map<std::string,std::unique_ptr<Param>> api_option_types;

    Optional<unsigned long> default_num = None;
    Optional<std::string> default_enum = None;
    bool default_tensor = false;
};

std::string str_mult(size_t n, std::string str) {
  std::string s;
  for (size_t i = 0; i < n; i++)
    s += str;
  return s;
}

std::string Param::to_string(int depth) {
  static const std::string indent = std::string("  ");

  std::string str;
  switch (ptype) {
    case INT:
      str += str_mult(depth, indent) + "INT";
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "\n";
      break;
    case BOOL:
      str += str_mult(depth, indent) + "BOOL";
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "\n";
      break;
    case ENUM:
      str += str_mult(depth, indent) + "ENUM: " + enum_name;
      if (default_enum)
        str += ": default= " + default_enum.getValue();
      str += "\n";
      break;
    case TENSOR:
      str += str_mult(depth, indent) + "TENSOR";
      if (default_tensor)
        str += ": default=Tensor()";
      str += "\n";
      break;
    case EXPENDINGARRAY:
      str += str_mult(depth, indent) + "EXPENDINGARRAY " + std::to_string(expendingarray_size);
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "\n";
      break;
    case VARIANT: {
      str += str_mult(depth, indent) + "VARIANT";
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "\n";
      for (auto&& t: variant_types)
        str += t->to_string(depth+1);
      break;
    }
    case API_OPTION: {
      str += str_mult(depth, indent) + "API_OPTION: " + api_option_name + "\n";
      for (auto&& entry: api_option_types) {
        str += str_mult(depth+1, indent) + entry.first + "\n";
        str += entry.second->to_string(depth+2);
      }
      break;
    }
    default: assert(false);
  }
  return str;
}

void Param::set_default(std::string param_name, CXXRecordDecl* cdecl) {
  std::cout << "param name: " << param_name << std::endl;
  for (auto field: cdecl->fields()) {
    std::string field_name = field->getNameAsString();
    std::cout << "field name: " << field_name << std::endl;
    if (param_name.length() + 1 == field_name.length() &&
        field_name.compare(0, field_name.length(), param_name + "_") == 0) {
      switch (ptype) {
        case TENSOR:
          default_tensor = true;
          break;
        case ENUM:
          assert(false); // TODO;
          break;
        default:
          auto e = field->getInClassInitializer()->IgnoreUnlessSpelledInSource();
          auto il = dyn_cast<IntegerLiteral>(e);
          assert (il != nullptr);
          unsigned long val = il->getValue().getZExtValue();
          std::cout << "default value: " << std::to_string(val) << std::endl;
          default_num = val;
      }
      //field->dump();
    }

    
    //field->getInClassInitializer()->dump();
  }

}

#endif
