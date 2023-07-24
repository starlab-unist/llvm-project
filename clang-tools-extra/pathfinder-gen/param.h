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

const size_t MAX_RANK = 5;
const size_t MAX_VECTOR_SIZE = 5;
const size_t DOUBLE_DICT_SIZE = 20;

size_t tensor_id;
size_t int_vector_id;
size_t float_vector_id;
size_t array_id;
size_t enum_id;

enum ParamType {
  INT,
  BOOL,
  FLOAT,
  DTYPE,
  ENUM,
  TENSOR,
  INTVECTOR,
  FLOATVECTOR,
  INTARRAYREF,
  EXPANDINGARRAY,
  OPTIONAL,
  VARIANT,
  MAP,
};

class Param {
  public:
    Param(ParamType ptype_): ptype(ptype_) {
      assert(ptype == INT || ptype == BOOL || ptype == FLOAT || ptype == DTYPE ||
             ptype == TENSOR || ptype == INTVECTOR || ptype == FLOATVECTOR || ptype == INTARRAYREF);
      if (ptype == INT || ptype == BOOL || ptype == FLOAT || ptype == DTYPE) {
        offset_size = 1;
      } else if (ptype == TENSOR || ptype == INTVECTOR || ptype == FLOATVECTOR || ptype == INTARRAYREF) {
        offset_size = 6;
      }
    }
    Param(ParamType ptype_, std::string enum_name_)
      : ptype(ptype_), enum_name(enum_name_)
    { assert(ptype == ENUM); }
    Param(ParamType ptype_, Optional<size_t> rank)
      : ptype(ptype_)
    {
      assert(ptype == TENSOR);
      tensor_rank = rank;
      if (tensor_rank) {
        offset_size = tensor_rank.getValue();
      } else {
        offset_size = 6;
      }
    }
    Param(ParamType ptype_, long size)
      : ptype(ptype_)
    {
      assert(ptype == EXPANDINGARRAY);
      expandingarray_size = size;
      offset_size = expandingarray_size;
    }
    Param(ParamType ptype_, std::unique_ptr<Param> base_)
      : ptype(ptype_), base(std::move(base_))
    {
      assert(ptype == OPTIONAL);
      offset_size = base->offset_size + 1;
    }
    Param(ParamType ptype_, std::vector<std::unique_ptr<Param>> enums_, std::unique_ptr<Param> expandingarray_)
      : ptype(ptype_), enums(std::move(enums_)), expandingarray(std::move(expandingarray_))
    {
      assert(ptype == VARIANT);
      assert((enums.size() > 0 && expandingarray == nullptr) || (enums.size() == 0 && expandingarray != nullptr));
      if (enums.size() > 0) {
        offset_size = 1;
      } else {
        offset_size = expandingarray->offset_size;
      }
    }
    Param(
      ParamType ptype_,
      std::string map_name_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries_)
      : ptype(ptype_), map_name(map_name_), ctor_params(std::move(ctor_params_)), entries(std::move(entries_))
    {
      assert(ptype == MAP);
      offset_size = 0;
      for (auto&& p: entries)
        offset_size += p.second->offset_size;
    }
    std::string to_string(int depth=0);
    void set_default(std::string param_name, const CXXRecordDecl* cdecl);
    size_t set_offset(size_t offset);
    void constraint(std::vector<std::string>& strs);
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>> to_code();
  private:
    ParamType ptype;

    // INT
    Optional<size_t> default_int = None;
    size_t offset_start;
    size_t offset_size;

    // ENUM
    std::string enum_name;

    // TENSOR
    Optional<size_t> tensor_rank = None;

    // EXPENDINGARRAY
    size_t expandingarray_size;

    // OPTIONAL
    std::unique_ptr<Param> base;

    // VARIANT
    std::vector<std::unique_ptr<Param>> enums;
    std::unique_ptr<Param> expandingarray;

    // MAP
    std::string map_name;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries;

    friend size_t set_idx(std::vector<std::unique_ptr<Param>>& params);
    friend void gen_pathfinder_fuzz_target(
      std::string target_api_name,
      std::vector<std::unique_ptr<Param>>& params,
      std::ostream& os);
    friend Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, ASTContext &Ctx);
};

size_t Param::set_offset(size_t offset) {
  switch(ptype) {
    case INT:
    case BOOL:
    case FLOAT:
    case DTYPE:
    case TENSOR:
    case INTVECTOR:
    case FLOATVECTOR:
    case INTARRAYREF:
    case EXPANDINGARRAY:
      offset_start = offset;
      offset += offset_size;
      break;
    case OPTIONAL:
      offset_start = offset;
      offset = base->set_offset(offset+1);
      break;
    case VARIANT:
      if (enums.size() > 0) {
        offset_start = offset;
        offset += offset_size;
      } else if (expandingarray != nullptr) {
        offset = expandingarray->set_offset(offset);
      }
      break;
    case MAP:
      for (auto&& p: ctor_params)
        offset = p.second->set_offset(offset);
      for (auto&& p: entries)
        offset = p.second->set_offset(offset);
      break;
    default:
      assert(false);
  }
  return offset;
}

void Param::set_default(std::string param_name, const CXXRecordDecl* cdecl) {
  Expr* e = nullptr;
  for (auto field: cdecl->fields()) {
    std::string field_name = field->getNameAsString();
    if (param_name.length() + 1 == field_name.length() &&
        field_name.compare(0, field_name.length(), param_name + "_") == 0) {
      e = field->getInClassInitializer()->IgnoreUnlessSpelledInSource();
    }
  }

  if (e !=nullptr) {
    switch(ptype) {
      case INT:
      case EXPANDINGARRAY:
        if (const auto* il = dyn_cast<IntegerLiteral>(e)) {
          unsigned long val = il->getValue().getZExtValue();
          default_int = val;
        }
        break;
      default:
        break;
    }
  }
}

void Param::constraint(std::vector<std::string>& strs) {
  switch(ptype) {
    case INT: {
      int min = default_int ? default_int.getValue() : 0;
      std::string arg = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back(arg + " >= " + std::to_string(min));
      break;
    }
    case BOOL: {
      std::string arg = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + arg + ", " + arg + " <= 1");
      break;
    }
    case FLOAT: {
      std::string arg = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + arg + ", " + arg + " < " + std::to_string(DOUBLE_DICT_SIZE));
      break;
    }
    case DTYPE: {
      std::string arg = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("DtypeFirst <= " + arg + ", " + arg + " < DtypeLast");
      break;
    }
    case TENSOR: {
      if (tensor_rank) {
        for (size_t i = offset_start; i < offset_start + offset_size; i++)
          strs.push_back("arg[" + std::to_string(i) + "] >= 1");
      } else {
        std::string rank = "arg[" + std::to_string(offset_start) + "]";
        strs.push_back("0 <= " + rank + ", " + rank + " <= " + std::to_string(MAX_RANK));
        for (size_t i = offset_start + 1; i < offset_start + offset_size; i++)
          strs.push_back("arg[" + std::to_string(i) + "] >= 1");
      }
      break;
    }
    case INTVECTOR: {
      std::string size = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + size + ", " + size + " <= " + std::to_string(MAX_VECTOR_SIZE));
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++)
        strs.push_back("arg[" + std::to_string(i) + "] >= 0");
      break;
    }
    case FLOATVECTOR: {
      std::string size = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + size + ", " + size + " <= " + std::to_string(MAX_VECTOR_SIZE));
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++)
        strs.push_back("0 <= arg[" + std::to_string(i) + "], arg[" + std::to_string(i) + "] < " + std::to_string(DOUBLE_DICT_SIZE));
      break;
    }
    case INTARRAYREF: {
      std::string size = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + size + ", " + size + " <= " + std::to_string(MAX_VECTOR_SIZE));
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++)
        strs.push_back("arg[" + std::to_string(i) + "] >= 0");
      break;
    }
    case EXPANDINGARRAY: {
      int min = default_int ? default_int.getValue() : 0;
      for (size_t i = offset_start; i < offset_start + offset_size; i++)
        strs.push_back("arg[" + std::to_string(i) + "] >= " + std::to_string(min));
      break;
    }
    case OPTIONAL: {
      std::string is_some = "arg[" + std::to_string(offset_start) + "]";
      strs.push_back("0 <= " + is_some + ", " + is_some + " <= 1");
      base->constraint(strs);
      break;
    }
    case VARIANT: {
      if (enums.size() > 0) {
        std::string arg = "arg[" + std::to_string(offset_start) + "]";
        strs.push_back("0 <= " + arg + ", " + arg + " < " + std::to_string(enums.size()));
      } else {
        expandingarray->constraint(strs);
      }
      break;
    }
    case MAP: {
      for (auto&& p: ctor_params)
        p.second->constraint(strs);
      for (auto&& p: entries)
        p.second->constraint(strs);
      break;
    }
    default:
      assert(false);
  }
}

std::string gen_setup(size_t offset_size, std::vector<std::unique_ptr<Param>>& params) {
  std::string code;
  code += "void PathFinderSetup() {\n";
  code += "  PathFinderSetArgSize(" + std::to_string(offset_size) + ");\n";
  //code += "  PathFinderWeakArg(arg[0], Float32);\n";
  code += "  PathFinderAddConstraint({\n";
  code += "    DtypeFirst <= arg[0], arg[0] <= DtypeLast,\n";

  std::vector<std::string> constraints;
  for (auto&& param: params)
    param->constraint(constraints);
  for (std::string constraint: constraints)
    code += "    " + constraint + ",\n";

  code += "  });\n";
  code += "}\n\n";
  
  return code;
}

std::vector<std::string> empty_strvec() {
  return std::vector<std::string>();
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>>
to_quad(std::vector<std::string> first, std::vector<std::string> second, std::string third, std::vector<std::string> fourth) {
  return
    std::make_tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>>(
      std::move(first), std::move(second), std::move(third), std::move(fourth));
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>> Param::to_code() {
  switch(ptype) {
    case INT:
    case BOOL:
      return to_quad(empty_strvec(),empty_strvec(),"arg[" + std::to_string(offset_start) + "]",empty_strvec());
    case FLOAT:
      return to_quad(empty_strvec(),empty_strvec(),"double_dict[arg[" + std::to_string(offset_start) + "]]",empty_strvec());
    case DTYPE:
      return to_quad(empty_strvec(),empty_strvec(),"get_dtype(arg[" + std::to_string(offset_start) + "])",empty_strvec());
    case TENSOR: {
      std::string var_name = "tensor_" + std::to_string(tensor_id++);
      if (tensor_rank) {
        std::string shape = "{";
        for (size_t i = offset_start; i < offset_start + offset_size; i++) {
          shape += "arg[" + std::to_string(i) + "]";
          if (i != offset_start + offset_size - 1)
            shape += ",";
        }
        shape += "}";
        std::string tensor_guard = "if (is_too_big(" + shape + "))";
        std::string tensor_init = "auto " + var_name + " = torch_tensor(arg[0], " + shape + ");";
        return to_quad({tensor_init},empty_strvec(),var_name,{tensor_guard, "  return -1;"});
      } else {
        std::string shape = "{";
        for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
          shape += "arg[" + std::to_string(i) + "]";
          if (i != offset_start + offset_size - 1)
            shape += ",";
        }
        shape += "}";
        std::string tensor_guard = "if (is_too_big(arg[" + std::to_string(offset_start) + "], " + shape + "))";
        std::string tensor_init = "auto " + var_name + " = torch_tensor(arg[0], arg[" + std::to_string(offset_start) + "], " + shape + ");";
        return to_quad({tensor_init},empty_strvec(),var_name,{tensor_guard, "  return -1;"});
      }
    }
    case INTVECTOR: {
      std::string var_name = "int_vector_" + std::to_string(int_vector_id++);
      std::string vec_init = "std::vector<long> " + var_name + "_ = {";
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
        vec_init += "arg[" + std::to_string(i) + "]";
        if (i != offset_start + offset_size - 1)
          vec_init += ",";
      }
      vec_init += "};";
      std::string vec_size_set = "std::vector<long> " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(offset_start) + "]]);";
      return to_quad({vec_init, vec_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case FLOATVECTOR: {
      std::string var_name = "float_vector_" + std::to_string(float_vector_id++);
      std::string vec_init = "std::vector<double> " + var_name + "_ = {";
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
        vec_init += "double_dict[arg[" + std::to_string(i) + "]]";
        if (i != offset_start + offset_size - 1)
          vec_init += ",";
      }
      vec_init += "};";
      std::string vec_size_set = "std::vector<double> " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(offset_start) + "]]);";
      return to_quad({vec_init, vec_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case INTARRAYREF: {
      std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = "std::vector<long> " + var_name + "_ = {";
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
        array_init += "arg[" + std::to_string(i) + "]";
        if (i != offset_start + offset_size - 1)
          array_init += ",";
      }
      array_init += "};";
      std::string array_size_set = "std::vector<long> " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(offset_start) + "]]);";
      return to_quad({array_init, array_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case EXPANDINGARRAY: {
      std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = "std::vector<long> " + var_name + " = {";
      for (size_t i = offset_start; i < offset_start + offset_size; i++) {
        array_init += "arg[" + std::to_string(i) + "]";
        if (i != offset_start + offset_size - 1)
          array_init += ",";
      }
      array_init += "};";

      return to_quad({array_init},empty_strvec(),var_name,empty_strvec());
    }
    case OPTIONAL: {
      std::vector<std::string> option_check;
      option_check.push_back("if (arg[" + std::to_string(offset_start) + "])");
      auto t = base->to_code();
      std::vector<std::string> preparation = std::get<0>(t);
      std::string expr = std::get<2>(t);
      std::vector<std::string> guard = std::get<3>(t);
      return to_quad(preparation,option_check,expr,guard);
    }
    case VARIANT: {
      if (enums.size() > 0) {
        std::string var_name = "enum_" + std::to_string(enum_id++);
        std::vector<std::string> enum_init;
        enum_init.push_back("typedef");
        enum_init.push_back("  c10::variant<");
        for (size_t i = 0; i < enums.size(); i++) {
          std::string enum_type = "    torch::enumtype::" + enums[i]->enum_name;
          if (i != enums.size()-1)
            enum_type += ",";
          else
            enum_type += ">";
          enum_init.push_back(enum_type);
        }
        enum_init.push_back("  " + var_name + "_t;");
        enum_init.push_back("std::vector<" + var_name + "_t> " + var_name + " = {");
        for (size_t i = 0; i < enums.size(); i++) {
          std::string enum_elem = "  torch::" + enums[i]->enum_name;
          if (i != enums.size()-1)
            enum_elem += ",";
          enum_init.push_back(enum_elem);
        }
        enum_init.push_back("};");
        return to_quad(enum_init, empty_strvec(), var_name + "[arg[" + std::to_string(offset_start) + "]]",empty_strvec());
      } else {
        return expandingarray->to_code();
      }
    }
    case MAP: {
      std::vector<std::string> preparation;
      std::vector<std::string> option_check;
      std::vector<std::string> map_init;
      std::vector<std::string> guard;

      bool separate_init = false;
      if (ctor_params.size() == 1 &&
          ctor_params[0].second->ptype == VARIANT &&
          ctor_params[0].second->enums.size() > 0) {
        map_init.push_back(map_name + " foptions;");
        map_init.push_back("if (arg[" + std::to_string(ctor_params[0].second->offset_start) + "] == 0) {");
        map_init.push_back("  foptions = " + map_name + "(torch::" + ctor_params[0].second->enums[0]->enum_name + ");");
        for (size_t i = 1; i < ctor_params[0].second->enums.size(); i++) {
          map_init.push_back("} else if (arg[" + std::to_string(ctor_params[0].second->offset_start) + "] == " + std::to_string(i) + ") {");
          map_init.push_back("  foptions = " + map_name + "(torch::" + ctor_params[0].second->enums[i]->enum_name + ");");
        }
        map_init.push_back("}");
        if (entries.size() > 0)
          map_init.push_back("foptions");
        separate_init = true;
      } else {
        std::string ctor_param_str;
        for (auto&& p: ctor_params) {
          auto t = p.second->to_code();
          auto preparation_ = std::get<0>(t);
          auto option_check_ = std::get<1>(t);
          auto exp_ = std::get<2>(t);
          auto guard_ = std::get<3>(t);
          assert(option_check.size() == 0);
          if (preparation_.size() > 0) {
            if (preparation.size() > 0)
              preparation.push_back("");
            preparation.insert(preparation.end(), preparation_.begin(), preparation_.end());
          }
          if (ctor_param_str.length() > 0)
            ctor_param_str += ",";
          ctor_param_str += exp_;
          if (guard_.size() > 0) 
            guard.insert(guard.end(), guard_.begin(), guard_.end());
        }
        map_init.push_back("auto foptions =");
        map_init.push_back("  " + map_name + "(" + ctor_param_str + ")");
      }

      for (auto&& p: entries) {
        auto t = p.second->to_code();
        auto preparation_ = std::get<0>(t);
        auto option_check_ = std::get<1>(t);
        auto exp_ = std::get<2>(t);
        auto guard_ = std::get<3>(t);
        if (preparation_.size() > 0) {
          if (preparation.size() > 0)
            preparation.push_back("");
          preparation.insert(preparation.end(), preparation_.begin(), preparation_.end());
        }
        if (option_check_.size() > 0) {
          assert(option_check_.size() == 1);
          option_check.push_back(option_check_[0]);
          option_check.push_back("  foptions." + p.first + "(" + exp_ + ");");
        } else {
          std::string param_set = "." + p.first + "(" + exp_ + ")";
          if (separate_init) {
            if (entries.size() == 1) {
              map_init[map_init.size()-1] = map_init[map_init.size()-1] + param_set;
            } else {
              map_init.push_back("  " + param_set);
            }
          } else {
            map_init.push_back("    " + param_set);
          }
        }
        if (guard_.size() > 0) 
          guard.insert(guard.end(), guard_.begin(), guard_.end());
      }
      if (map_init.size() > 0)
        map_init[map_init.size()-1] = map_init[map_init.size()-1] + ";";
      if (preparation.size() > 0)
        preparation.push_back("");
      preparation.insert(preparation.end(), map_init.begin(), map_init.end());
      preparation.insert(preparation.end(), option_check.begin(), option_check.end());
      return to_quad(preparation,empty_strvec(),"foptions",guard);
    }
    default:
      assert(false);
  }
}

std::string gen_api_call(std::string api_name, std::vector<std::unique_ptr<Param>>& params) {
  std::string code;
  code += "int PathFinderTestOneInput(const long* arg) {\n";
  code += "  torch::set_num_threads(1);\n\n";



  

  std::vector<std::string> preparation;
  std::vector<std::string> positional_arg;
  std::vector<std::string> guard;
  for (auto&& param: params) {
    auto t = param->to_code();
    auto preparation_ = std::get<0>(t);
    auto exp_ = std::get<2>(t);
    auto guard_ = std::get<3>(t);
    if (preparation_.size() > 0) {
      if (preparation.size() > 0)
        preparation.push_back("");
      preparation.insert(preparation.end(), preparation_.begin(), preparation_.end());
    }
    positional_arg.push_back(exp_);
    if (guard_.size() > 0)
      guard.insert(guard.end(), guard_.begin(), guard_.end());
  }

  for (auto line: guard) {
    if (line != "")
      code += "  " + line;
    code += "\n";
  }

  code += "\n  try {\n";
  for (auto line: preparation) {
    if (line != "")
      code += "    " + line;
    code += "\n";
  }

  code += "\n    PathFinderExecuteTarget(\n";
  code += "      auto result = " + api_name + "(";
  for (size_t i = 0; i < positional_arg.size(); i++) {
    code += positional_arg[i];
    if (i != positional_arg.size()-1) {
      code += ", ";
    }
  }
  code += "));\n";
  code += "  } catch (std::exception& e) {\n";
  code += "    return -2;\n";
  code += "  }\n\n";

  code += "  return 0;\n";
  code += "}\n\n";
  
  return code;
}

std::string gen_header() {
  std::string code;

  code += "#include <stdint.h>\n";
  code += "#include <stddef.h>\n";
  code += "#include <c10/util/irange.h>\n";
  code += "#include <cassert>\n";
  code += "#include <cstdlib>\n";
  code += "#include <torch/torch.h>\n";
  code += "#include \"pathfinder.h\"\n";
  code += "#include \"fuzzer_util.h\"\n\n";

  code += "using namespace fuzzer_util;\n\n";

  code += "extern \"C\" {\n\n";

  return code;
}

std::string gen_footer() {
  std::string code;

  code += "}  // extern \"C\"\n\n";

  return code;
}

std::string gen_code(std::string api_name, std::vector<std::unique_ptr<Param>>& params) {
  tensor_id = 0;
  int_vector_id = 0;
  float_vector_id = 0;
  array_id = 0;
  enum_id = 0;

  //size_t offset = 0;
  size_t offset = 1; // offset 0 for dtype
  for (auto&& param: params)
    offset = param->set_offset(offset);

  std::string code;
  code += gen_header();
  code += gen_setup(offset, params);
  code += gen_api_call(api_name, params);
  code += gen_footer();

  return code;

}

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
      if (default_int)
        str += ": default= " + std::to_string(default_int.getValue());
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case BOOL:
      str += str_mult(depth, indent) + "BOOL";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case FLOAT:
      str += str_mult(depth, indent) + "FLOAT";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case ENUM:
      str += str_mult(depth, indent) + "ENUM: " + enum_name;
      str += "\n";
      break;
    case INTVECTOR:
      str += str_mult(depth, indent) + "INTVECTOR";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case FLOATVECTOR:
      str += str_mult(depth, indent) + "FLOATVECTOR";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case TENSOR:
      str += str_mult(depth, indent) + "TENSOR";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case EXPANDINGARRAY:
      str += str_mult(depth, indent) + "EXPANDINGARRAY " + std::to_string(expandingarray_size);
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case OPTIONAL:
      str += str_mult(depth, indent) + "OPTIONAL";
      break;
    case VARIANT: {
      str += str_mult(depth, indent) + "VARIANT";
      if (default_int)
        str += ": default= " + std::to_string(default_int.getValue());
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      for (auto&& t: enums)
        str += t->to_string(depth+1);
      if (expandingarray != nullptr)
        str += expandingarray->to_string(depth+1);
      break;
    }
    case MAP: {
      str += str_mult(depth, indent) + "MAP: " + map_name + "\n";
      for (auto&& entry: entries) {
        str += str_mult(depth+1, indent) + entry.first + "\n";
        str += entry.second->to_string(depth+2);
      }
      break;
    }
    default: assert(false);
  }
  return str;
}

#endif
