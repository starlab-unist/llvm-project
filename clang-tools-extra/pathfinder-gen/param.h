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
  ENUM,
  INTVECTOR,
  FLOATVECTOR,
  TENSOR,
  EXPANDINGARRAY,
  OPTIONAL,
  VARIANT,
  MAP,
};

class Param {
  public:
    Param(ParamType ptype_): ptype(ptype_) {
      assert(ptype == INT || ptype == BOOL || ptype == FLOAT || ptype == TENSOR || ptype == INTVECTOR || ptype == FLOATVECTOR);
      if (ptype == INT || ptype == BOOL || ptype == FLOAT) {
        offset_size = 1;
      } else if (ptype == TENSOR || ptype == INTVECTOR || ptype == FLOATVECTOR) {
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
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::string> to_code();
  private:
    ParamType ptype;

    // INT
    Optional<size_t> default_int = None;
    size_t offset_start;
    size_t offset_size;
    //size_t start_idx = 0;
    //size_t idx_size = 0;

    // BOOL
    //Optional<bool> default_bool = None;

    // FLOAT
    //Optional<double> default_float = None;

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
    //std::vector<std::unique_ptr<Param>> variant_types;
    //bool variant_use_expanding_array = false;
    //bool variant_use_enum = false;
    //size_t variant_num_enums = 0;
    //std::vector<std::string> variant_enum_vec;

    // MAP
    std::string map_name;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries;

    //Optional<unsigned long> default_num = None;
    //Optional<double> default_float = None;
    //Optional<std::string> default_enum = None;
    //bool default_tensor = false;

    friend size_t set_idx(std::vector<std::unique_ptr<Param>>& params);
    friend void gen_pathfinder_fuzz_target(
      std::string target_api_name,
      std::vector<std::unique_ptr<Param>>& params,
      std::ostream& os);
    friend Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, ASTContext &Ctx);
};

size_t Param::set_offset(size_t offset) {
  switch(ptype) {
    case INT:
    case BOOL:
    case FLOAT:
    case INTVECTOR:
    case FLOATVECTOR:
    case TENSOR:
    case EXPANDINGARRAY:
      offset_start = offset;
      offset += offset_size;
      break;
    case OPTIONAL:
      offset_start = offset;
      offset = base->set_offset(offset+1);
      break;
    case VARIANT:
      //offset_start = offset;
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
    //std::cout << "field name: " << field_name << std::endl;
    //field->dump();
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
          //std::cout << "default value: " << std::to_string(val) << std::endl;
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
  code += "  PathFinderAddConstraint({\n";

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

std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
to_triple(std::vector<std::string> first, std::vector<std::string> second, std::string third) {
  return
    std::make_tuple<std::vector<std::string>, std::vector<std::string>, std::string>(
      std::move(first), std::move(second), std::move(third));
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::string> Param::to_code() {
  switch(ptype) {
    case INT:
    case BOOL:
      return to_triple(empty_strvec(),empty_strvec(),"arg[" + std::to_string(offset_start) + "]");
    case FLOAT:
      return to_triple(empty_strvec(),empty_strvec(),"double_dict[arg[" + std::to_string(offset_start) + "]]");
    case INTVECTOR: {
      std::string var_name = "int_vector_" + std::to_string(int_vector_id++);
      std::string vec_init = "std::vector<long> " + var_name + "_ = {";
      for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
        vec_init += "arg[" + std::to_string(i) + "]";
        if (i != offset_start + offset_size - 1)
          vec_init += ",";
      }
      vec_init += "};";
      std::string vec_size_set = "std::vector<long>" + var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(offset_start) + "]]);";
      return to_triple({vec_init, vec_size_set},empty_strvec(),var_name);
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
      std::string vec_size_set = "std::vector<double>" + var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(offset_start) + "]]);";
      return to_triple({vec_init, vec_size_set},empty_strvec(),var_name);
    }
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
        std::string tensor_init = "auto " + var_name + " = torch::randn(" + shape + ", toptions);";
        return to_triple({tensor_init},empty_strvec(),var_name);
      } else {
        std::string shape_init = "std::vector<long> " + var_name + "_shape_ = {";
        for (size_t i = offset_start + 1; i < offset_start + offset_size; i++) {
          shape_init += "arg[" + std::to_string(i) + "]";
          if (i != offset_start + offset_size - 1)
            shape_init += ",";
        }
        shape_init += "};";
        std::string shape_set_rank = "std::vector<long> " + var_name + "_shape(&" + var_name + "_shape_[0],&" + var_name + "_shape_[arg[" + std::to_string(offset_start) + "]]);";
        std::string tensor_init = "auto " + var_name + " = torch::randn(" + var_name + "_shape, toptions);";
        return to_triple({shape_init,shape_set_rank,tensor_init},empty_strvec(),var_name);
      }
    }
    case EXPANDINGARRAY: {
      std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = "std::initializer_list<long> " + var_name + " = {";
      for (size_t i = offset_start; i < offset_start + offset_size; i++) {
        array_init += "arg[" + std::to_string(i) + "]";
        if (i != offset_start + offset_size - 1)
          array_init += ",";
      }
      array_init += "};";

      return to_triple({array_init},empty_strvec(),var_name);
    }
    case OPTIONAL: {
      std::vector<std::string> option_check;
      option_check.push_back("if (arg[" + std::to_string(offset_start) + "])");
      auto t = base->to_code();
      std::vector<std::string> preparation = std::get<0>(t);
      std::string expr = std::get<2>(t);
      return to_triple(preparation,option_check,expr);
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
        return to_triple(enum_init, empty_strvec(), var_name + "[arg[" + std::to_string(offset_start) + "]]");
      } else {
        std::string exp = "{";
        for (size_t i = expandingarray->offset_start; i < expandingarray->offset_start + expandingarray->offset_size; i++) {
          exp += "arg[" + std::to_string(i) + "]";
          if (i != expandingarray->offset_start + expandingarray->offset_size - 1)
            exp += ",";
        }
        exp += "}";
        return to_triple(empty_strvec(),empty_strvec(),exp);
      }
    }
    case MAP: {
      std::vector<std::string> preparation;
      std::vector<std::string> option_check;
      std::vector<std::string> map_init;

      std::string ctor_param_str;
      for (auto&& p: ctor_params) {
        auto t = p.second->to_code();
        auto preparation_ = std::get<0>(t);
        auto option_check_ = std::get<1>(t);
        auto exp_ = std::get<2>(t);
        assert(option_check.size() == 0);
        if (preparation_.size() > 0) {
          if (preparation.size() > 0)
            preparation.push_back("");
          preparation.insert(preparation.end(), preparation_.begin(), preparation_.end());
        }
        if (ctor_param_str.length() > 0)
          ctor_param_str += ",";
        ctor_param_str += exp_;
      }

      map_init.push_back("auto foptions =");
      map_init.push_back("  " + map_name + "(" + ctor_param_str + ")");

      //std::string map_name;
      for (auto&& p: entries) {
        auto t = p.second->to_code();
        auto preparation_ = std::get<0>(t);
        auto option_check_ = std::get<1>(t);
        auto exp_ = std::get<2>(t);
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
          map_init.push_back("    ." + p.first + "(" + exp_ + ")");
        }
      }
      if (map_init.size() > 0)
        map_init[map_init.size()-1] = map_init[map_init.size()-1] + ";";
      if (preparation.size() > 0)
        preparation.push_back("");
      preparation.insert(preparation.end(), map_init.begin(), map_init.end());
      preparation.insert(preparation.end(), option_check.begin(), option_check.end());
      return to_triple(preparation,empty_strvec(),"foptions");
    }
    default:
      assert(false);
  }
}

std::string gen_api_call(std::string api_name, std::vector<std::unique_ptr<Param>>& params) {
  std::string code;
  code += "int PathFinderTestOneInput(const long* arg) {\n";
  code += "  torch::set_num_threads(1);\n\n";
  code += "  try {\n";

  std::vector<std::string> preparation;
  preparation.push_back("torch::TensorOptions toptions = torch::TensorOptions();");
  std::vector<std::string> positional_arg;
  for (auto&& param: params) {
    auto t = param->to_code();
    auto preparation_ = std::get<0>(t);
    auto exp_ = std::get<2>(t);
    if (preparation_.size() > 0) {
      preparation.push_back("");
      preparation.insert(preparation.end(), preparation_.begin(), preparation_.end());
    }
    positional_arg.push_back(exp_);
  }
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
  code += "  } catch (::c10::Error& e) {\n";
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

  size_t offset = 0;
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
      //if (default_bool)
      //  str += ": default= " + std::to_string(default_num.getValue());
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case FLOAT:
      str += str_mult(depth, indent) + "FLOAT";
      //if (default_num)
      //  str += ": default= " + std::to_string(default_float.getValue());
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case ENUM:
      str += str_mult(depth, indent) + "ENUM: " + enum_name;
      //if (default_enum)
      //  str += ": default= " + default_enum.getValue();
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
      //if (default_tensor)
      //  str += ": default=Tensor()";
      str += "  // start idx: " + std::to_string(offset_start) + ", idx size: " + std::to_string(offset_size);
      str += "\n";
      break;
    case EXPANDINGARRAY:
      str += str_mult(depth, indent) + "EXPANDINGARRAY " + std::to_string(expandingarray_size);
      //if (default_num)
      //  str += ": default= " + std::to_string(default_num.getValue());
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

/* void Param::set_default(std::string param_name, const CXXRecordDecl* cdecl) {
  //std::cout << "param name: " << param_name << std::endl;
  //cdecl->dump();
  for (auto field: cdecl->fields()) {
    std::string field_name = field->getNameAsString();
    //std::cout << "field name: " << field_name << std::endl;
    if (param_name.length() + 1 == field_name.length() &&
        field_name.compare(0, field_name.length(), param_name + "_") == 0) {
      //field->dump();
      if (const auto* e = field->getInClassInitializer()->IgnoreUnlessSpelledInSource()){
        //e->dump();
        switch (ptype) {
          case INTVECTOR:
          case FLOATVECTOR:
            break;
          case TENSOR:
            default_tensor = true;
            break;
          case ENUM:
            assert(false); // TODO;
            break;
          case BOOL: {
            auto b = dyn_cast<CXXBoolLiteralExpr>(e);
            assert(b != nullptr);
            break;
          }
          case FLOAT: {
            if (const auto* fl = dyn_cast<FloatingLiteral>(e)) {
              double val = fl->getValue().convertToDouble();
              //std::cout << "default value: " << std::to_string(val) << std::endl;
              default_float = val;
            }
            break;
          }
          default: {
            if (const auto* il = dyn_cast<IntegerLiteral>(e)) {
              unsigned long val = il->getValue().getZExtValue();
              //std::cout << "default value: " << std::to_string(val) << std::endl;
              default_num = val;
            }
            break;
          }
        }
      }
    }
  }
} */

/* size_t set_idx(std::vector<std::unique_ptr<Param>>& params) {
  //size_t idx = 2;
  size_t idx = 0;
  for (auto&& param: params) {
    switch (param->ptype) {
      case INT:
      case BOOL:
        param->start_idx = idx;
        param->idx_size = 1;
        idx += param->idx_size;
        break;
      case TENSOR:
        param->start_idx = idx;
        if (param->tensor_rank) {
          param->idx_size = param->tensor_rank.getValue();
        } else {
          param->idx_size = MAX_RANK + 1;
        }
        idx += param->idx_size;
        break;
      case API_OPTION: {
        for (auto&& entry: param->api_option_types) {
          switch (entry.second->ptype) {
            case INT:
            case BOOL:
            case FLOAT:
              entry.second->start_idx = idx;
              entry.second->idx_size = 1;
              idx += entry.second->idx_size;
              break;
            case INTVECTOR:
            case FLOATVECTOR:
              entry.second->start_idx = idx;
              entry.second->idx_size = MAX_VECTOR_SIZE + 1;
              idx += entry.second->idx_size;
              break;
            case TENSOR:
              entry.second->start_idx = idx;
              if (entry.second->tensor_rank) {
                entry.second->idx_size = entry.second->tensor_rank.getValue()+1;
                //std::cout << "idx size is: " << std::to_string(entry.second->idx_size) << std::endl;
              } else {
                entry.second->idx_size = MAX_RANK + 2;
              }
              idx += entry.second->idx_size;
              break;
            case EXPANDINGARRAY:
              entry.second->start_idx = idx;
              entry.second->idx_size = entry.second->expandingarray_size;
              idx += entry.second->idx_size;
              break;
            case OPTIONAL: {
              //entry.second->start_idx = idx;
              switch (entry.second->base->ptype) {
                case FLOAT:
                  entry.second->start_idx = idx;
                  entry.second->idx_size = 2;
                  break;
                default:
                  assert(false);
              }
              break;
            }
            case VARIANT: {
              size_t expandingarray_size_ = 0;
              size_t num_enums = 0;
              //entry.second->to_string();
              for (auto&& param2: entry.second->variant_types) {
                switch (param2->ptype) {
                  case EXPANDINGARRAY:
                    expandingarray_size_ = param2->expandingarray_size;
                    break;
                  case ENUM:
                    num_enums++;
                    //std::cout << "num_enum: " << std::to_string(num_enums) << std::endl;
                    break;
                  default:
                    assert(false);
                }
              }
              entry.second->start_idx = idx;
              //std::cout << " expanding array size: " << std::to_string(expandingarray_size_) << std::endl;
              //std::cout << " num enums: " << std::to_string(num_enums) << std::endl;
              if (expandingarray_size_ > 0) {
                entry.second->variant_use_expanding_array = true;
                entry.second->idx_size = expandingarray_size_;
                idx += entry.second->idx_size;
              } else if (num_enums > 0) {
                entry.second->variant_use_enum = true;
                entry.second->idx_size = 1;
                entry.second->variant_num_enums = num_enums;
                idx += entry.second->idx_size;
                //std::cout << entry.second->to_string() << std::endl;
              } else {
                assert(false);
              }
              break;
            }
            default:
              std::cout << entry.second->to_string() << std::endl;
              assert(false);
          }
        }
        break;
      }
      default:
        assert(false);
    }
  }
  return idx;
} */

/* void gen_pathfinder_fuzz_target(
  std::string target_api_name,
  std::vector<std::unique_ptr<Param>>& params,
  std::ostream& os)
{
  //const size_t NUM_DEVICE = 2;
  //const size_t NUM_DTYPE = 11;

  size_t num_args = set_idx(params);

  os << "#include <stdint.h>\n";
  os << "#include <stddef.h>\n";
  os << "#include <c10/util/irange.h>\n";
  os << "#include <cassert>\n";
  os << "#include <cstdlib>\n";
  os << "#include <torch/torch.h>\n";
  os << "#include \"pathfinder.h\"\n";
  os << "#include \"fuzzer_util.h\"\n\n";

  os << "extern \"C\" {\n\n";

  os << "void PathFinderSetup() {\n";
  os << "  PathFinderSetArgSize(" << std::to_string(num_args) << ");\n";
  os << "  PathFinderAddConstraint({\n";
  //os << "    0 <= arg[0], arg[0] < " << std::to_string(NUM_DEVICE) << ",    // device\n";
  //os << "    0 <= arg[1], arg[1] < " << std::to_string(NUM_DTYPE) << ",    // dtype\n";

  for (auto&& param: params) {
    switch (param->ptype) {
      case INT: {
        int min = param->default_num ? param->default_num.getValue() : 0;
        os << "    arg[" << std::to_string(param->start_idx) << "] >= " << std::to_string(min) << ",\n";
        break;
      }
      case BOOL: {
        std::string arg = "arg[" + std::to_string(param->start_idx) + "]";
        os << "    0 <= " << arg << ", " << arg << " <= 1,\n";
        break;
      }
      case FLOAT: {
        std::string arg = "arg[" + std::to_string(param->start_idx) + "]";
        os << "    0 <= " << arg << ", " << arg << " < " << std::to_string(DOUBLE_DICT_SIZE) << ",\n";
        break;
      }
      case TENSOR: {
        if (param->tensor_rank) {
          for (size_t i = param->start_idx; i < param->start_idx + param->idx_size; i++) {
            os << "    arg[" << std::to_string(i) << "] >= 1,\n";
          }
        } else {
          std::string rank = "arg[" + std::to_string(param->start_idx) + "]";
          os << "    0 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_RANK) << ",\n";
          for (size_t i = param->start_idx + 1; i < param->start_idx + param->idx_size; i++) {
            os << "    arg[" << std::to_string(i) << "] >= 1,\n";
          }
        }
        break;
      }
      case API_OPTION: {
        for (auto&& entry: param->api_option_types) {
          switch (entry.second->ptype) {
            case INT: {
              int min = entry.second->default_num ? entry.second->default_num.getValue() : 0;
              os << "    arg[" << std::to_string(entry.second->start_idx) << "] >= " << std::to_string(min) << ",\n";
              break;
            }
            case BOOL: {
              std::string arg = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << arg << ", " << arg << " <= 1,\n";
              break;
            }
            case FLOAT: {
              std::string arg = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << arg << ", " << arg << " < " << std::to_string(DOUBLE_DICT_SIZE) << ",\n";
              break;
            }
            case INTVECTOR:{
              std::string rank = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_VECTOR_SIZE) << ",\n";
              for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                os << "    arg[" << std::to_string(i) << "] >= 0,\n";
              }
              break;
            }
            case FLOATVECTOR:{
              std::string rank = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_VECTOR_SIZE) << ",\n";
              for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                os << "    0 <= arg[" << std::to_string(i) << "], arg[" << std::to_string(i) << "] < " << std::to_string(DOUBLE_DICT_SIZE) << ",\n";
              }
              break;
            }
            case TENSOR:{
              std::string is_some = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << is_some << ", " << is_some << " <= 1,\n";
              if (entry.second->tensor_rank) {
                //std::cout << "TENSOR RANK iS: " << std::to_string(entry.second->tensor_rank.getValue()) << std::endl;
                //std::cout << "entry.second->start_idx: " << std::to_string(entry.second->start_idx) << std::endl;
                //std::cout << "entry.second->idx_size: " << std::to_string(entry.second->idx_size) << std::endl;
                for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  os << "    arg[" << std::to_string(i) << "] >= 1,\n";
                }
              } else {
                std::string rank = "arg[" + std::to_string(entry.second->start_idx+1) + "]";
                os << "    1 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_RANK) << ",\n";
                for (size_t i = entry.second->start_idx + 2; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  os << "    arg[" << std::to_string(i) << "] >= 1,\n";
                }
              }
              break;
            }
            case EXPANDINGARRAY: {
              for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                int min = entry.second->default_num ? entry.second->default_num.getValue() : 0;
                os << "    arg[" << std::to_string(i) << "] >= " << std::to_string(min) << ",\n";
              }
              break;
            }
            case OPTIONAL: {
              switch (entry.second->base->ptype) {
                case FLOAT: {
                  std::string is_some = "arg[" + std::to_string(entry.second->start_idx) + "]";
                  std::string arg = "arg[" + std::to_string(entry.second->start_idx+1) + "]";
                  os << "    0 <= " << is_some << ", " << is_some << " <= 1,\n";
                  os << "    0 <= " << arg << ", " << arg << " < " << std::to_string(DOUBLE_DICT_SIZE) << ",\n";
                  break;
                }
                default:
                  assert(false);
              }
              break;
            }
            case VARIANT: {
              if (entry.second->variant_use_expanding_array) {
                for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  int min = entry.second->default_num ? entry.second->default_num.getValue() : 0;
                  os << "    arg[" << std::to_string(i) << "] >= " << std::to_string(min) << ",\n";
                }
              } else if (entry.second->variant_use_enum) {
                //std::cout << entry.second->to_string() << std::endl;
                std::string arg = "arg[" + std::to_string(entry.second->start_idx) + "]";
                os << "    0 <= " << arg << ", " << arg << " < " << std::to_string(entry.second->variant_num_enums) << ",\n";
              }
              break;
            }
            default:
              assert(false);
          }
        }
        break;
      }
      default:
        assert(false);
    }
  }
  os << "  });\n";
  os << "}\n\n";

  os << "int PathFinderTestOneInput(const long* arg) {\n";
  os << "  torch::set_num_threads(1);\n\n";

  os << "  try {\n";
  os << "    torch::TensorOptions toptions = torch::TensorOptions();\n\n";
  //os << "    torch::TensorOptions toptions =\n";
  //os << "      torch::TensorOptions()\n";
  //os << "        .device(fuzzer_util::get_device(arg[0]))\n";
  //os << "        .dtype(fuzzer_util::get_dtype(arg[1]))\n";
  //os << "        .requires_grad(true);\n\n";

  std::vector<std::string> function_args;
  std::vector<std::tuple<Param*, std::string, std::string>> optional_tensor;
  //size_t tensor_id = 0;
  //size_t int_vector_id = 0;
  std::vector<std::tuple<Param*, std::string, std::string>> optional_arg;
  for (auto&& param: params) {
    switch (param->ptype) {
      case INT:
      case BOOL:
      case FLOAT:
        function_args.push_back("arg[" + std::to_string(param->start_idx) + "]");
        break;
      case TENSOR: {
        std::string var_name = "tensor_" + std::to_string(tensor_id++);
        if (param->tensor_rank) {
          //std::string shape = "std::vector<long> " + var_name + "_shape = {";
          std::string shape = "{";
          for (size_t i = param->start_idx; i < param->start_idx + param->idx_size; i++) {
            shape += "arg[" + std::to_string(i) + "]";
            if (i != param->start_idx + param->idx_size - 1)
              shape += ",";
          }
          shape += "}";
          //os << "    " << shape;
          os << "    auto " << var_name << " = torch::randn(" << shape + ", toptions);\n\n";
        } else {
          std::string shape = "std::vector<long> " + var_name + "_shape_ = {";
          for (size_t i = param->start_idx + 1; i < param->start_idx + param->idx_size; i++) {
            shape += "arg[" + std::to_string(i) + "]";
            if (i != param->start_idx + param->idx_size - 1)
              shape += ",";
          }
          shape += "};\n";
          os << "    " << shape;
          os << "    std::vector<long> " << var_name + "_shape(&" + var_name + "_shape_[0],&" + var_name + "_shape_[arg[" + std::to_string(param->start_idx) + "]]);\n";
          os << "    auto " << var_name << " = torch::randn(" << var_name + "_shape, toptions);\n\n";
        }
        function_args.push_back(var_name);
        break;
      }
      case API_OPTION: {
        std::vector<std::pair<std::string,std::string>> opt;
        for (auto&& entry: param->api_option_types) {
          std::string option_name = entry.first;
          std::string option_arg;
          switch (entry.second->ptype) {
            case INT:
            case BOOL:
              option_arg = "arg[" + std::to_string(entry.second->start_idx) + "]";
              break;
            case FLOAT:
              option_arg = "double_dict[arg[" + std::to_string(entry.second->start_idx) + "]]";
              break;
            case INTVECTOR:{
              std::string var_name = "int_vector_" + std::to_string(int_vector_id++);
              std::string vec = "std::vector<long> " + var_name + "_ = {";
              for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                vec += "arg[" + std::to_string(i) + "]";
                if (i != entry.second->start_idx + entry.second->idx_size - 1)
                  vec += ",";
              }
              vec += "};\n";
              os << "    " << vec;
              os << "    std::vector<long> " << var_name + "(&" + var_name + "_[0],&" + var_name + "_[arg[" + std::to_string(entry.second->start_idx) + "]]);\n\n";
              option_arg = var_name;
              break;
            }
            case TENSOR:{
              std::string var_name = "tensor_" + std::to_string(tensor_id++);
              if (entry.second->tensor_rank) {
                //std::string shape = "std::vector<long> " + var_name + "_shape = {";
                std::string shape = "{";
                for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  shape += "arg[" + std::to_string(i) + "]";
                  if (i != entry.second->start_idx + entry.second->idx_size - 1)
                    shape += ",";
                }
                shape += "}";
                //os << "    " << shape;
                os << "    auto " << var_name << " = torch::randn(" << shape + ", toptions);\n\n";
              } else {
                std::string shape = "std::vector<long> " + var_name + "_shape_ = {";
                for (size_t i = entry.second->start_idx + 2; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  shape += "arg[" + std::to_string(i) + "]";
                  if (i != entry.second->start_idx + entry.second->idx_size - 1)
                    shape += ",";
                }
                shape += "};\n";
                os << "    " << shape;
                os << "    std::vector<long> " << var_name + "_shape(&" + var_name + "_shape_[0],&" + var_name + "_shape_[arg[" + std::to_string(entry.second->start_idx+1) + "]]);\n";
                os << "    auto " << var_name << " = torch::randn(" << var_name + "_shape, toptions);\n\n";
              }
              optional_tensor.push_back(std::make_tuple(entry.second.get(),option_name,var_name));
              break;
            }
            case EXPANDINGARRAY: {
              option_arg = "{";
              for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                option_arg += "arg[" + std::to_string(i) + "]";
                if (i != entry.second->start_idx + entry.second->idx_size - 1)
                  option_arg += ",";
              }
              option_arg += "}";
              break;
            }
            case OPTIONAL: {
              switch (entry.second->base->ptype) {
                case FLOAT: {
                  optional_arg.push_back(std::make_tuple(entry.second->base.get(),option_name,"double_dict[arg[" + std::to_string(entry.second->base->start_idx+1) + "]]"));
                  break;
                }
                default:
                  assert(false);
              }
              break;
            }
            case VARIANT: {
              if (entry.second->variant_use_expanding_array) {
                option_arg = "{";
                for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  option_arg += "arg[" + std::to_string(i) + "]";
                  if (i != entry.second->start_idx + entry.second->idx_size - 1)
                    option_arg += ",";
                }
                option_arg += "}";
              } else if (entry.second->variant_use_enum) {
                os << "    std::vector<std::string> enums = {\n";
                for (size_t i = 0; i < entry.second->variant_enum_vec.size(); i++) {
                  os << "      torch::" + entry.second->variant_enum_vec[i];
                  if (i != entry.second->variant_enum_vec.size()-1)
                    os << ",\n";
                  else
                    os << "};\n\n";
                }
                option_arg = "enums[arg[" + std::to_string(entry.second->start_idx) + "]]";
              }
              break;
            }
            default:
              assert(false);
          }
          if (option_arg != "")
            opt.push_back({option_name,option_arg});
        }
        os << "    auto foptions =\n";
        os << "      " << param->api_option_name << "()\n";
        for (size_t i = 0; i < opt.size(); i++) {
          os << "        ." << opt[i].first << "(" << opt[i].second << ")";
          if (i != opt.size()-1)
            os << "\n";
          else
            os << ";\n";
        }
        function_args.push_back("foptions");
        for (auto t: optional_tensor) {
          os << "    if (arg[" << std::to_string(std::get<0>(t)->start_idx) << "])\n";
          os << "      foptions." << std::get<1>(t) << "(" << std::get<2>(t) << ");\n";
        }
        for (auto a: optional_arg) {
          os << "    if (arg[" << std::to_string(std::get<0>(a)->start_idx) << "])\n";
          os << "      foptions." << std::get<1>(a) << "(" << std::get<2>(a) << ");\n";
        }
        os << "\n";
        break;
      }
      default:
        assert(false);
    }
  }

  os << "    PathFinderExecuteTarget(\n";
  os << "      auto result = " << target_api_name << "(";
  for (size_t i = 0; i < function_args.size(); i++) {
    os << function_args[i];
    if (i != function_args.size()-1) {
      os << ", ";
    }
  }
  os << "));\n";
  os << "  } catch (::c10::Error& e) {\n";
  os << "    return -2;\n";
  os << "  }\n\n";

  os << "  return 0;\n";
  os << "}\n\n";

  os << "}  // extern \"C\"\n\n";

} */

#endif
