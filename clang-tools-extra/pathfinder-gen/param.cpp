#include "param.h"
#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace clang;

/* const size_t MAX_RANK = 5;
const size_t MAX_VECTOR_SIZE = 6;
const size_t DOUBLE_DICT_SIZE = 20; */

std::string var_ctr = "sym_int_arg";
std::string var_caller = "x";
size_t tensor_id;
size_t int_vector_id;
size_t float_vector_id;
size_t array_id;
size_t enum_id;

Param::Param(ParamType ptype_, std::string name_): ptype(ptype_), name(name_) {
  assert(ptype == INT || ptype == BOOL || ptype == FLOAT || ptype == DTYPE || ptype == ENUM ||
          ptype == TENSOR || ptype == INTVECTOR || ptype == FLOATVECTOR || ptype == INTARRAYREF);
  if (ptype == BOOL || ptype == DTYPE || ptype == FLOAT) {
    enum_offset_size = 1;
  } else if (ptype == INTVECTOR || ptype == INTARRAYREF) {
    enum_offset_size = 1;
    int_offset_size = MAX_VECTOR_SIZE;
    name_size = name + "_size";
    for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
      name_elem.push_back(name + std::to_string(i));
  } else if (ptype == FLOATVECTOR) {
    enum_offset_size = 1 + MAX_VECTOR_SIZE;
    name_size = name + "_size";
    for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
      name_elem.push_back(name + std::to_string(i));
  } else if (ptype == TENSOR) {
    enum_offset_size = 2; // dtype, rank
    int_offset_size = MAX_RANK;
    name_dtype = name + "_dtype";
    name_rank = name + "_rank";
    for (size_t i = 0; i < MAX_RANK; i++)
      name_dim.push_back(name + std::to_string(i));
  } else { // INT
    int_offset_size = 1;
  }
}
Param::Param(ParamType ptype_, std::string name_, long size)
  : ptype(ptype_), name(name_)
{
  assert(ptype == EXPANDINGARRAY || ptype == EXPANDINGARRAYWITHOPTIONALELEM);
  array_size = size;
  int_offset_size = array_size;
  for (size_t i = 0; i < array_size; i++)
    name_elem.push_back(name + std::to_string(i));
}
Param::Param(ParamType ptype_, std::string name_, std::unique_ptr<Param> base_)
  : ptype(ptype_), name(name_), base(std::move(base_))
{
  assert(ptype == OPTIONAL);
  enum_offset_size = base->enum_offset_size + 1;
  int_offset_size = base->int_offset_size;
}
Param::Param(ParamType ptype_, std::string name_, std::vector<std::unique_ptr<Param>> enums_, std::unique_ptr<Param> expandingarray_)
  : ptype(ptype_), name(name_), enums(std::move(enums_)), expandingarray(std::move(expandingarray_))
{
  assert(ptype == VARIANT);
  assert(enums.size() > 0);
  enum_offset_size = 1;
  if (expandingarray != nullptr)
    int_offset_size = expandingarray->int_offset_size;
}
Param::Param(
  ParamType ptype_,
  std::string name_,
  std::string map_name_,
  std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params_,
  std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries_)
  : ptype(ptype_), name(name_), map_name(map_name_), ctor_params(std::move(ctor_params_)), entries(std::move(entries_))
{
  assert(ptype == MAP);
  enum_offset_size = 0;
  int_offset_size = 0;
  for (auto&& p: ctor_params) {
    enum_offset_size += p.second->enum_offset_size;
    int_offset_size += p.second->int_offset_size;
  }
  for (auto&& p: entries) {
    enum_offset_size += p.second->enum_offset_size;
    int_offset_size += p.second->int_offset_size;
  }
}
bool Param::is_map() { return ptype == MAP; }

std::pair<size_t, size_t> Param::set_offset(size_t enum_offset, size_t int_offset) {
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
    case EXPANDINGARRAYWITHOPTIONALELEM:
      enum_offset_start = enum_offset;
      enum_offset += enum_offset_size;
      int_offset_start = int_offset;
      int_offset += int_offset_size;
      break;
    case OPTIONAL:
      enum_offset_start = enum_offset;
      std::tie(enum_offset, int_offset) = base->set_offset(enum_offset + 1, int_offset);
      break;
    case VARIANT:
      enum_offset_start = enum_offset;
      if (expandingarray != nullptr)
        std::tie(enum_offset, int_offset) = expandingarray->set_offset(enum_offset + 1, int_offset);
      else
        enum_offset += enum_offset_size;
      break;
    case MAP:
      for (auto&& p: ctor_params)
        std::tie(enum_offset, int_offset) = p.second->set_offset(enum_offset, int_offset);
      for (auto&& p: entries)
        std::tie(enum_offset, int_offset) = p.second->set_offset(enum_offset, int_offset);
      break;
    default:
      assert(false);
  }
  return std::make_pair(enum_offset, int_offset);
}

void Param::set_default(std::string param_name, const CXXRecordDecl* cdecl) {
  Expr* e = nullptr;
  for (auto field: cdecl->fields()) {
    std::string field_name = field->getNameAsString();
    if (param_name.length() + 1 == field_name.length() &&
        field_name.compare(0, field_name.length(), param_name + "_") == 0) {
      //std::cout << "filed: \n";
      //field->dump();
      e = field->getInClassInitializer()->IgnoreUnlessSpelledInSource();
      //e->dump();
    }
  }

  if (e !=nullptr) {
    switch(ptype) {
      case INT:
      case EXPANDINGARRAY:
      case EXPANDINGARRAYWITHOPTIONALELEM: {
        //std::cout << "aaa\n";
        if (const auto* il = dyn_cast<IntegerLiteral>(e)) {
          assert(il != nullptr);
          //std::cout << "bbb\n";
          unsigned long val = il->getValue().getZExtValue();
          default_int = val;
          //std::cout << "default value: " << std::to_string(val) << std::endl;
        }
        break;
      }
      case VARIANT: {
        if (const auto* il = dyn_cast<IntegerLiteral>(e)) {
          //std::cout << "bbb\n";
          unsigned long val = il->getValue().getZExtValue();
          if(expandingarray != nullptr)
            expandingarray->default_int = val;
        }
        break;
      }
      default:
        break;
    }
  }
}

void Param::setup_arg(std::vector<std::string>& args) {
  switch(ptype) {
    case INT: {
      args.push_back("PathFinderIntArg(\"" + name + "\");");
      break;
    }
    case BOOL: {
      args.push_back("PathFinderEnumArg(\"" + name + "\", {\"true\", \"false\"});");
      break;
    }
    case FLOAT: {
      args.push_back("PathFinderEnumArg(\"" + name + "\", double_dict_str);");
      break;
    }
    case DTYPE: {
      args.push_back("PathFinderEnumArg(\"" + name + "\", dtype_str);");
      break;
    }
    case TENSOR: {
      args.push_back("PathFinderEnumArg(\"" + name_dtype + "\", dtype_str);");
      args.push_back("PathFinderEnumArg(\"" + name_rank + "\", " + std::to_string(MAX_RANK + 1) +");");
      for (size_t i = 0; i < MAX_RANK; i++)
        args.push_back("PathFinderIntArg(\"" + name_dim[i] + "\");");
      break;
    }
    case INTVECTOR: {
      args.push_back("PathFinderEnumArg(\"" + name_size + "\", " + std::to_string(MAX_VECTOR_SIZE + 1) +");");
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        args.push_back("PathFinderIntArg(\"" + name_elem[i] + "\");");
      break;
    }
    case FLOATVECTOR: {
      args.push_back("PathFinderEnumArg(\"" + name_size + "\", " + std::to_string(MAX_VECTOR_SIZE + 1) +");");
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        args.push_back("PathFinderEnumArg(\"" + name_elem[i] + "\", double_dict_str);");
      break;
    }
    case INTARRAYREF: {
      args.push_back("PathFinderEnumArg(\"" + name_size + "\", " + std::to_string(MAX_VECTOR_SIZE + 1) +");");
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        args.push_back("PathFinderIntArg(\"" + name_elem[i] + "\");");
      break;
    }
    case EXPANDINGARRAY: {
      for (size_t i = 0; i < int_offset_size; i++)
        args.push_back("PathFinderIntArg(\"" + name_elem[i] + "\");");
      break;
    }
    case EXPANDINGARRAYWITHOPTIONALELEM: {
      for (size_t i = 0; i < int_offset_size; i++)
        args.push_back("PathFinderIntArg(\"" + name_elem[i] + "\");");
      break;
    }
    case OPTIONAL: {
      args.push_back("PathFinderEnumArg(\"" + name + "\", {\"some\", \"none\"});");
      base->setup_arg(args);
      break;
    }
    case VARIANT: {
      std::string str = "PathFinderEnumArg(\"" + name + "\", {";
      if (expandingarray != nullptr)
        str += "\"" + expandingarray->name + "\", ";
      for (size_t i = 0; i < enums.size(); i++) {
        str += "\"" + enums[i]->name + "\"";
        if (i != enums.size() - 1)
          str += ", ";
      }
      str += "});";
      args.push_back(str);
      if (expandingarray != nullptr)
        expandingarray->setup_arg(args);
      break;
    }
    case MAP: {
      for (auto&& p: ctor_params)
        p.second->setup_arg(args);
      for (auto&& p: entries)
        p.second->setup_arg(args);
      break;
    }
    default:
      assert(false);
  }
}

void Param::constraint(std::vector<std::string>& hard_ctrs, std::vector<std::string>& soft_ctrs, bool is_module) {
  std::string sep = ", ";
  std::string arg = var_ctr + "[\"" + name + "\"]";
  switch(ptype) {
    case INT: {
      int min =
        default_int ?
        default_int.getValue() :
        (is_module ? 1 : 0);
      soft_ctrs.push_back(arg + " >= " + std::to_string(min));
      break;
    }
    case TENSOR: {
      for (size_t i = 0; i < int_offset_size; i++)
        hard_ctrs.push_back(var_ctr + "[\"" + name_dim[i] + "\"] >= 1");
      break;
    }
    case INTVECTOR: {
      for (size_t i = 0; i < int_offset_size; i++)
        soft_ctrs.push_back(var_ctr + "[\"" + name_elem[i] + "\"] >= 0");
      break;
    }
    case INTARRAYREF: {
      for (size_t i = 0; i < int_offset_size; i++)
        soft_ctrs.push_back(var_ctr + "[\"" + name_elem[i] + "\"] >= 0");
      break;
    }
    case EXPANDINGARRAY: {
      int min =
        default_int ?
        default_int.getValue() :
        (is_module ? 1 : 0);
      for (size_t i = 0; i < int_offset_size; i++)
        soft_ctrs.push_back(var_ctr + "[\"" + name_elem[i] + "\"] >= " + std::to_string(min));
      break;
    }
    case EXPANDINGARRAYWITHOPTIONALELEM: {
      int min = default_int ? default_int.getValue() : 0;
      for (size_t i = 0; i < int_offset_size; i++)
        soft_ctrs.push_back(var_ctr + "[\"" + name_elem[i] + "\"] >= " + std::to_string(min));
      break;
    }
    case OPTIONAL: {
      base->constraint(hard_ctrs, soft_ctrs, is_module);
      break;
    }
    case VARIANT: {
      if (expandingarray != nullptr)
        expandingarray->constraint(hard_ctrs, soft_ctrs, is_module);
      break;
    }
    case MAP: {
      for (auto&& p: ctor_params)
        p.second->constraint(hard_ctrs, soft_ctrs, is_module);
      for (auto&& p: entries)
        p.second->constraint(hard_ctrs, soft_ctrs, is_module);
      break;
    }
    default:
      break;
  }
}

std::string gen_setup(std::vector<std::unique_ptr<Param>>& params, bool is_module) {
  std::vector<std::string> args;
  std::vector<std::string> hard_ctrs;
  std::vector<std::string> soft_ctrs;
  for (auto&& param: params) {
    param->setup_arg(args);
    param->constraint(hard_ctrs, soft_ctrs, is_module);
  }

  std::string code;
  code += "void PathFinderSetup() {\n";
  for (auto arg: args)
    code += "  " + arg + "\n";
  if (hard_ctrs.size() > 0) {
    code += "  PathFinderAddHardConstraint({\n";
    for (auto hard_ctr: hard_ctrs)
      code += "    " + hard_ctr + ",\n";
    code += "  });\n";
  }
  if (soft_ctrs.size() > 0) {
    code += "  PathFinderAddSoftConstraint({\n";
    for (auto soft_ctr: soft_ctrs)
      code += "    " + soft_ctr + ",\n";
    code += "  });\n";
  }
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

std::string Param::type_str() {
  switch(ptype) {
    case ENUM: return "torch::enumtype::" + name;
    case TENSOR: return "auto";
    case INTVECTOR:
    case INTARRAYREF: return "std::vector<long>";
    case FLOATVECTOR: return "std::vector<double>";
    case EXPANDINGARRAY: return "torch::ExpandingArray<" + std::to_string(array_size) + ">";
    case EXPANDINGARRAYWITHOPTIONALELEM:
      return "torch::ExpandingArrayWithOptionalElem<" + std::to_string(array_size) + ">";
    default:
      assert(false);
  }
}

std::string Param::array_str() {
  switch(ptype) {
    case EXPANDINGARRAY: {
      std::string array_init = type_str() + "({";
      for (size_t i = 0; i < int_offset_size; i++) {
        array_init += var_caller + "[\"" + name_elem[i] + "\"]";
        if (i != int_offset_size - 1)
          array_init += ",";
      }
      array_init += "})";
      return array_init;
    }
    case EXPANDINGARRAYWITHOPTIONALELEM: {
      std::string array_init =
        "expandingarray_with_optional_elem<" + std::to_string(array_size) + ">({";
      for (size_t i = 0; i < int_offset_size; i++) {
        array_init += var_caller + "[\"" + name_elem[i] + "\"]";
        if (i != int_offset_size - 1)
          array_init += ",";
      }
      array_init += "})";
      return array_init;
    }
    default:
      assert(false);
  }
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>> Param::to_code(std::string api_name, bool is_module) {
  std::string arg = var_caller + "[\"" + name + "\"]";
  std::string var_name = name;
  switch(ptype) {
    case INT:
    case BOOL:
      return to_quad(empty_strvec(),empty_strvec(), arg, empty_strvec());
    case FLOAT:
      return to_quad(empty_strvec(),empty_strvec(),"double_dict[" + arg + "]",empty_strvec());
    case DTYPE:
      return to_quad(empty_strvec(),empty_strvec(),"get_dtype(" + arg + ")",empty_strvec());
    case TENSOR: {
      //std::string var_name = "tensor_" + std::to_string(tensor_id++);
      std::string arg_dtype = var_caller + "[\"" + name_dtype + "\"]";
      std::string arg_rank = var_caller + "[\"" + name_rank + "\"]";
      std::string device = "device";
      std::string dtype = "get_dtype(" + arg_dtype + ")";
      std::string shape = "{";
      for (size_t i = 0; i < int_offset_size; i++) {
        shape += var_caller + "[\"" + name_dim[i] + "\"]";
        if (i != int_offset_size - 1)
          shape += ",";
      }
      shape += "}";
      std::vector<std::string> tensor_guard =
        std::vector<std::string>({"PathFinderPassIf(is_too_big(" + arg_rank + ", " + shape + "));"});
        
      std::string tensor_init = type_str() + " " + var_name + " = torch_tensor(" + device + ", " + arg_dtype + ", " + arg_rank + ", " + shape + ");";
      return to_quad({tensor_init},empty_strvec(),var_name,tensor_guard);
    }
    case INTVECTOR: {
      //std::string var_name = "int_vector_" + std::to_string(int_vector_id++);
      std::string vec_init = type_str() + " " + var_name + "_ = {";
      for (size_t i = 0; i < int_offset_size; i++) {
        vec_init += var_caller + "[\"" + name_elem[i] + "\"]";
        if (i != int_offset_size - 1)
          vec_init += ",";
      }
      vec_init += "};";
      std::string vec_size_set = type_str() + " " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[" + var_caller + "[\""+ name_size + "\"]" + "]);";
      return to_quad({vec_init, vec_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case FLOATVECTOR: {
      //std::string var_name = "float_vector_" + std::to_string(float_vector_id++);
      std::string vec_init = type_str() + " " + var_name + "_ = {";
      for (size_t i = 0; i < enum_offset_size - 1; i++) {
        vec_init += "double_dict[" + var_caller + "[\"" + name_elem[i] + "\"]" + "]";
        if (i != enum_offset_size - 2)
          vec_init += ",";
      }
      vec_init += "};";
      std::string vec_size_set = type_str() + " " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[" + var_caller + "[\""+ name_size + "\"]" + "]);";
      return to_quad({vec_init, vec_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case INTARRAYREF: {
      //std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = type_str() + " " + var_name + "_ = {";
      for (size_t i = 0; i < int_offset_size; i++) {
        array_init += var_caller + "[\"" + name_elem[i] + "\"]";
        if (i != int_offset_size - 1)
          array_init += ",";
      }
      array_init += "};";
      std::string array_size_set = type_str() + " " + var_name + "(&" + var_name + "_[0],&" + var_name + "_[" + var_caller + "[\""+ name_size + "\"]" + "]);";
      return to_quad({array_init, array_size_set},empty_strvec(),var_name,empty_strvec());
    }
    case EXPANDINGARRAY: {
      //std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = type_str() + " " + var_name + " = " + array_str() + ";";
      return to_quad({array_init},empty_strvec(),var_name,empty_strvec());
    }
    case EXPANDINGARRAYWITHOPTIONALELEM: {
      //std::string var_name = "array_" + std::to_string(array_id++);
      std::string array_init = type_str() + " " + var_name + " = " + array_str() + ";";
      return to_quad({array_init},empty_strvec(),var_name,empty_strvec());
    }
    case OPTIONAL: {
      std::vector<std::string> option_check;
      option_check.push_back("if (" + var_caller + "[\"" + name + "\"])");
      auto t = base->to_code(api_name, is_module);
      std::vector<std::string> preparation = std::get<0>(t);
      std::string expr = std::get<2>(t);
      std::vector<std::string> guard = std::get<3>(t);
      return to_quad(preparation,option_check,expr,guard);
    }
    case VARIANT: {
      //std::string var_name = "enum_" + std::to_string(enum_id++);
      std::vector<std::string> enum_init;
      enum_init.push_back("typedef");
      enum_init.push_back("  c10::variant<");
      if (expandingarray != nullptr)
        enum_init.push_back("    " + expandingarray->type_str() + ",");
      for (size_t i = 0; i < enums.size(); i++) {
        std::string enum_type = "    " + enums[i]->type_str();
        if (i != enums.size()-1)
          enum_type += ",";
        else
          enum_type += ">";
        enum_init.push_back(enum_type);
      }
      enum_init.push_back("  " + var_name + "_t;");
      enum_init.push_back("std::vector<" + var_name + "_t> " + var_name + " = {");
      if (expandingarray != nullptr)
        enum_init.push_back("  " + expandingarray->array_str() + ",");
      for (size_t i = 0; i < enums.size(); i++) {
        std::string enum_elem = "  torch::" + enums[i]->name;
        if (i != enums.size()-1)
          enum_elem += ",";
        enum_init.push_back(enum_elem);
      }
      enum_init.push_back("};");
      return to_quad(enum_init, empty_strvec(), var_name + "[" + var_caller + "[\"" + name + "\"]]",empty_strvec());
    }
    case MAP: {
      std::vector<std::string> preparation;
      std::vector<std::string> option_check;
      std::vector<std::string> map_init;
      std::vector<std::string> guard;
      //std::string var_name = is_module ? "moptions" : "foptions";

      bool separate_init = false;
      if (ctor_params.size() == 1 &&
          ctor_params[0].second->ptype == VARIANT &&
          ctor_params[0].second->enums.size() > 0) {
        map_init.push_back(map_name + " " + var_name + ";");
        map_init.push_back("if (" + var_caller + "[\"" + ctor_params[0].second->name + "\"] == 0) {");
        map_init.push_back("  " + var_name + " = " + map_name + "(torch::" + ctor_params[0].second->enums[0]->name + ");");
        for (size_t i = 1; i < ctor_params[0].second->enums.size(); i++) {
          map_init.push_back("} else if (" + var_caller + "[\"" + ctor_params[0].second->name + "\"] == " + std::to_string(i) + ") {");
          map_init.push_back("  " + var_name + " = " + map_name + "(torch::" + ctor_params[0].second->enums[i]->name + ");");
        }
        map_init.push_back("}");
        if (entries.size() > 0)
          map_init.push_back(var_name);
        separate_init = true;
      } else {
        std::string ctor_param_str;
        for (auto&& p: ctor_params) {
          auto t = p.second->to_code(api_name, is_module);
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
        map_init.push_back("auto " + var_name + " =");
        map_init.push_back("  " + map_name + "(" + ctor_param_str + ")");
      }

      for (auto&& p: entries) {
        auto t = p.second->to_code(api_name, is_module);
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
          option_check.push_back("  " + var_name + "." + p.first + "(" + exp_ + ");");
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
      return to_quad(preparation,empty_strvec(),var_name,guard);
    }
    default:
      assert(false);
  }
}

std::string gen_api_call(std::string api_name, std::vector<std::unique_ptr<Param>>& params, bool is_module, size_t num_input_tensor) {
  std::string code;
  code += "int PathFinderTestOneInput(const pathfinder::Input& x) {";

  std::vector<std::string> preparation;
  std::vector<std::string> positional_arg;
  std::vector<std::string> guard;
  for (auto&& param: params) {
    auto t = param->to_code(api_name, is_module);
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

  code += "\n";
  for (auto line: guard) {
    if (line != "")
      code += "  " + line;
    code += "\n";
  }

  code += "\n  torch::set_num_threads(1);\n";
  code += "  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);\n\n";

  code += "  try {\n";
  for (auto line: preparation) {
    if (line != "")
      code += "    " + line;
    code += "\n";
  }
  if (is_module) {
    code += "\n    auto m = " + api_name + "(";
    for (size_t i = num_input_tensor; i < positional_arg.size(); i++) {
      code += positional_arg[i];
      if (i != positional_arg.size()-1) {
        code += ", ";
      }
    }
    code += ");\n";
    code += "    m->to(device);\n";
    code += "    m->to(get_dtype(" + var_caller + "[\"" + params[0]->name_dtype + "\"]));\n";
  }

  std::string api_call;
  if (is_module) {
    if (api_name == "torch::nn::LSTM" ||
        api_name == "torch::nn::LSTMCell") {
      api_call += "auto result = m->forward(" + positional_arg[0] + ", {{" + positional_arg[1] + ", " + positional_arg[2] + "}})\n";
    } else {
      api_call += "auto result = m->forward(";
      for (size_t i = 0; i < num_input_tensor; i++) {
        api_call += positional_arg[i];
        if (i != num_input_tensor - 1)
          api_call += ", ";
      }
      api_call += ")";
    }
  } else {
    api_call += "auto result = " + api_name + "(";
    for (size_t i = 0; i < positional_arg.size(); i++) {
      api_call += positional_arg[i];
      if (i != positional_arg.size()-1) {
        api_call += ", ";
      }
    }
    api_call += ")";
  }

  code += "\n    PathFinderExecuteTarget(\n";
  code += "      " + api_call + ");\n";

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

std::string gen_code(std::string api_name, std::vector<std::unique_ptr<Param>>& params, bool is_module, size_t num_input_tensor) {
  tensor_id = 0;
  int_vector_id = 0;
  float_vector_id = 0;
  array_id = 0;
  enum_id = 0;

  size_t enum_offset = 0;
  size_t int_offset = 0;
  for (auto&& param: params)
    std::tie(enum_offset, int_offset) = param->set_offset(enum_offset, int_offset);

  std::string code;
  code += gen_header();
  code += gen_setup( params, is_module);
  code += gen_api_call(api_name, params, is_module, num_input_tensor);
  code += gen_footer();

  return code;

}

std::string gen_torch_function_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params) {
  return gen_code(api_name, params, false, 0);
}
std::string gen_torch_module_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params, size_t num_input_tensor) {
  return gen_code(api_name, params, true, num_input_tensor);
}

std::string str_mult(size_t n, std::string str) {
  std::string s;
  for (size_t i = 0; i < n; i++)
    s += str;
  return s;
}
