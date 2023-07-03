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
  INTVECTOR,
  TENSOR,
  EXPENDINGARRAY,
  VARIANT,
  API_OPTION,
};

class Param {
  public:
    Param(ParamType ptype_): ptype(ptype_) {
      assert(ptype == INT || ptype == BOOL || ptype == TENSOR || ptype == INTVECTOR);
    }
    Param(ParamType ptype_, std::string enum_name_)
      : ptype(ptype_), enum_name(enum_name_)
    { assert(ptype == ENUM); }
    Param(ParamType ptype_, long x)
      : ptype(ptype_)
    {
      assert(ptype == EXPENDINGARRAY || ptype == TENSOR);
      if (ptype == EXPENDINGARRAY)
        expendingarray_size = x;
      else if (ptype == TENSOR) {
        tensor_rank = x;
        std::cout << "tensor rank is: " << std::to_string(x) << std::endl;
      }
    }
    Param(ParamType ptype_, std::vector<std::unique_ptr<Param>> variant_types_, std::vector<std::string> enum_vec)
      : ptype(ptype_), variant_types(std::move(variant_types_)), variant_enum_vec(enum_vec) { assert(ptype == VARIANT); }
    Param(
      ParamType ptype_,
      std::string api_option_name_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> api_option_types_)
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
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> api_option_types;

    Optional<unsigned long> default_num = None;
    Optional<std::string> default_enum = None;
    bool default_tensor = false;

    Optional<size_t> tensor_rank = None;

    size_t start_idx = 0;
    size_t idx_size = 0;
    bool variant_use_expending_array = false;
    bool variant_use_enum = false;
    size_t variant_num_enums = 0;
    std::vector<std::string> variant_enum_vec;

    friend size_t set_idx(std::vector<std::unique_ptr<Param>>& params);
    friend void gen_pathfinder_fuzz_target(
      std::string target_api_name,
      std::vector<std::unique_ptr<Param>>& params,
      std::ostream& os);
    friend Optional<std::unique_ptr<Param>> parseIntVector(clang::QualType t, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, ASTContext &Ctx);
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
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
      str += "\n";
      break;
    case BOOL:
      str += str_mult(depth, indent) + "BOOL";
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
      str += "\n";
      break;
    case ENUM:
      str += str_mult(depth, indent) + "ENUM: " + enum_name;
      if (default_enum)
        str += ": default= " + default_enum.getValue();
      str += "\n";
      break;
    case INTVECTOR:
      str += str_mult(depth, indent) + "INTVECTOR";
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
      str += "\n";
      break;
    case TENSOR:
      str += str_mult(depth, indent) + "TENSOR";
      if (default_tensor)
        str += ": default=Tensor()";
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
      str += "\n";
      break;
    case EXPENDINGARRAY:
      str += str_mult(depth, indent) + "EXPENDINGARRAY " + std::to_string(expendingarray_size);
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
      str += "\n";
      break;
    case VARIANT: {
      str += str_mult(depth, indent) + "VARIANT";
      if (default_num)
        str += ": default= " + std::to_string(default_num.getValue());
      str += "  // start idx: " + std::to_string(start_idx) + ", idx size: " + std::to_string(idx_size);
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
      field->dump();
      auto e = field->getInClassInitializer()->IgnoreUnlessSpelledInSource();
      switch (ptype) {
        case INTVECTOR:
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
        default: {
          if (const auto* il = dyn_cast<IntegerLiteral>(e)) {
            unsigned long val = il->getValue().getZExtValue();
            std::cout << "default value: " << std::to_string(val) << std::endl;
            default_num = val;
          }
          break;
        }
      }
    }
  }
}

const size_t MAX_RANK = 5;
const size_t MAX_VECTOR_SIZE = 5;

size_t set_idx(std::vector<std::unique_ptr<Param>>& params) {
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
              entry.second->start_idx = idx;
              entry.second->idx_size = 1;
              idx += entry.second->idx_size;
              break;
            case INTVECTOR:
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
            case EXPENDINGARRAY:
              entry.second->start_idx = idx;
              entry.second->idx_size = entry.second->expendingarray_size;
              idx += entry.second->idx_size;
              break;
            case VARIANT: {
              size_t expendingarray_size_ = 0;
              size_t num_enums = 0;
              //entry.second->to_string();
              for (auto&& param2: entry.second->variant_types) {
                switch (param2->ptype) {
                  case EXPENDINGARRAY:
                    expendingarray_size_ = param2->expendingarray_size;
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
              //std::cout << " expending array size: " << std::to_string(expendingarray_size_) << std::endl;
              //std::cout << " num enums: " << std::to_string(num_enums) << std::endl;
              if (expendingarray_size_ > 0) {
                entry.second->variant_use_expending_array = true;
                entry.second->idx_size = expendingarray_size_;
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
}

void gen_pathfinder_fuzz_target(
  std::string target_api_name,
  std::vector<std::unique_ptr<Param>>& params,
  std::ostream& os)
{
  const size_t NUM_DEVICE = 2;
  const size_t NUM_DTYPE = 11;

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
      case TENSOR: {
        if (param->tensor_rank) {
          for (size_t i = param->start_idx; i < param->start_idx + param->idx_size; i++) {
            os << "    arg[" << std::to_string(i) << "] >= 1,\n";
          }
        } else {
          std::string rank = "arg[" + std::to_string(param->start_idx) + "]";
          os << "    1 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_RANK) << ",\n";
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
            case INTVECTOR:{
              std::string rank = "arg[" + std::to_string(entry.second->start_idx) + "]";
              os << "    0 <= " << rank << ", " << rank << " <= " << std::to_string(MAX_VECTOR_SIZE) << ",\n";
              for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                os << "    arg[" << std::to_string(i) << "] >= 0,\n";
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
            case EXPENDINGARRAY: {
              for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                int min = entry.second->default_num ? entry.second->default_num.getValue() : 0;
                os << "    arg[" << std::to_string(i) << "] >= " << std::to_string(min) << ",\n";
              }
              break;
            }
            case VARIANT: {
              if (entry.second->variant_use_expending_array) {
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
  os << "    torch::TensorOptions toptions =\n";
  os << "      torch::TensorOptions()\n";
  os << "        .device(fuzzer_util::get_device(arg[0]))\n";
  os << "        .dtype(fuzzer_util::get_dtype(arg[1]))\n";
  os << "        .requires_grad(true);\n\n";

  std::vector<std::string> function_args;
  std::vector<std::tuple<Param*, std::string, std::string>> optional_tensor;
  size_t tensor_id = 0;
  size_t int_vector_id = 0;
  for (auto&& param: params) {
    switch (param->ptype) {
      case INT:
      case BOOL:
        function_args.push_back("arg[" + std::to_string(param->start_idx) + "]");
        break;
      case TENSOR: {
        std::string var_name = "tensor_" + std::to_string(tensor_id++);
        if (param->tensor_rank) {
          std::string shape = "std::vector<long> " + var_name + "_shape = {";
          for (size_t i = param->start_idx; i < param->start_idx + param->idx_size; i++) {
            shape += "arg[" + std::to_string(i) + "]";
            if (i != param->start_idx + param->idx_size - 1)
              shape += ",";
          }
          shape += "};\n";
          os << "    " << shape;
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
        }
        os << "    auto " << var_name << " = torch::randn(" << var_name + "_shape, toptions);\n\n";

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
                std::string shape = "std::vector<long> " + var_name + "_shape = {";
                for (size_t i = entry.second->start_idx + 1; i < entry.second->start_idx + entry.second->idx_size; i++) {
                  shape += "arg[" + std::to_string(i) + "]";
                  if (i != entry.second->start_idx + entry.second->idx_size - 1)
                    shape += ",";
                }
                shape += "};\n";
                os << "    " << shape;
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
              }
              os << "    auto " << var_name << " = torch::randn(" << var_name + "_shape, toptions);\n\n";
              optional_tensor.push_back(std::make_tuple(entry.second.get(),option_name,var_name));
              break;
            }
            case EXPENDINGARRAY: {
              option_arg = "{";
              for (size_t i = entry.second->start_idx; i < entry.second->start_idx + entry.second->idx_size; i++) {
                option_arg += "arg[" + std::to_string(i) + "]";
                if (i != entry.second->start_idx + entry.second->idx_size - 1)
                  option_arg += ",";
              }
              option_arg += "}";
              break;
            }
            case VARIANT: {
              if (entry.second->variant_use_expending_array) {
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

}

#endif
