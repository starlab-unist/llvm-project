#ifndef PATHFINDER_GEN_PARAM
#define PATHFINDER_GEN_PARAM

#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace clang;

const size_t MAX_RANK = 5;
const size_t MAX_VECTOR_SIZE = 6;
const size_t DOUBLE_DICT_SIZE = 20;

static const std::string symbolic_int_var = "sym_int_arg";
static const std::string pathfinder_input_var = "x";
static const std::string double_value_dictionary = "double_dict";
static const std::string double_dict_list = "double_dict_list";
static const std::string dtype_list = "dtype_list";

static const std::string space = " ";
static const std::string gte = " >= ";
static const std::string assign = " = ";
static const std::string comma = ", ";
static const std::string semicolon = ";";

std::string quoted(std::string param_name) {
  return "\"" + param_name + "\"";
}
std::string sq_quoted(std::string param_name) {
  return "[" + quoted(param_name) + "]";
}
std::string bracket(std::string str) {
  return "(" + str + ")";
}
std::string square(std::string str) {
  return "[" + str + "]";
}
std::string curly(std::string str) {
  return "{" + str + "}";
}
std::string ctr_var(std::string param_name) {
  return symbolic_int_var + sq_quoted(param_name);
}
std::string callback_var(std::string param_name) {
  return pathfinder_input_var + sq_quoted(param_name);
}
std::string double_dict(std::string param_name) {
  return double_value_dictionary + callback_var(param_name);
}
std::string get_dtype(std::string param_name) {
  return "get_dtype(" + callback_var(param_name) + ")";
}
std::string join(const std::vector<std::string>& strs, std::string sep=comma) {
  std::string joined;
  for (size_t i = 0; i < strs.size(); i++) {
    joined += strs[i];
    if (i != strs.size() - 1)
      joined += sep;
  }
  return joined;
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

template<typename T>
void concat(std::vector<T>& left, const std::vector<T>& right) {
  for (auto& elem: right)
    left.push_back(elem);
}

static bool module_mode;
void set_function_mode() { module_mode = false; }
void set_module_mode() { module_mode = true; }
bool is_module_mode() { return module_mode; }

class TorchParam {
  public:
    TorchParam(std::string name_): name(name_) {}
    std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      for (auto& enum_arg: enum_arg_string())
        arg_setup.push_back("PathFinderEnumArg" + bracket(enum_arg) + semicolon);
      for (auto& int_arg: int_arg_string())
        arg_setup.push_back("PathFinderIntArg" + bracket(int_arg) + semicolon);
      return arg_setup;
    }
    virtual std::vector<std::string> enum_arg_string() const { return {}; }
    virtual std::vector<std::string> int_arg_string() const { return {}; }

    virtual void set_default(Expr* default_expr) {}

    virtual std::string type() const = 0;
    virtual std::string var() const { return ""; }
    virtual std::string initializer() const = 0;
    virtual std::string expr() const {
      return var() != "" ? var() : initializer();
    };

    virtual std::vector<std::string> gen_hard_constraint() const { return {}; }
    virtual std::vector<std::string> gen_soft_constraint() const { return {}; }
    virtual std::vector<std::string> gen_preparation() const { return {}; }
    virtual std::vector<std::string> gen_ignore_condition() const { return {}; }
    virtual std::string gen_optional_condition() const { return ""; }

  protected:
    std::string name;
};

class TorchIntParam: public TorchParam {
  public:
    TorchIntParam(std::string name_): TorchParam(name_) {}
    virtual std::vector<std::string> int_arg_string() const {
      return { quoted(name) };
    }

    virtual std::string type() const { return "long"; }
    virtual std::string initializer() const { return callback_var(name); }

    virtual void set_default(Expr* default_expr) {
      if (default_expr == nullptr)
        return;

      if (const auto* il = dyn_cast<IntegerLiteral>(default_expr)) {
        assert(il != nullptr);
        unsigned long val = il->getValue().getZExtValue();
        default_value = val;
      }
    }
    void set_default(int value) {
      default_value = value;
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      if (!default_value.hasValue())
        return {};
      return
        {ctr_var(name) + gte + std::to_string(default_value.getValue())};  
    }
  private:
    Optional<int> default_value = None;
};

class TorchEnumParam: public TorchParam {
  public:
    TorchEnumParam(std::string name_, size_t size_): TorchParam(name_) {
      size = size_;
    }
    TorchEnumParam(std::string name_, const std::vector<std::string>& enumerators_)
      : TorchParam(name_)
    {
      for (auto enumerator_: enumerators_)
        enumerators.push_back(quoted(enumerator_));
    }
    TorchEnumParam(std::string name_, const std::string& enum_list_var): TorchParam(name_) {
      enumerators.push_back(enum_list_var);
    }
    virtual std::vector<std::string> enum_arg_string() const {
      if (size.hasValue())
        return {quoted(name) + comma + std::to_string(size.getValue())};
      else if (!enumerators.empty())
        return {quoted(name) + comma + curly(join(enumerators))};
      else
        return {quoted(name) + comma + enum_list_var};
    }

    virtual std::string type() const { assert(false); }
    virtual std::string initializer() const { callback_var(name); }
  private:
    Optional<size_t> size;
    std::vector<std::string> enumerators;
    std::string enum_list_var;
};

class TorchBoolParam: public TorchEnumParam {
  public:
    TorchBoolParam(std::string name_)
      : TorchEnumParam(name_, std::vector<std::string>({"false", "true"})) {}

    virtual std::string type() const { return "bool"; }
    virtual std::string initializer() const {
      return quoted(type()) + bracket(callback_var(name));
    }
};

class TorchFloatParam: public TorchEnumParam {
  public:
    TorchFloatParam(std::string name_): TorchEnumParam(name_, double_dict_list) {}

    virtual std::string type() const { return "double"; }
    virtual std::string initializer() const {
      return double_dict(name);
    }
};

class TorchDtypeParam: public TorchEnumParam {
  public:
    TorchDtypeParam(std::string name_): TorchEnumParam(name_, dtype_list) {}

    virtual std::string type() const { return "torch::Dtype"; }
    virtual std::string initializer() const {
      return get_dtype(name);
    }
};

/* class TorchEnumParam: public TorchParam {
  public:
    TorchEnumParam(std::string name_): TorchParam(name_) {}
    virtual std::string expr() const {
      return name;
    }
}; */

class TorchVectorParam: public TorchParam {
  public:
    TorchVectorParam(std::string name_): TorchParam(name_) {
      vec_size = std::make_unique<TorchEnumParam>(name + "_size", MAX_VECTOR_SIZE + 1);
    }
    virtual std::vector<std::string> enum_arg_string() const {
      return vec_size->enum_arg_string();
    }

    virtual std::string type() const = 0;
    virtual std::string var() const { return name; }
    virtual std::string initializer() const = 0;
    virtual std::vector<std::string> gen_preparation() const {
      return {type() + space + var() + assign + initializer() + semicolon};
    }
  protected:
    std::unique_ptr<TorchEnumParam> vec_size;
};

class TorchIntVectorParam: public TorchVectorParam {
  public:
    TorchIntVectorParam(std::string name_): TorchVectorParam(name_) {
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        int_params.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
      for (auto&& int_param: int_params)
        int_param->set_default(0);
    }
    virtual std::vector<std::string> int_arg_string() const {
      std::vector<std::string> int_arg_str;
      for (auto& int_param: int_params)
        concat(int_arg_str, int_param->int_arg_string());
      return int_arg_str;
    }

    virtual std::string type() const { return "std::vector<long>"; }
    virtual std::string initializer() const {
      return "int_vector" + bracket(vec_size->expr() + comma + to_string(int_params));
    }

    virtual std::vector<std::string> gen_soft_constraint() const {
      std::vector<std::string> soft_ctrs;
      for (auto& int_param: int_params)
        concat(soft_ctrs, int_param->gen_soft_constraint());
      return soft_ctrs;
    }
  private:
    std::vector<std::unique_ptr<TorchIntParam>> int_params;
};

class TorchFloatVectorParam: public TorchVectorParam {
  public:
    TorchFloatVectorParam(std::string name_): TorchVectorParam(name_) {
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        float_params.push_back(std::make_unique<TorchFloatParam>(name + "_" + std::to_string(i)));
    }
    virtual std::vector<std::string> enum_arg_string() const {
      std::vector<std::string> float_arg_str;
      concat(float_arg_str, TorchVectorParam::enum_arg_string());
      for (auto& float_param: float_params)
        concat(float_arg_str, float_param->enum_arg_string());
      return float_arg_str;
    }

    virtual std::string type() const { return "std::vector<double>"; }
    virtual std::string initializer() const {
      return "float_vector" + bracket(vec_size->expr() + comma + to_string(float_params));
    }
  private:
    std::vector<std::unique_ptr<TorchFloatParam>> float_params;
};

class TorchExpandingArrayParam: public TorchParam {
  public:
    TorchExpandingArrayParam(std::string name_, size_t size_): TorchParam(name_) {
      size = size_;
      for (size_t i = 0; i < size; i++)
        int_params.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
      for (auto&& int_param: int_params)
        int_param->set_default(is_module_mode() ? 1 : 0); // TDOO: is it required?
    }
    virtual void set_default(Expr* default_expr) {
      for (auto&& int_param: int_params)
        int_param->set_default(default_expr);
    }
    virtual std::vector<std::string> int_arg_string() const {
      std::vector<std::string> int_arg_str;
      for (auto& int_param: int_params)
        concat(int_arg_str, int_param->int_arg_string());
      return int_arg_str;
    }

    virtual std::string type() const {
      return "torch::ExpandingArray<" + std::to_string(size) + ">";
    }
    virtual std::string var() const { return name; }
    virtual std::string initializer() const {
      return type() + bracket(to_string(int_params));
    }

    virtual std::vector<std::string> gen_soft_constraint() const {
      std::vector<std::string> soft_ctrs;
      for (auto& int_param: int_params)
        concat(soft_ctrs, int_param->gen_soft_constraint());
      return soft_ctrs;
    }
    virtual std::vector<std::string> gen_preparation() const {
      return {type() + space + var() + assign + initializer() + semicolon};
    }
  protected:
    size_t size;
    std::vector<std::unique_ptr<TorchIntParam>> int_params;
};

class TorchExpandingArrayWithOptionalElemParam: public TorchExpandingArrayParam {
  public:
    TorchExpandingArrayWithOptionalElemParam(std::string name_, size_t size_)
      : TorchExpandingArrayParam(name_, size_)
    {
      for (auto&& int_param: int_params)
        int_param->set_default(0);
    }

    virtual std::string type() const {
      return "torch::ExpandingArrayWithOptionalElem<" + std::to_string(size) + ">";
    }
    virtual std::string initializer() const {
      return "expandingarray_with_optional_elem<" + std::to_string(size) + ">" + bracket(to_string(int_params));
    }
};

class TorchTensorParam: public TorchParam {
  public:
    TorchTensorParam(std::string name_): TorchParam(name_) {
      dtype = std::make_unique<TorchDtypeParam>(name + "_dtype");
      rank = std::make_unique<TorchEnumParam>(name + "_rank", MAX_RANK + 1);
      for (size_t i = 0; i < MAX_RANK; i++)
        dims.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
      for (auto&& dim: dims)
        dim->set_default(1);
    }
    virtual std::vector<std::string> enum_arg_string() const {
      std::vector<std::string> enum_arg_str;
      concat(enum_arg_str, dtype->enum_arg_string());
      concat(enum_arg_str, rank->enum_arg_string());
      return enum_arg_str;
    }
    virtual std::vector<std::string> int_arg_string() const {
      std::vector<std::string> int_arg_str;
      for (auto& dim: dims)
        concat(int_arg_str, dim->int_arg_string());
      return int_arg_str;
    }

    virtual std::string type() const { return "torch::Tensor "; }
    virtual std::string var() const { return name; }
    virtual std::string initializer() const {
      return "torch_tensor" +  bracket(dtype->expr() + comma + rank->expr() + comma + to_string(dims));
    };

    virtual std::vector<std::string> gen_hard_constraint() const {
      std::vector<std::string> hard_ctrs;
      for (auto& dim: dims)
        concat(hard_ctrs, dim->gen_soft_constraint());
      return hard_ctrs;
    }
    virtual std::vector<std::string> gen_preparation() const {
      return {type() + space + var() + assign + initializer() + semicolon};
    }
    virtual std::vector<std::string> gen_ignore_condition() const {
      return {"is_too_big" + bracket(rank->expr() + comma + to_string(dims))};
    }
  private:
    std::unique_ptr<TorchDtypeParam> dtype;
    std::unique_ptr<TorchEnumParam> rank;
    std::vector<std::unique_ptr<TorchIntParam>> dims;
};

class TorchOptionalParam: public TorchParam {
  public:
    TorchOptionalParam(std::string name_, std::unique_ptr<TorchParam> param_): TorchParam(name_) {
      opt = std::make_unique<TorchEnumParam>(name + "_opt", std::vector<std::string>({"none", "some"}));
      param = std::move(param_);
    }
    virtual std::vector<std::string> enum_arg_string() const {
      std::vector<std::string> enum_arg_str;
      concat(enum_arg_str, opt->enum_arg_string());
      concat(enum_arg_str, param->enum_arg_string());
      return enum_arg_str;
    }
    virtual std::vector<std::string> int_arg_string() const {
      return param->int_arg_string();
    }
    virtual void set_default(Expr* default_expr) {
      param->set_default(default_expr);
    }

    // block being involved by other param
    virtual std::string type() const { assert(false); }
    virtual std::string var() const { assert(false); }
    virtual std::string initializer() const { assert(false); }

    virtual std::vector<std::string> gen_hard_constraint() const {
      return param->gen_hard_constraint();
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      return param->gen_soft_constraint();
    }
    virtual std::vector<std::string> gen_preparation() const {
      return param->gen_preparation();
    }
    virtual std::vector<std::string> gen_ignore_condition() const {
      return param->gen_ignore_condition();
    }
    virtual std::string gen_optional_condition() const {
      // does not support nested optional(and not likely to exist)
      return opt->expr();
    }
  private:
    std::unique_ptr<TorchEnumParam> opt;
    std::unique_ptr<TorchParam> param;
};

class TorchVariantParam: public TorchEnumParam {
  public:
    TorchVariantParam(
      std::string name_,
      const std::vector<std::string>& enumerators_,
      std::vector<std::unique_ptr<TorchParam>> params_={}): TorchEnumParam(name_, enumerators_)
    {
      params = std::move(params_);
    }
    virtual std::vector<std::string> enum_arg_string() const {
      std::vector<std::string> enum_arg_str;
      concat(enum_arg_str, TorchEnumParam::enum_arg_string());
      for (auto& param: params)
        concat(enum_arg_str, param->enum_arg_string());
      return enum_arg_str;
    }
    virtual std::vector<std::string> int_arg_string() const {
      std::vector<std::string> int_arg_str;
      for (auto& param: params)
        concat(int_arg_str, param->int_arg_string());
      return int_arg_str;
    }
    virtual void set_default(Expr* default_expr) {
      // TODO: is it correct?
      for (auto& param: params)
        param->set_default(default_expr);
    }

    // block being involved by other param
    virtual std::string type() const { assert(false); }
    virtual std::string var() const { assert(false); }
    virtual std::string initializer() const { assert(false); }

    virtual std::vector<std::string> gen_hard_constraint() const {
      std::vector<std::string> hard_ctrs;
      for (auto& param: params)
        concat(hard_ctrs, param->gen_hard_constraint());
      return hard_ctrs;
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      std::vector<std::string> soft_ctrs;
      for (auto& param: params)
        concat(soft_ctrs, param->gen_soft_constraint());
      return soft_ctrs;
    }
    virtual std::vector<std::string> gen_preparation() const {
      //TODO
    }
    virtual std::vector<std::string> gen_ignore_condition() const {
      std::vector<std::string> ignore_conds;
      for (auto& param: params)
        concat(ignore_conds, param->gen_ignore_condition());
      return ignore_conds;
    }
  private:
    std::vector<std::unique_ptr<TorchParam>> params;
};





enum ParamType {
  INT,
  BOOL,
  FLOAT,
  DTYPE,
  ENUM,
  INTVECTOR,
  FLOATVECTOR,
  INTARRAYREF,
  EXPANDINGARRAY,
  EXPANDINGARRAYWITHOPTIONALELEM,
  TENSOR,
  OPTIONAL,
  VARIANT,
  MAP,
};

class Param {
  public:
    Param(ParamType ptype_, std::string name_);
    Param(ParamType ptype_, std::string name_, long size);
    Param(ParamType ptype_, std::string name_, std::unique_ptr<Param> base_);
    Param(ParamType ptype_, std::string name_, std::vector<std::unique_ptr<Param>> enums_, std::unique_ptr<Param> expandingarray_);
    Param(
      ParamType ptype_,
      std::string name_,
      std::string map_name_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries_);
    std::pair<size_t, size_t> set_offset(size_t enum_offset, size_t int_offset);
    void set_default(std::string param_name, const CXXRecordDecl* cdecl);
    void setup_arg(std::vector<std::string>& args);
    void constraint(std::vector<std::string>& hard_ctrs, std::vector<std::string>& soft_ctrs, bool is_module);
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>> to_code(std::string api_name, bool is_module);
    bool is_map();
  private:
    std::string type_str();
    std::string array_str();

    ParamType ptype;
    std::string name;
    size_t enum_offset_start;
    size_t enum_offset_size = 0;
    size_t int_offset_start;
    size_t int_offset_size = 0;

    // INT
    Optional<size_t> default_int = None;

    // TENSOR
    std::string name_dtype;
    std::string name_rank;
    std::vector<std::string> name_dim;

    // ARRAY, VECTOR
    size_t array_size;
    std::string name_size;
    std::vector<std::string> name_elem;

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
    friend std::string gen_api_call(std::string api_name, std::vector<std::unique_ptr<Param>>& params, bool is_module, size_t num_input_tensor);
    friend void gen_pathfinder_fuzz_target(
      std::string target_api_name,
      std::vector<std::unique_ptr<Param>>& params,
      std::ostream& os);
    friend Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx);
};

std::string gen_torch_function_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params);
std::string gen_torch_module_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params, size_t num_input_tensor);

#endif
