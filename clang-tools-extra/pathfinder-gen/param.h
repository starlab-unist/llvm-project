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
static const std::string newline = "\n";

std::string quoted(std::string param_name) {
  return "\"" + param_name + "\"";
}
std::string sq_quoted(std::string param_name) {
  return "[" + quoted(param_name) + "]";
}
std::string bracket(std::string str="") {
  return "(" + str + ")";
}
std::string square(std::string str) {
  return "[" + str + "]";
}
std::string curly(std::string str) {
  return "{" + str + "}";
}
std::string setup_var(std::string param_name) {
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
std::string join(
  std::string prefix,
  const std::vector<std::string>& strs,
  std::string postfix=newline) {
  std::string joined;
  for (auto& str: strs)
    joined += prefix + str + postfix;
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

void concat(
  std::vector<std::string>& left,
  const std::string& prefix,
  const std::vector<std::string>& right,
  const std::string& postfix=newline)
{
  for (auto& elem: right)
    left.push_back(prefix + elem + postfix);
}

static bool module_mode;
void set_function_mode() { module_mode = false; }
void set_module_mode() { module_mode = true; }
bool is_module_mode() { return module_mode; }

class TorchParam;

std::vector<std::string> get_names(const std::vector<std::unique_ptr<TorchParam>> params) {
  std::vector<std::string> names;
  for (auto& param: params)
    names.push_back(param->get_name());
  return names;
}

class TorchParam {
  public:
    TorchParam(std::string name_): name(name_) {}
    virtual void set_default(Expr* default_expr) {}

    virtual std::string type() const = 0;
    virtual std::string var() const { return ""; }
    virtual std::string initializer() const = 0;
    virtual std::string expr() const {
      return var() != "" ? var() : initializer();
    };

    virtual std::vector<std::string> gen_arg_setup() const { return {}; }
    virtual std::vector<std::string> gen_hard_constraint() const { return {}; }
    virtual std::vector<std::string> gen_soft_constraint() const { return {}; }
    virtual std::vector<std::string> gen_input_pass_condition() const { return {}; }
    virtual std::vector<std::string> gen_arg_initialization() const { return {}; }
    

    std::string get_name() const { name; }
  protected:
    std::string name;
};

class TorchIntParam: public TorchParam {
  public:
    TorchIntParam(std::string name_): TorchParam(name_) {}
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

    virtual std::string type() const { return "long"; }
    virtual std::string initializer() const { return callback_var(name); }

    virtual std::vector<std::string> gen_arg_setup() const {
      return { "PathFinderIntArg" + bracket(quoted(name)) + semicolon  };
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      int min =
        default_value.hasValue() ?
        default_value.getValue() :
        (is_module_mode() ? 1 : 0);
      return { setup_var(name) + gte + std::to_string(min) };
    }
  private:
    Optional<int> default_value = None;
};

class TorchBoundedParam: public TorchParam {
  public:
    TorchBoundedParam(std::string name_, const std::vector<std::string>& value_names_)
      : TorchParam(name_)
    {
      for (auto value_name_: value_names_)
        value_names.push_back(quoted(value_name_));
    }
    TorchBoundedParam(std::string name_, size_t size_): TorchParam(name_) {
      size = size_;
    }
    TorchBoundedParam(std::string name_, const std::string& value_list_var_)
      : TorchParam(name_)
    {
      value_list_var = value_list_var_;
    }

    virtual std::string type() const { assert(false); }
    virtual std::string initializer() const { callback_var(name); }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::string setup_args;
      if (!value_names.empty())
        setup_args = quoted(name) + comma + curly(join(value_names));
      else if (size.hasValue())
        setup_args = quoted(name) + comma + std::to_string(size.getValue());
      else
        setup_args = quoted(name) + comma + value_list_var;
        
      return {"PathFinderEnumArg" + bracket(setup_args) + semicolon};
    }
  protected:
    std::vector<std::string> value_names;
    Optional<size_t> size;
    std::string value_list_var;
};

class TorchBoolParam: public TorchBoundedParam {
  public:
    TorchBoolParam(std::string name_)
      : TorchBoundedParam(name_, std::vector<std::string>({"false", "true"})) {}

    virtual std::string type() const { return "bool"; }
    virtual std::string initializer() const {
      return quoted(type()) + bracket(callback_var(name));
    }
};

class TorchFloatParam: public TorchBoundedParam {
  public:
    TorchFloatParam(std::string name_): TorchBoundedParam(name_, double_dict_list) {}

    virtual std::string type() const { return "double"; }
    virtual std::string initializer() const {
      return double_dict(name);
    }
};

class TorchDtypeParam: public TorchBoundedParam {
  public:
    TorchDtypeParam(std::string name_): TorchBoundedParam(name_, dtype_list) {}

    virtual std::string type() const { return "torch::Dtype"; }
    virtual std::string initializer() const {
      return get_dtype(name);
    }
};

class TorchEnumParam: public TorchParam {
  public:
    TorchEnumParam(std::string name_): TorchParam(name_) {}

    virtual std::string type() const { return "torch::enumtype::" + name; };
    virtual std::string initializer() const { return "torch::" + name; };
};

class TorchVectorParam: public TorchParam {
  public:
    TorchVectorParam(std::string name_): TorchParam(name_) {
      vec_size = std::make_unique<TorchBoundedParam>(name + "_size", MAX_VECTOR_SIZE + 1);
    }

    virtual std::string type() const = 0;
    virtual std::string var() const { return name; }
    virtual std::string initializer() const = 0;

    virtual std::vector<std::string> gen_arg_setup() const {
      return vec_size->gen_arg_setup();
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      return {type() + space + var() + assign + initializer() + semicolon + newline};
    }
  protected:
    std::unique_ptr<TorchBoundedParam> vec_size;
};

class TorchIntVectorParam: public TorchVectorParam {
  public:
    TorchIntVectorParam(std::string name_): TorchVectorParam(name_) {
      for (size_t i = 0; i < MAX_VECTOR_SIZE; i++)
        int_params.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
      for (auto&& int_param: int_params)
        int_param->set_default(0);
    }

    virtual std::string type() const { return "std::vector<long>"; }
    virtual std::string initializer() const {
      return "int_vector" + bracket(vec_size->expr() + comma + to_string(int_params));
    }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      concat(arg_setup, TorchVectorParam::gen_arg_setup());
      for (auto& int_param: int_params)
        concat(arg_setup, int_param->gen_arg_setup());
      return arg_setup;
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

    virtual std::string type() const { return "std::vector<double>"; }
    virtual std::string initializer() const {
      return "float_vector" + bracket(vec_size->expr() + comma + to_string(float_params));
    }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      concat(arg_setup, TorchVectorParam::gen_arg_setup());
      for (auto& float_param: float_params)
        concat(arg_setup, float_param->gen_arg_setup());
      return arg_setup;
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

    virtual std::string type() const {
      return "torch::ExpandingArray<" + std::to_string(size) + ">";
    }
    virtual std::string var() const { return name; }
    virtual std::string initializer() const {
      return type() + bracket(to_string(int_params));
    }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      for (auto& int_param: int_params)
        concat(arg_setup, int_param->gen_arg_setup());
      return arg_setup;
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      std::vector<std::string> soft_ctrs;
      for (auto& int_param: int_params)
        concat(soft_ctrs, int_param->gen_soft_constraint());
      return soft_ctrs;
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      return {type() + space + var() + assign + initializer() + semicolon + newline};
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
      rank = std::make_unique<TorchBoundedParam>(name + "_rank", MAX_RANK + 1);
      for (size_t i = 0; i < MAX_RANK; i++)
        dims.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
      for (auto&& dim: dims)
        dim->set_default(1);
    }

    virtual std::string type() const { return "torch::Tensor "; }
    virtual std::string var() const { return name; }
    virtual std::string initializer() const {
      return "torch_tensor" +  bracket(dtype->expr() + comma + rank->expr() + comma + to_string(dims));
    };

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      concat(arg_setup, dtype->gen_arg_setup());
      concat(arg_setup, rank->gen_arg_setup());
      for (auto& dim: dims)
        concat(arg_setup, dim->gen_arg_setup());
      return arg_setup;
    }
    virtual std::vector<std::string> gen_hard_constraint() const {
      std::vector<std::string> hard_ctrs;
      for (auto& dim: dims)
        concat(hard_ctrs, dim->gen_soft_constraint());
      return hard_ctrs;
    }
    virtual std::vector<std::string> gen_input_pass_condition() const {
      return {"is_too_big" + bracket(rank->expr() + comma + to_string(dims))};
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      return {type() + space + var() + assign + initializer() + semicolon + newline};
    }
  private:
    std::unique_ptr<TorchDtypeParam> dtype;
    std::unique_ptr<TorchBoundedParam> rank;
    std::vector<std::unique_ptr<TorchIntParam>> dims;
};

class TorchOptionalParam: public TorchParam {
  public:
    TorchOptionalParam(std::string name_, std::unique_ptr<TorchParam> param_): TorchParam(name_) {
      has_value = std::make_unique<TorchBoolParam>(name + "_hasValue");
      param = std::move(param_);
    }
    virtual void set_default(Expr* default_expr) {
      param->set_default(default_expr);
    }

    // block being involved by other param
    virtual std::string type() const { "c10::optional<" + param->type() + ">"; }
    virtual std::string var() const { name; }
    virtual std::string initializer() const { has_value->expr() + " ? " + param->expr() +  " : " + "c10::nullopt"; }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      concat(arg_setup, has_value->gen_arg_setup());
      concat(arg_setup, param->gen_arg_setup());
      return arg_setup;
    }
    virtual std::vector<std::string> gen_hard_constraint() const {
      return param->gen_hard_constraint();
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      return param->gen_soft_constraint();
    }
    virtual std::vector<std::string> gen_input_pass_condition() const {
      return param->gen_input_pass_condition();
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      return {type() + space + var() + assign + initializer() + semicolon + newline};
    }
  private:
    std::unique_ptr<TorchBoolParam> has_value;
    std::unique_ptr<TorchParam> param;
};

class TorchVariantParam: public TorchBoundedParam {
  public:
    TorchVariantParam(
      std::string name_,
      std::vector<std::unique_ptr<TorchParam>> params_)
      : TorchBoundedParam(name_, get_names(params_))
    {
      params = std::move(params_);
      for (size_t i = 0; i < params.size(); i++)
        value_names.push_back(name + "_" + std::to_string(i));
    }
    virtual void set_default(Expr* default_expr) {
      for (auto& param: params)
        param->set_default(default_expr);
    }

    virtual std::string type() const { name + "_t"; }
    virtual std::string initializer() const { name + square(callback_var(name)); }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      concat(arg_setup, TorchBoundedParam::gen_arg_setup());
      for (auto& param: params)
        concat(arg_setup, param->gen_arg_setup());
      return arg_setup;
    }
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
    virtual std::vector<std::string> gen_input_pass_condition() const {
      std::vector<std::string> ignore_conds;
      for (auto& param: params)
        concat(ignore_conds, param->gen_input_pass_condition());
      return ignore_conds;
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      std::vector<std::string> preparation_str;
      for (auto& param: params)
        concat(preparation_str, param->gen_arg_initialization());
      concat(preparation_str, gen_typedef());
      concat(preparation_str, gen_vector());
      preparation_str.back() = preparation_str.back() + newline;
    }

    std::vector<std::string> gen_api_options_init(
      std::string api_optons_class_name, std::string api_optons_var_name) const
    {
      std::vector<std::string> api_options_init;
      api_options_init.push_back(api_optons_class_name + space + api_optons_var_name + semicolon);
      for (size_t i = 0; i < params.size(); i++) {
        std::string if_cond = i == 0 ? "" : "} else ";
        api_options_init.push_back(if_cond + "if " + bracket(callback_var(name) + " == " + std::to_string(i)) + " {");
        api_options_init.push_back(
          "  " + api_optons_var_name + assign +
          api_optons_class_name + bracket(params[i]->expr()) + semicolon);
      }
      api_options_init.push_back("}");
      return api_options_init;
    }
  private:
    std::vector<std::unique_ptr<TorchParam>> params;

    std::vector<std::string> gen_typedef() const {
      std::vector<std::string> typedef_str;
      typedef_str.push_back("typedef");
      typedef_str.push_back("  c10::variant<");
      for (auto& param: params)
        typedef_str.push_back("    " + param->type() + comma);
      typedef_str.push_back("  " + type() + semicolon);
    }
    std::vector<std::string> gen_vector() const {
      std::vector<std::string> vector_str;
      vector_str.push_back("std::vector<" + type() + "> " + name + " = {");
      for (auto& param: params)
        vector_str.push_back("  " + param->expr() + comma);
      vector_str.push_back("}" + semicolon);
    }
};

class TorchAPIOptionsParam: public TorchParam {
  public:
    TorchAPIOptionsParam(
      std::string name_,
      std::string api_optons_class_name_,
      std::vector<std::unique_ptr<TorchParam>> ctor_params_,
      std::vector<std::unique_ptr<TorchParam>> member_params_)
      : TorchParam(name_)
    {
      api_optons_class_name = api_optons_class_name_;
      ctor_params = std::move(ctor_params_);
      member_params = std::move(member_params_);
    }

    virtual std::string type() const { return api_optons_class_name; }
    virtual std::string var() const { return name; }

    virtual std::vector<std::string> gen_arg_setup() const {
      std::vector<std::string> arg_setup;
      for (auto& param: ctor_params)
        concat(arg_setup, param->gen_arg_setup());
      for (auto& param: member_params)
        concat(arg_setup, param->gen_arg_setup());
      return arg_setup;
    }
    virtual std::vector<std::string> gen_hard_constraint() const {
      std::vector<std::string> hard_ctrs;
      for (auto& param: ctor_params)
        concat(hard_ctrs, param->gen_hard_constraint());
      for (auto& param: member_params)
        concat(hard_ctrs, param->gen_hard_constraint());
      return hard_ctrs;
    }
    virtual std::vector<std::string> gen_soft_constraint() const {
      std::vector<std::string> soft_ctrs;
      for (auto& param: ctor_params)
        concat(soft_ctrs, param->gen_soft_constraint());
      for (auto& param: member_params)
        concat(soft_ctrs, param->gen_soft_constraint());
      return soft_ctrs;
    }
    virtual std::vector<std::string> gen_input_pass_condition() const {
      std::vector<std::string> ignore_conds;
      for (auto& param: ctor_params)
        concat(ignore_conds, param->gen_input_pass_condition());
      for (auto& param: member_params)
        concat(ignore_conds, param->gen_input_pass_condition());
      return ignore_conds;
    }
    virtual std::vector<std::string> gen_arg_initialization() const {
      std::vector<std::string> preparation_str;

      for (auto& param: ctor_params)
        concat(preparation_str, param->gen_arg_initialization());
      for (auto& param: member_params)
        concat(preparation_str, param->gen_arg_initialization());
      concat(preparation_str, gen_api_options_init());
      preparation_str.back() = preparation_str.back() + newline;
      return preparation_str;
    }
  private:
    std::string api_optons_class_name;
    std::vector<std::unique_ptr<TorchParam>> ctor_params;
    std::vector<std::unique_ptr<TorchParam>> member_params;

    std::vector<std::string> gen_member_param_set() const {
      std::vector<std::string> member_param_set;
      for (auto& param: member_params)
        member_param_set.push_back("." + param->get_name() + bracket(param->expr()));
      return member_param_set;
    }

    std::vector<std::string> gen_api_options_init() const {
      std::vector<std::string> api_options_init;

      bool is_sole_variant_ctor =
        ctor_params.size() == 1 && dynamic_cast<TorchVariantParam*>(ctor_params[0].get());

      if (is_sole_variant_ctor) {
        TorchVariantParam* sole_variant_ctor =
          dynamic_cast<TorchVariantParam*>(ctor_params[0].get());
        concat(
          api_options_init,
          sole_variant_ctor->gen_api_options_init(api_optons_class_name, name));
        api_options_init.push_back(name);
        for (auto& param_set: gen_member_param_set())
          api_options_init.push_back("  " + param_set);
      } else {
        api_options_init.push_back("auto " + name + assign);
        std::string initializer = "  " + api_optons_class_name + "(";
        for (size_t i = 0; i < ctor_params.size(); i++) {
          initializer += ctor_params[i]->expr();
          if (i != ctor_params.size() - 1)
            initializer += comma;
        }
        initializer += ")";
        api_options_init.push_back(initializer);
        for (auto& param_set: gen_member_param_set())
          api_options_init.push_back("    " + param_set);
      }
      api_options_init.back() = api_options_init.back() + semicolon;
      return api_options_init;
    }
};

class TorchAPI {
  public:
    TorchAPI(std::string api_name_): api_name(api_name_) {}
    std::string gen_fuzz_target() const {
      std::vector<std::string> lines;
      concat(lines, header());
      concat(lines, setup());
      concat(lines, callback());
      concat(lines, footer());
      return join(lines, newline);
    }
  protected:
    std::string api_name;
    std::vector<std::unique_ptr<TorchParam>> params;
  private:
    std::vector<std::string> arg_setup_code() const {
      std::vector<std::string> arg_setup;
      for (auto& param: params)
        concat(arg_setup, param->gen_arg_setup());
      return arg_setup;
    }
    std::vector<std::string> hard_constraint_code() const {
      std::vector<std::string> hard_constraint;
      hard_constraint.push_back("PathFinderAddHardConstraint({");
      for (auto& param: params)
        concat(hard_constraint, "  ", param->gen_hard_constraint(), comma);
      hard_constraint.push_back("});");
      return hard_constraint;
    }
    std::vector<std::string> soft_constraint_code() const {
      std::vector<std::string> soft_constraint;
      soft_constraint.push_back("PathFinderAddSoftConstraint({");
      for (auto& param: params)
        concat(soft_constraint, "  ", param->gen_hard_constraint(), comma);
      soft_constraint.push_back("});");
      return soft_constraint;
    }
    std::vector<std::string> input_pass_condition_code() const {
      std::vector<std::string> input_pass_condition;
      for (auto& param: params)
        concat(
          input_pass_condition,
          "PathFinderPassIf(", param->gen_input_pass_condition(), ");");
      return input_pass_condition;
    }
    std::vector<std::string> arg_initialization_code() const {
      std::vector<std::string> arg_initialization;
      for (auto& param: params)
        concat(arg_initialization, param->gen_arg_initialization());
      return arg_initialization;
    }
    virtual std::vector<std::string> api_call_code() const;

    std::vector<std::string> header() const {
      return {
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <c10/util/irange.h>",
        "#include <cassert>",
        "#include <cstdlib>",
        "#include <torch/torch.h>",
        "#include \"pathfinder.h\"",
        "#include \"fuzzer_util.h\"\n",

        "using namespace fuzzer_util;\n",

        "extern \"C\" {\n",
      };
    }
    std::vector<std::string> setup() const {
      std::vector<std::string> setup_code;

      setup_code.push_back("void PathFinderSetup() {");

      concat(setup_code, "  ", arg_setup_code());

      std::vector<std::string> hard_constraint = hard_constraint_code();
      if (!hard_constraint.empty()) {
        setup_code.push_back("  PathFinderAddHardConstraint({");
        concat(setup_code, "    ", hard_constraint, comma + newline);
        setup_code.push_back("  });");
      }

      std::vector<std::string> soft_constraint = soft_constraint_code();
      if (!soft_constraint.empty()) {
        setup_code.push_back("  PathFinderAddSoftConstraint({");
        concat(setup_code, "    ", soft_constraint, comma + newline);
        setup_code.push_back("  });");
      }

      setup_code.push_back("}\n");
    }
    std::vector<std::string> callback() const {
      std::vector<std::string> callback_code;

      callback_code.push_back(
        "int PathFinderTestOneInput(const pathfinder::Input& " + pathfinder_input_var + ") {");
      callback_code.push_back(
        "  torch::set_num_threads(1);\n");

      concat(callback_code, "  ", input_pass_condition_code());

      callback_code.push_back("\n  try\n");
      concat(callback_code, "    ", arg_initialization_code());
      concat(callback_code, "    ", api_call_code());
      callback_code.push_back("  } catch (std::exception& e) {");
      callback_code.push_back("    return -2;");
      callback_code.push_back("  }\n");
      callback_code.push_back("  return 0;");
      callback_code.push_back("}\n");
    }
    std::vector<std::string> footer() const {
      return {
       "}  // extern \"C\"\n",
      };
    };
};

class TorchFunction: public TorchAPI {
  public:
    TorchFunction(
      std::string func_name,
      std::vector<std::unique_ptr<TorchParam>> params_)
      : TorchAPI(func_name)
    {
      params = std::move(params_);
    }
  private:
    virtual std::vector<std::string> api_call_code() const {
      std::string api_call =
        "auto result = " + api_name + "(";
      for (size_t i = 0; i < params.size(); i++) {
        api_call += params[i]->expr();
        if (i != params.size() - 1)
          api_call += comma;
      }
      api_call += ")";
      
      return {
        "PathFinderExecuteTarget(",
        "  " + api_call + ");",
      };
    }
};

class TorchModule: public TorchAPI {
  public:
    TorchModule(
      std::string module_name,
      std::unique_ptr<TorchParam> module_dtype_,
      std::vector<std::unique_ptr<TorchParam>> module_params_,
      std::vector<std::unique_ptr<TorchParam>> forward_params_)
      : TorchAPI(module_name)
    {
      module_dtype = module_dtype_.get();
      params.push_back(std::move(module_dtype_));
      for (auto& param: module_params_) {
        module_params.push_back(param.get());
        params.push_back(std::move(param));
      }
      for (auto& param: forward_params_) {
        forward_params.push_back(param.get());
        params.push_back(std::move(param));
      }
    }
  private:
    TorchParam* module_dtype;
    std::vector<TorchParam*> module_params;
    std::vector<TorchParam*> forward_params;

    virtual std::vector<std::string> api_call_code() const {
      const std::string module_var = "module";

      std::string module_init =
        "auto " + module_var + assign + api_name + "(";
      for (size_t i = 0; i < params.size(); i++) {
        module_init += module_params[i]->expr();
        if (i != params.size() - 1)
          module_init += comma;
      }
      module_init += ");\n";
      
      std::string forward_call =
        "auto result = " + module_var + "->forward(";
      for (size_t i = 0; i < params.size(); i++) {
        forward_call += forward_params[i]->expr();
        if (i != params.size() - 1)
          forward_call += comma;
      }
      forward_call += ")";

      return {
        module_init,
        module_var + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
        "PathFinderExecuteTarget(",
        "  " + forward_call + ");",
      };
    }
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
    /* friend Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx); */
};

std::string gen_torch_function_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params);
std::string gen_torch_module_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params, size_t num_input_tensor);

#endif
