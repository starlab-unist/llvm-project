#include "param.h"

const size_t MAX_RANK = 5;

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

std::vector<std::string> get_names(const std::vector<std::unique_ptr<TorchParam>>& params) {
  std::vector<std::string> names;
  for (auto& param: params)
    names.push_back(param->get_name());
  return names;
}

static bool module_mode;
void set_function_mode() { module_mode = false; }
void set_module_mode() { module_mode = true; }
bool is_module_mode() { return module_mode; }


TorchIntParam::TorchIntParam(std::string name_): TorchParam(TPK_Int, name_) {}
void TorchIntParam::set_default(Expr* default_expr) {
  if (default_expr == nullptr)
    return;

  if (const auto* il = dyn_cast<IntegerLiteral>(default_expr)) {
    assert(il != nullptr);
    unsigned long val = il->getValue().getZExtValue();
    default_value = val;
  }
}
void TorchIntParam::set_default(int value) {
  default_value = value;
}

std::string TorchIntParam::type() const {
  return "long";
}
std::string TorchIntParam::initializer() const {
  return callback_var(name);
}

std::vector<std::string> TorchIntParam::gen_arg_setup() const {
  return { "PathFinderIntArg" + bracket(quoted(name)) + semicolon  };
}
std::vector<std::string> TorchIntParam::gen_soft_constraint() const {
  int min =
    default_value.hasValue() ?
    default_value.getValue() :
    (is_module_mode() ? 1 : 0);
  return { setup_var(name) + gte + std::to_string(min) };
}

bool TorchIntParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Int;
}



TorchBoundedParam::TorchBoundedParam(TorchParamKind kind_, std::string name_, const std::vector<std::string>& value_names_)
  : TorchParam(kind_, name_)
{
  assert(TPK_Bounded_First <= get_kind() && get_kind() <= TPK_Bounded_Last);
  for (auto value_name_: value_names_)
    value_names.push_back(quoted(value_name_));
}
TorchBoundedParam::TorchBoundedParam(TorchParamKind kind_, std::string name_, size_t size_)
  : TorchParam(kind_, name_)
{
  assert(TPK_Bounded_First <= get_kind() && get_kind() <= TPK_Bounded_Last);
  size = size_;
}
TorchBoundedParam::TorchBoundedParam(TorchParamKind kind_, std::string name_, const std::string& value_list_var_)
  : TorchParam(kind_, name_)
{
  assert(TPK_Bounded_First <= get_kind() && get_kind() <= TPK_Bounded_Last);
  value_list_var = value_list_var_;
}

std::vector<std::string> TorchBoundedParam::gen_arg_setup() const {
  std::string setup_args;
  if (!value_names.empty())
    setup_args = quoted(name) + comma + curly(join_strs(value_names));
  else if (size.hasValue())
    setup_args = quoted(name) + comma + std::to_string(size.getValue());
  else
    setup_args = quoted(name) + comma + value_list_var;
    
  return {"PathFinderEnumArg" + bracket(setup_args) + semicolon};
}


TorchBoundedIntParam::TorchBoundedIntParam(std::string name_, size_t size_)
  : TorchBoundedParam(TPK_BoundedInt, name_, size_) {}

std::string TorchBoundedIntParam::type() const {
  return "size_t";
}
std::string TorchBoundedIntParam::initializer() const {
  return bracket(type()) + bracket(callback_var(name));
}

bool TorchBoundedIntParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_BoundedInt;
}


TorchBoolParam::TorchBoolParam(std::string name_)
  : TorchBoundedParam(TPK_Bool, name_, std::vector<std::string>({"false", "true"})) {}

std::string TorchBoolParam::type() const {
  return "bool";
}
std::string TorchBoolParam::initializer() const {
  return  bracket(type()) + bracket(callback_var(name));
}

bool TorchBoolParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Bool;
}


TorchFloatParam::TorchFloatParam(std::string name_): TorchBoundedParam(TPK_Float, name_, double_dict_list) {}

std::string TorchFloatParam::type() const {
  return "double";
}
std::string TorchFloatParam::initializer() const {
  return double_dict(name);
}

bool TorchFloatParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Float;
}


TorchDtypeParam::TorchDtypeParam(std::string name_): TorchBoundedParam(TPK_Dtype, name_, dtype_list) {}

std::string TorchDtypeParam::type() const {
  return "torch::Dtype";
}
std::string TorchDtypeParam::initializer() const {
  return get_dtype(name);
}

bool TorchDtypeParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Dtype;
}


TorchVariantParam::TorchVariantParam(
  std::string name_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchBoundedParam(TPK_Variant, name_, get_names(params_))
{
  params = std::move(params_);
}
void TorchVariantParam::set_default(Expr* default_expr) {
  for (auto& param: params)
    param->set_default(default_expr);
}

std::string TorchVariantParam::type() const {
  return name + "_t";
}
std::string TorchVariantParam::initializer() const {
  return name + square(callback_var(name));
}

std::vector<std::string> TorchVariantParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, TorchBoundedParam::gen_arg_setup());
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchVariantParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchVariantParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchVariantParam::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& param: params)
    concat(ignore_conds, param->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TorchVariantParam::gen_arg_initialization() const {
  std::vector<std::string> preparation_str;
  for (auto& param: params)
    concat(preparation_str, param->gen_arg_initialization());
  concat(preparation_str, gen_typedef());
  concat(preparation_str, gen_vector());
  preparation_str.back() = preparation_str.back() + newline;
  return preparation_str;
}

std::vector<std::string> TorchVariantParam::gen_api_options_init(
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

bool TorchVariantParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Variant;
}

std::vector<std::string> TorchVariantParam::gen_typedef() const {
  std::vector<std::string> typedef_str;
  typedef_str.push_back("typedef");
  typedef_str.push_back("  c10::variant<");
  for (auto& param: params)
    typedef_str.push_back("    " + param->type() + comma);
  typedef_str.push_back("  " + type() + semicolon);
  return typedef_str;
}
std::vector<std::string> TorchVariantParam::gen_vector() const {
  std::vector<std::string> vector_str;
  vector_str.push_back("std::vector<" + type() + "> " + name + " = {");
  for (auto& param: params)
    vector_str.push_back("  " + param->expr() + comma);
  vector_str.push_back("}" + semicolon);
  return vector_str;
}


TorchEnumParam::TorchEnumParam(std::string name_, std::string enum_name_)
  : TorchParam(TPK_Enum, name_)
{
  enum_name = enum_name_;
}

std::string TorchEnumParam::type() const {
  return "torch::enumtype::" + enum_name;
}
std::string TorchEnumParam::initializer() const {
  return "torch::" + enum_name;
}

bool TorchEnumParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Enum;
}


TorchVectorParam::TorchVectorParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchParam(TPK_Vector, name_)
{
  vec_size = std::make_unique<TorchBoundedIntParam>(name + "_size", MAX_VECTOR_SIZE + 1);
  params = std::move(params_);
  assert(params.size() == MAX_VECTOR_SIZE);
}

std::string TorchVectorParam::type() const {
  return "std::vector<" + params[0]->type() + ">";
}
std::string TorchVectorParam::var() const {
  return name;
}
std::string TorchVectorParam::initializer() const {
  return "vector_init" + bracket(vec_size->expr() + comma + to_string(params));
}

std::vector<std::string> TorchVectorParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, vec_size->gen_arg_setup());
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchVectorParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchVectorParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchVectorParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}

bool TorchVectorParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Vector;
}


TorchSizedArrayParam::TorchSizedArrayParam(
  TorchParamKind kind_,
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchParam(kind_, name_)
{
  assert(TPK_SizedArray_First <= get_kind() && get_kind() <= TPK_SizedArray_Last);
  size = size_;
  params = std::move(params_);
  assert(params.size() == size);
}
void TorchSizedArrayParam::set_default(Expr* default_expr) {
  for (auto&& param: params)
    param->set_default(default_expr);
}

std::string TorchSizedArrayParam::var() const {
  return name;
}

std::vector<std::string> TorchSizedArrayParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchSizedArrayParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchSizedArrayParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchSizedArrayParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}


TorchExpandingArrayParam::TorchExpandingArrayParam(
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchSizedArrayParam(TPK_ExpandingArray, name_, size_, std::move(params_)) {}

std::string TorchExpandingArrayParam::type() const {
  return "torch::ExpandingArray<" + std::to_string(size) + comma + params[0]->type() + ">";
}
std::string TorchExpandingArrayParam::initializer() const {
  return type() + bracket(to_string(params));
}

bool TorchExpandingArrayParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_ExpandingArray;
}


TorchExpandingArrayWithOptionalElemParam::TorchExpandingArrayWithOptionalElemParam(
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchSizedArrayParam(TPK_ExpandingArrayWithOptionalElem, name_, size_, std::move(params_))
{
  for (auto& param: params)
    assert(isa<TorchOptionalParam>(param.get()));
}

std::string TorchExpandingArrayWithOptionalElemParam::type() const {
  
  return "torch::ExpandingArrayWithOptionalElem<" + std::to_string(size) + comma + base_type() + ">";
}
std::string TorchExpandingArrayWithOptionalElemParam::initializer() const {
  return
    "expandingarray_with_optional_elem<" +
      std::to_string(size) + comma + base_type() +
    ">" + bracket(to_string(params));
}

bool TorchExpandingArrayWithOptionalElemParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_ExpandingArrayWithOptionalElem;
}

std::string TorchExpandingArrayWithOptionalElemParam::base_type() const {
  TorchOptionalParam* param = dyn_cast<TorchOptionalParam>(params[0].get());
  return param->base_type();
}


TorchTensorParam::TorchTensorParam(std::string name_): TorchParam(TPK_Tensor, name_) {
  dtype = std::make_unique<TorchDtypeParam>(name + "_dtype");
  rank = std::make_unique<TorchBoundedIntParam>(name + "_rank", MAX_RANK + 1);
  for (size_t i = 0; i < MAX_RANK; i++)
    dims.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i)));
  for (auto&& dim: dims)
    dim->set_default(1);
}

std::string TorchTensorParam::type() const { return "torch::Tensor "; }
std::string TorchTensorParam::var() const { return name; }
std::string TorchTensorParam::initializer() const {
  return "torch_tensor" +  bracket(dtype->expr() + comma + rank->expr() + comma + to_string(dims));
};

std::vector<std::string> TorchTensorParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, dtype->gen_arg_setup());
  concat(arg_setup, rank->gen_arg_setup());
  for (auto& dim: dims)
    concat(arg_setup, dim->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchTensorParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& dim: dims)
    concat(hard_ctrs, dim->gen_soft_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchTensorParam::gen_input_pass_condition() const {
  return {"is_too_big" + bracket(rank->expr() + comma + to_string(dims))};
}
std::vector<std::string> TorchTensorParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}

bool TorchTensorParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Tensor;
}


TorchOptionalParam::TorchOptionalParam(std::string name_, std::unique_ptr<TorchParam> param_)
  : TorchParam(TPK_Optional, name_)
{
  has_value = std::make_unique<TorchBoolParam>(name + "_hasValue");
  param = std::move(param_);
}
void TorchOptionalParam::set_default(Expr* default_expr) {
  param->set_default(default_expr);
}

std::string TorchOptionalParam::type() const {
  return "c10::optional<" + param->type() + ">";
}
std::string TorchOptionalParam::var() const {
  return name;
}
std::string TorchOptionalParam::initializer() const {
  return has_value->expr() + " ? " + param->expr() +  " : " + "c10::nullopt";
}

std::vector<std::string> TorchOptionalParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, has_value->gen_arg_setup());
  concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchOptionalParam::gen_hard_constraint() const {
  return param->gen_hard_constraint();
}
std::vector<std::string> TorchOptionalParam::gen_soft_constraint() const {
  return param->gen_soft_constraint();
}
std::vector<std::string> TorchOptionalParam::gen_input_pass_condition() const {
  return param->gen_input_pass_condition();
}
std::vector<std::string> TorchOptionalParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}

std::string TorchOptionalParam::base_type() const {
  return param->type();
}

bool TorchOptionalParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Optional;
}


TorchAPIOptionsParam::TorchAPIOptionsParam(
  std::string name_,
  std::string api_optons_class_name_,
  std::vector<std::unique_ptr<TorchParam>> ctor_params_,
  std::vector<std::unique_ptr<TorchParam>> member_params_)
  : TorchParam(TPK_APIOptions, name_)
{
  api_optons_class_name = api_optons_class_name_;
  ctor_params = std::move(ctor_params_);
  member_params = std::move(member_params_);
}

std::string TorchAPIOptionsParam::type() const {
  return api_optons_class_name;
}
std::string TorchAPIOptionsParam::var() const {
  return name;
}
std::string TorchAPIOptionsParam::initializer() const {
  assert(false);
}

std::vector<std::string> TorchAPIOptionsParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& param: ctor_params)
    concat(arg_setup, param->gen_arg_setup());
  for (auto& param: member_params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchAPIOptionsParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: ctor_params)
    concat(hard_ctrs, param->gen_hard_constraint());
  for (auto& param: member_params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchAPIOptionsParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: ctor_params)
    concat(soft_ctrs, param->gen_soft_constraint());
  for (auto& param: member_params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchAPIOptionsParam::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& param: ctor_params)
    concat(ignore_conds, param->gen_input_pass_condition());
  for (auto& param: member_params)
    concat(ignore_conds, param->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TorchAPIOptionsParam::gen_arg_initialization() const {
  std::vector<std::string> preparation_str;

  for (auto& param: ctor_params)
    concat(preparation_str, param->gen_arg_initialization());
  for (auto& param: member_params)
    concat(preparation_str, param->gen_arg_initialization());
  concat(preparation_str, gen_api_options_init());
  preparation_str.back() = preparation_str.back() + newline;
  return preparation_str;
}

bool TorchAPIOptionsParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_APIOptions;
}

std::vector<std::string> TorchAPIOptionsParam::gen_member_param_set() const {
  std::vector<std::string> member_param_set;
  for (auto& param: member_params)
    member_param_set.push_back("." + param->get_name() + bracket(param->expr()));
  return member_param_set;
}

std::vector<std::string> TorchAPIOptionsParam::gen_api_options_init() const {
  std::vector<std::string> api_options_init;

  bool is_sole_variant_ctor =
    ctor_params.size() == 1 && isa<TorchVariantParam>(ctor_params[0].get());

  if (is_sole_variant_ctor) {
    TorchVariantParam* sole_variant_ctor =
      dyn_cast<TorchVariantParam>(ctor_params[0].get());
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
