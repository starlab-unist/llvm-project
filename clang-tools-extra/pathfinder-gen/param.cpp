#include "param.h"

// Should be consistent with
// <TORCH_HOME>/test/cpp/fuzzing/fuzzer_util.h
const size_t MAX_RANK = 5;
const size_t MAX_VECTOR_SIZE = 6;
const size_t MAX_ARRAYREF_SIZE = 6;

static FuzzTargetType ftt = FTT_Basic;
void set_fuzz_target_type(FuzzTargetType ftt_) {
  ftt = ftt_;
}
FuzzTargetType get_fuzz_target_type() {
  return ftt;
}

std::string setup_var(std::string param_name) {
  return symbolic_int_var + sq_quoted(param_name);
}
std::string callback_var(std::string param_name) {
  return callback_input_var + sq_quoted(param_name);
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


TorchIntParam::TorchIntParam(std::string name_, std::string range_)
  : TorchParam(TPK_Int, name_), range(range_) {}
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
  return range;
}
std::string TorchIntParam::initializer() const {
  return  bracket(type()) + bracket(callback_var(name));
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


TorchSymIntParam::TorchSymIntParam(std::string name_)
  : TorchParam(TPK_SymInt, name_)
{
  intparam = std::make_unique<TorchIntParam>(name + "_int", "long");
}
void TorchSymIntParam::set_default(Expr* default_expr) {
  intparam->set_default(default_expr);
}
void TorchSymIntParam::set_default(int value) {
  intparam->set_default(value);
}

std::string TorchSymIntParam::type() const {
  return "c10::SymInt";
}
std::string TorchSymIntParam::initializer() const {
  return  type() + bracket(intparam->expr());
}

std::vector<std::string> TorchSymIntParam::gen_arg_setup() const {
  return intparam->gen_arg_setup();
}
std::vector<std::string> TorchSymIntParam::gen_soft_constraint() const {
  return intparam->gen_soft_constraint();
}

bool TorchSymIntParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_SymInt;
}


TorchUnsignedIntParam::TorchUnsignedIntParam(std::string name_): TorchParam(TPK_UnsignedInt, name_) {}
void TorchUnsignedIntParam::set_default(Expr* default_expr) {
  if (default_expr == nullptr)
    return;

  if (const auto* il = dyn_cast<IntegerLiteral>(default_expr)) {
    assert(il != nullptr);
    unsigned long val = il->getValue().getZExtValue();
    default_value = val;
  }
}
void TorchUnsignedIntParam::set_default(unsigned int value) {
  default_value = value;
}

std::string TorchUnsignedIntParam::type() const {
  return "unsigned long";
}
std::string TorchUnsignedIntParam::initializer() const {
  return callback_var(name);
}

std::vector<std::string> TorchUnsignedIntParam::gen_arg_setup() const {
  return { "PathFinderIntArg" + bracket(quoted(name)) + semicolon };
}
std::vector<std::string> TorchUnsignedIntParam::gen_hard_constraint() const {
  return { setup_var(name) + gte + std::to_string(0) };
}
std::vector<std::string> TorchUnsignedIntParam::gen_soft_constraint() const {
  if (default_value.hasValue())
    return {setup_var(name) + gte + std::to_string(default_value.getValue())};
  else
    return {};
}

bool TorchUnsignedIntParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_UnsignedInt;
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
  return bracket(type()) + bracket(callback_var(name));
}

bool TorchBoolParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Bool;
}


TorchStringParam::TorchStringParam(std::string name_, bool is_view_)
  : TorchBoundedParam(TPK_String, name_, "string_dict().size()"), is_view(is_view_) {}

std::string TorchStringParam::type() const {
  if (is_view)
    return "c10::string_view";

  return "std::string";
}
std::string TorchStringParam::initializer() const {
  return "get_string" + bracket(callback_var(name));
}

bool TorchStringParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_String;
}


TorchBFloatParam::TorchBFloatParam(std::string name_)
  : TorchBoundedParam(TPK_Float, name_, "bfloat_dict().size()") {}

std::string TorchBFloatParam::type() const {
  return "__bf16";
}
std::string TorchBFloatParam::initializer() const {
  return "get_bfloat" + bracket(callback_var(name));
}

bool TorchBFloatParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_BFloat;
}


TorchHalfParam::TorchHalfParam(std::string name_)
  : TorchBoundedParam(TPK_Float, name_, "half_dict().size()") {}

std::string TorchHalfParam::type() const {
  return "__fp16";
}
std::string TorchHalfParam::initializer() const {
  return "get_half" + bracket(callback_var(name));
}

bool TorchHalfParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Half;
}


TorchFloatParam::TorchFloatParam(std::string name_)
  : TorchBoundedParam(TPK_Float, name_, "float_dict().size()") {}

std::string TorchFloatParam::type() const {
  return "float";
}
std::string TorchFloatParam::initializer() const {
  return "get_float" + bracket(callback_var(name));
}

bool TorchFloatParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Float;
}


TorchDoubleParam::TorchDoubleParam(std::string name_)
  : TorchBoundedParam(TPK_Float, name_, "double_dict().size()") {}

std::string TorchDoubleParam::type() const {
  return "double";
}
std::string TorchDoubleParam::initializer() const {
  return "get_double" + bracket(callback_var(name));
}

bool TorchDoubleParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Double;
}


TorchMemoryFormatParam::TorchMemoryFormatParam(std::string name_)
  : TorchBoundedParam(TPK_MemoryFormat, name_, "memory_format_dict().size()") {}

std::string TorchMemoryFormatParam::type() const {
  return "c10::MemoryFormat";
}
std::string TorchMemoryFormatParam::initializer() const {
  return "get_memory_format" + bracket(callback_var(name));
}

bool TorchMemoryFormatParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_MemoryFormat;
}


TorchLayoutParam::TorchLayoutParam(std::string name_)
  : TorchBoundedParam(TPK_Layout, name_, "layout_dict().size()") {}

std::string TorchLayoutParam::type() const {
  return "c10::Layout";
}
std::string TorchLayoutParam::initializer() const {
  return "get_layout" + bracket(callback_var(name));
}

bool TorchLayoutParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Layout;
}


TorchDeviceParam::TorchDeviceParam(std::string name_)
  : TorchBoundedParam(TPK_Device, name_, "device_dict().size()") {}

std::string TorchDeviceParam::type() const {
  return "c10::Device";
}
std::string TorchDeviceParam::initializer() const {
  return "get_device" + bracket(callback_var(name));
}

bool TorchDeviceParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Device;
}


TorchBasicDtypeParam::TorchBasicDtypeParam(std::string name_)
  : TorchBoundedParam(TPK_BasicDtype, name_, "dtype_dict().size()") {}

std::string TorchBasicDtypeParam::type() const {
  return "c10::ScalarType";
}
std::string TorchBasicDtypeParam::initializer() const {
  return "get_dtype" + bracket(callback_var(name));
}

bool TorchBasicDtypeParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_BasicDtype;
}


TorchScalarDtypeParam::TorchScalarDtypeParam(std::string name_)
  : TorchBoundedParam(TPK_ScalarDtype, name_, "scalar_dtype_dict().size()") {}

std::string TorchScalarDtypeParam::type() const {
  return "c10::ScalarType";
}
std::string TorchScalarDtypeParam::initializer() const {
  return "get_scalar_dtype" + bracket(callback_var(name));
}

bool TorchScalarDtypeParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_ScalarDtype;
}


TorchSparseDtypeParam::TorchSparseDtypeParam(std::string name_)
  : TorchBoundedParam(TPK_SparseDtype, name_, "sparse_dtype_dict().size()") {}

std::string TorchSparseDtypeParam::type() const {
  return "c10::ScalarType";
}
std::string TorchSparseDtypeParam::initializer() const {
  return "get_sparse_dtype" + bracket(callback_var(name));
}

bool TorchSparseDtypeParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_SparseDtype;
}


TorchDtypeParam::TorchDtypeParam(std::string name_): TorchParam(TPK_Dtype, name_) {
  basic = std::make_unique<TorchBasicDtypeParam>(name_);
  sparse = std::make_unique<TorchSparseDtypeParam>(name_);
}

std::string TorchDtypeParam::type() const {
  return "c10::ScalarType";
}
std::string TorchDtypeParam::initializer() const {
  if (get_fuzz_target_type() == FTT_Basic) {
    return basic->initializer();
  } else if (get_fuzz_target_type() == FTT_Sparse) {
    return sparse->initializer();
  } else {
    assert(false);
  }
}

std::vector<std::string> TorchDtypeParam::gen_arg_setup() const {
  if (get_fuzz_target_type() == FTT_Basic) {
    return basic->gen_arg_setup();
  } else if (get_fuzz_target_type() == FTT_Sparse) {
    return sparse->gen_arg_setup();
  } else {
    assert(false);
  }
}
void TorchDtypeParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  if (get_fuzz_target_type() == FTT_Basic) {
    return basic->resolve_name_conflict(names_seen);
  } else if (get_fuzz_target_type() == FTT_Sparse) {
    return sparse->resolve_name_conflict(names_seen);
  } else {
    assert(false);
  }
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
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  concat(arg_initialization, gen_typedef());
  concat(arg_initialization, gen_vector());
  arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}
void TorchVariantParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TorchParam::resolve_name_conflict(names_seen);
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}

bool TorchVariantParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Variant;
}

std::vector<std::string> TorchVariantParam::gen_typedef() const {
  std::vector<std::string> typedef_str;
  typedef_str.push_back("typedef");
  typedef_str.push_back("  std::variant<");
  for (size_t i = 0; i < params.size(); i++) {
    std::string param_type = "    " + params[i]->type();
    if (i != params.size() - 1)
      param_type += comma;
    typedef_str.push_back(param_type);
  }
  typedef_str.push_back("  > " + type() + semicolon);
  return typedef_str;
}
std::vector<std::string> TorchVariantParam::gen_vector() const {
  std::vector<std::string> vector_str;
  vector_str.push_back("std::vector<" + type() + "> " + name + " = {");
  for (size_t i = 0; i < params.size(); i++) {
    std::string param_expr = "  " + params[i]->expr();
    if (i != params.size() - 1)
      param_expr += comma;
    vector_str.push_back(param_expr);
  }
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


TorchUnfixedArrayParam::TorchUnfixedArrayParam(
  TorchParamKind kind_,
  std::string name_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchParam(kind_, name_)
{
  assert(TPK_UnfixedArray_First <= get_kind() && get_kind() <= TPK_UnfixedArray_Last);
  params = std::move(params_);
  size = std::make_unique<TorchBoundedIntParam>(name + "_size", params.size() + 1);
}
TorchUnfixedArrayParam::TorchUnfixedArrayParam(
  TorchParamKind kind_,
  std::string name_,
  std::string base_type_str_)
  : TorchParam(kind_, name_), base_type_str(base_type_str_)
{
  assert(TPK_UnfixedArray_First <= get_kind() && get_kind() <= TPK_UnfixedArray_Last);
}

std::string TorchUnfixedArrayParam::var() const {
  //if (!stable())
  //  return "";

  return name;
}

std::vector<std::string> TorchUnfixedArrayParam::gen_arg_setup() const {
  if (!stable())
    return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, size->gen_arg_setup());
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchUnfixedArrayParam::gen_hard_constraint() const {
  if (!stable())
    return {};

  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchUnfixedArrayParam::gen_soft_constraint() const {
  if (!stable())
    return {};

  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchUnfixedArrayParam::gen_arg_initialization() const {
  //if (!stable())
  //  return {};

  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TorchUnfixedArrayParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  //if (!stable())
  //  return;

  TorchParam::resolve_name_conflict(names_seen);
  if (stable()) {
    size->resolve_name_conflict(names_seen);
    for (auto& param: params)
      param->resolve_name_conflict(names_seen);
  }
}

bool TorchUnfixedArrayParam::stable() const {
  return !params.empty(); //&& params[0]->stable();
}
std::string TorchUnfixedArrayParam::base_type() const {
  if (!stable())
    return base_type_str;

  return params[0]->type();
}


TorchVectorParam::TorchVectorParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchUnfixedArrayParam(TPK_Vector, name_, std::move(params_)) {}
TorchVectorParam::TorchVectorParam(std::string name_, std::string base_type_str_)
  : TorchUnfixedArrayParam(TPK_Vector, name_, base_type_str_) {}

std::string TorchVectorParam::type() const {
  return "std::vector<" + TorchUnfixedArrayParam::base_type() + ">";
}
std::string TorchVectorParam::initializer() const {
  if (!stable())
    return type() + "({})";

  return "vector_init<" + params[0]->type() + ">" + bracket(size->expr() + comma + to_string(params));
}

bool TorchVectorParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Vector;
}


/* TorchArrayRefParam::TorchArrayRefParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchUnfixedArrayParam(TPK_ArrayRef, name_, std::move(params_))
{ assert(params.size() == MAX_ARRAYREF_SIZE); }

std::string TorchArrayRefParam::type() const {
  return "c10::ArrayRef<" + params[0]->type() + ">";
}
std::string TorchArrayRefParam::initializer() const {
  return "arrayref_init<" + params[0]->type() + ">" + bracket(size->expr() + comma + to_string(params));
}

bool TorchArrayRefParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_ArrayRef;
} */

TorchArrayRefParam::TorchArrayRefParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchParam(TPK_ArrayRef, name_)
{
  vec = std::make_unique<TorchVectorParam>(name + "_vec", std::move(params_));
}
TorchArrayRefParam::TorchArrayRefParam(std::string name_, std::string base_type_str_)
  : TorchParam(TPK_ArrayRef, name_)
{
  vec = std::make_unique<TorchVectorParam>(name + "_vec", base_type_str_);
}

std::string TorchArrayRefParam::type() const {
  return "c10::ArrayRef<" + vec->base_type() + ">";
}
std::string TorchArrayRefParam::var() const {
  //if (!stable())
  //  return "";

  return name;
}
std::string TorchArrayRefParam::initializer() const {
  //if (!stable())
  //  return type() + "({})";

  return type() + bracket(vec->expr());
}

std::vector<std::string> TorchArrayRefParam::gen_arg_setup() const {
  return vec->gen_arg_setup();
}
std::vector<std::string> TorchArrayRefParam::gen_hard_constraint() const {
  return vec->gen_hard_constraint();
}
std::vector<std::string> TorchArrayRefParam::gen_soft_constraint() const {
  return vec->gen_soft_constraint();
}
std::vector<std::string> TorchArrayRefParam::gen_arg_initialization() const {
  //if (!stable())
  //  return {};

  std::vector<std::string> arg_initialization;
  concat(arg_initialization, vec->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TorchArrayRefParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  //if (!stable())
  //  return;

  TorchParam::resolve_name_conflict(names_seen);
  vec->resolve_name_conflict(names_seen);
}

/* bool TorchArrayRefParam::stable() const {
  return vec->stable();
} */

bool TorchArrayRefParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_ArrayRef;
}

TorchFixedArrayParam::TorchFixedArrayParam(
  TorchParamKind kind_,
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchParam(kind_, name_)
{
  assert(TPK_FixedArray_First <= get_kind() && get_kind() <= TPK_FixedArray_Last);
  size = size_;
  params = std::move(params_);
  assert(params.size() == size);
}

std::string TorchFixedArrayParam::var() const {
  return name;
}

std::vector<std::string> TorchFixedArrayParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchFixedArrayParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchFixedArrayParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchFixedArrayParam::gen_arg_initialization() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TorchFixedArrayParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TorchParam::resolve_name_conflict(names_seen);
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}


TorchExpandingArrayParam::TorchExpandingArrayParam(
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchFixedArrayParam(TPK_ExpandingArray, name_, size_, std::move(params_)) {}
void TorchExpandingArrayParam::set_default(Expr* default_expr) {
  for (auto&& param: params)
    param->set_default(default_expr);
}

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
  : TorchFixedArrayParam(TPK_ExpandingArrayWithOptionalElem, name_, size_, std::move(params_))
{
  for (auto& param: params)
    assert(isa<TorchOptionalParam>(param.get()));
}
void TorchExpandingArrayWithOptionalElemParam::set_default(Expr* default_expr) {
  for (auto&& param: params)
    param->set_default(default_expr);
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


TorchTupleParam::TorchTupleParam(
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchFixedArrayParam(TPK_Tuple, name_, size_, std::move(params_)) {}

std::string TorchTupleParam::type() const {
  std::string type_str = "std::tuple<";
  for (size_t i = 0; i < params.size(); i++) {
    type_str += params[i]->type();
    if (i != params.size() - 1)
      type_str += comma;
  }
  type_str += ">";

  return type_str;
}
std::string TorchTupleParam::initializer() const {
  return type() + bracket(to_string(params));
}

bool TorchTupleParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Tuple;
}

TorchPairParam::TorchPairParam(
  std::string name_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchFixedArrayParam(TPK_Pair, name_, 2, std::move(params_)) {}

std::string TorchPairParam::type() const {
  return "std::pair<" + params[0]->type() + comma + params[1]->type() + ">";
}
std::string TorchPairParam::initializer() const {
  return type() + bracket(to_string(params));
}

bool TorchPairParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Pair;
}

TorchTensorParam::TorchTensorParam(std::string name_): TorchParam(TPK_Tensor, name_) {
  dtype = std::make_unique<TorchDtypeParam>(name + "_dtype");
  layout = std::make_unique<TorchLayoutParam>(name + "_layout");
  rank = std::make_unique<TorchBoundedIntParam>(name + "_rank", MAX_RANK + 1);
  for (size_t i = 0; i < MAX_RANK; i++)
    dims.push_back(std::make_unique<TorchIntParam>(name + "_" + std::to_string(i), "long"));
  for (auto&& dim: dims)
    dim->set_default(1);
}

std::string TorchTensorParam::type() const { return "torch::Tensor"; }
std::string TorchTensorParam::var() const { return name; }
std::string TorchTensorParam::initializer() const {
  std::string layout_expr =
    get_fuzz_target_type() == FTT_Sparse ? layout->expr() + comma : "";
  std::string args =
    dtype->expr() + comma + layout_expr + rank->expr() + comma + to_string(dims);
  return "torch_tensor" + bracket(args);
}

std::vector<std::string> TorchTensorParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, dtype->gen_arg_setup());
  if (get_fuzz_target_type() == FTT_Sparse)
    concat(arg_setup, layout->gen_arg_setup());
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
void TorchTensorParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TorchParam::resolve_name_conflict(names_seen);
  dtype->resolve_name_conflict(names_seen);
  if (get_fuzz_target_type() == FTT_Sparse)
    layout->resolve_name_conflict(names_seen);
  rank->resolve_name_conflict(names_seen);
  for (auto& dim: dims)
    dim->resolve_name_conflict(names_seen);
}

bool TorchTensorParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Tensor;
}

TorchScalarParam::TorchScalarParam(std::string name_): TorchParam(TPK_Scalar, name_) {
  dtype = std::make_unique<TorchScalarDtypeParam>(name + "_dtype");
  intValue = std::make_unique<TorchIntParam>(name + "_int", "int");
  uintValue = std::make_unique<TorchUnsignedIntParam>(name + "_uint");
  bfloatValue = std::make_unique<TorchBFloatParam>(name + "_bfloat");
  halfValue = std::make_unique<TorchHalfParam>(name + "_half");
  floatValue = std::make_unique<TorchFloatParam>(name + "_float");
  doubleValue = std::make_unique<TorchDoubleParam>(name + "_double");
  realValue32 = std::make_unique<TorchHalfParam>(name + "_real32");
  imaginaryValue32 = std::make_unique<TorchHalfParam>(name + "_imag32");
  realValue64 = std::make_unique<TorchFloatParam>(name + "_real64");
  imaginaryValue64 = std::make_unique<TorchFloatParam>(name + "_imag64");
  realValue128 = std::make_unique<TorchDoubleParam>(name + "_real128");
  imaginaryValue128 = std::make_unique<TorchDoubleParam>(name + "_imag128");
  boolValue = std::make_unique<TorchBoolParam>(name + "_bool");
  params.push_back(dtype.get());
  params.push_back(intValue.get());
  params.push_back(uintValue.get());
  params.push_back(bfloatValue.get());
  params.push_back(halfValue.get());
  params.push_back(floatValue.get());
  params.push_back(doubleValue.get());
  params.push_back(realValue32.get());
  params.push_back(imaginaryValue32.get());
  params.push_back(realValue64.get());
  params.push_back(imaginaryValue64.get());
  params.push_back(realValue128.get());
  params.push_back(imaginaryValue128.get());
  params.push_back(boolValue.get());
}

std::string TorchScalarParam::type() const { return "c10::Scalar"; }
std::string TorchScalarParam::var() const { return name; }
std::string TorchScalarParam::initializer() const {
  std::string candidates;
  for (size_t i = 0; i < params.size(); i++) {
    candidates += params[i]->expr();
    if (i != params.size() - 1)
      candidates += comma;
  }
  return "torch_scalar" + bracket(candidates);
}

std::vector<std::string> TorchScalarParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchScalarParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchScalarParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchScalarParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}
void TorchScalarParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TorchParam::resolve_name_conflict(names_seen);
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}

bool TorchScalarParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_Scalar;
}

TorchOptionalParam::TorchOptionalParam(std::string name_, std::unique_ptr<TorchParam> param_)
  : TorchParam(TPK_Optional, name_)
{
  has_value = std::make_unique<TorchBoolParam>(name + "_hasValue");
  param = std::move(param_);
}
TorchOptionalParam::TorchOptionalParam(std::string name_, std::string base_type_str_)
  : TorchParam(TPK_Optional, name_), base_type_str(base_type_str_) {}
void TorchOptionalParam::set_default(Expr* default_expr) {
  if (!stable())
    return;
  
  param->set_default(default_expr);
}

std::string TorchOptionalParam::type() const {
  return "c10::optional<" + base_type() + ">";
}
std::string TorchOptionalParam::var() const {
  //if (!stable())
  //  return "";

  return name;
}
std::string TorchOptionalParam::initializer() const {
  if (!stable())
    //return "c10::nullopt";
    return type() + bracket("c10::nullopt");

  return has_value->expr() + " ? " + type() + bracket(param->expr()) +  " : " + "c10::nullopt";
}

std::vector<std::string> TorchOptionalParam::gen_arg_setup() const {
  if (!stable())
    return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, has_value->gen_arg_setup());
  concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchOptionalParam::gen_hard_constraint() const {
  if (!stable())
    return {};

  return param->gen_hard_constraint();
}
std::vector<std::string> TorchOptionalParam::gen_soft_constraint() const {
  if (!stable())
    return {};

  return param->gen_soft_constraint();
}
std::vector<std::string> TorchOptionalParam::gen_input_pass_condition() const {
  if (!stable())
    return {};

  return param->gen_input_pass_condition();
}
std::vector<std::string> TorchOptionalParam::gen_arg_initialization() const {
  //if (!stable())
  //  return {};

  std::vector<std::string> arg_initialization;
  if (stable())
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TorchOptionalParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  //if (!stable())
  //  return;

  TorchParam::resolve_name_conflict(names_seen);
  if (stable()) {
    has_value->resolve_name_conflict(names_seen);
    param->resolve_name_conflict(names_seen);
  }
}

bool TorchOptionalParam::stable() const {
  return param != nullptr; //&& param->stable();
}

std::string TorchOptionalParam::base_type() const {
  if (!stable())
    return base_type_str;

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
  for (auto& member_param: member_params)
    member_param_setters.push_back(member_param->get_name());
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
  std::vector<std::string> arg_initialization;
  for (auto& param: ctor_params)
    concat(arg_initialization, param->gen_arg_initialization());
  for (auto& param: member_params)
    concat(arg_initialization, param->gen_arg_initialization());
  concat(arg_initialization, gen_api_options_init());
  arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}
void TorchAPIOptionsParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TorchParam::resolve_name_conflict(names_seen);
  for (auto& param: ctor_params)
    param->resolve_name_conflict(names_seen);
  for (auto& param: member_params)
    param->resolve_name_conflict(names_seen);
}

bool TorchAPIOptionsParam::classof(const TorchParam *param) {
  return param->get_kind() == TPK_APIOptions;
}

std::vector<std::string> TorchAPIOptionsParam::gen_member_param_set() const {
  std::vector<std::string> member_param_set;
  assert(member_params.size() == member_param_setters.size());
  for (size_t i = 0; i < member_params.size(); i++)
    member_param_set.push_back("." + member_param_setters[i] + bracket(member_params[i]->expr()));
  return member_param_set;
}

std::vector<std::string> TorchAPIOptionsParam::gen_api_options_init() const {
  std::vector<std::string> api_options_init;

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
  api_options_init.back() = api_options_init.back() + semicolon;
  return api_options_init;
}
