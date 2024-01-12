#include "param.h"

// Should be consistent with
// <TENSORFLOW_HOME>/tensorflow/core/kernels/pathfinder/fuzzer_util.h
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

std::vector<std::string> get_names(const std::vector<std::unique_ptr<TFParam>>& params) {
  std::vector<std::string> names;
  for (auto& param: params)
    names.push_back(param->get_name());
  return names;
}

static bool module_mode;
void set_function_mode() { module_mode = false; }
void set_module_mode() { module_mode = true; }
bool is_module_mode() { return module_mode; }


TFScopeParam::TFScopeParam(): TFParam(TFPK_Scope, scope_var) {}

std::string TFScopeParam::type() const {
  return "Scope";
}
std::string TFScopeParam::var() const {
  return name;
}
std::string TFScopeParam::initializer() const {
  return "Scope::DisabledShapeInferenceScope()";
}

std::vector<std::string> TFScopeParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}
void TFScopeParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  return;
}


TFIntParam::TFIntParam(std::string name_, std::string range_)
  : TFParam(TFPK_Int, name_), range(range_) {}
void TFIntParam::set_default(Expr* default_expr) {
  if (default_expr == nullptr)
    return;

  if (const auto* il = dyn_cast<IntegerLiteral>(default_expr)) {
    assert(il != nullptr);
    unsigned long val = il->getValue().getZExtValue();
    default_value = val;
  }
}
void TFIntParam::set_default(int value) {
  default_value = value;
}

std::string TFIntParam::type() const {
  return range;
}
std::string TFIntParam::initializer() const {
  return  bracket(type()) + bracket(callback_var(name));
}

std::vector<std::string> TFIntParam::gen_arg_setup() const {
  return { "PathFinderIntArg" + bracket(quoted(name)) + semicolon  };
}
std::vector<std::string> TFIntParam::gen_soft_constraint() const {
  int min =
    default_value.has_value() ?
    default_value.value() : 0;
    //(is_module_mode() ? 1 : 0);
  return { setup_var(name) + gte + std::to_string(min) };
}

bool TFIntParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Int;
}


TFUnsignedIntParam::TFUnsignedIntParam(std::string name_): TFParam(TFPK_UnsignedInt, name_) {}
void TFUnsignedIntParam::set_default(Expr* default_expr) {
  if (default_expr == nullptr)
    return;

  if (const auto* il = dyn_cast<IntegerLiteral>(default_expr)) {
    assert(il != nullptr);
    unsigned long val = il->getValue().getZExtValue();
    default_value = val;
  }
}
void TFUnsignedIntParam::set_default(unsigned int value) {
  default_value = value;
}

std::string TFUnsignedIntParam::type() const {
  return "unsigned long";
}
std::string TFUnsignedIntParam::initializer() const {
  return callback_var(name);
}

std::vector<std::string> TFUnsignedIntParam::gen_arg_setup() const {
  return { "PathFinderIntArg" + bracket(quoted(name)) + semicolon };
}
std::vector<std::string> TFUnsignedIntParam::gen_hard_constraint() const {
  return { setup_var(name) + gte + std::to_string(0) };
}
std::vector<std::string> TFUnsignedIntParam::gen_soft_constraint() const {
  if (default_value.has_value())
    return {setup_var(name) + gte + std::to_string(default_value.value())};
  else
    return {};
}

bool TFUnsignedIntParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_UnsignedInt;
}


TFBoundedParam::TFBoundedParam(TFParamKind kind_, std::string name_, const std::vector<std::string>& value_names_)
  : TFParam(kind_, name_)
{
  assert(TFPK_Bounded_First <= get_kind() && get_kind() <= TFPK_Bounded_Last);
  for (auto value_name_: value_names_)
    value_names.push_back(quoted(value_name_));
}
TFBoundedParam::TFBoundedParam(TFParamKind kind_, std::string name_, size_t size_)
  : TFParam(kind_, name_)
{
  assert(TFPK_Bounded_First <= get_kind() && get_kind() <= TFPK_Bounded_Last);
  size = size_;
}
TFBoundedParam::TFBoundedParam(TFParamKind kind_, std::string name_, const std::string& value_list_var_)
  : TFParam(kind_, name_)
{
  assert(TFPK_Bounded_First <= get_kind() && get_kind() <= TFPK_Bounded_Last);
  value_list_var = value_list_var_;
}

std::vector<std::string> TFBoundedParam::gen_arg_setup() const {
  std::string setup_args;
  if (!value_names.empty())
    setup_args = quoted(name) + comma + curly(join_strs(value_names));
  else if (size.has_value())
    setup_args = quoted(name) + comma + std::to_string(size.value());
  else
    setup_args = quoted(name) + comma + value_list_var;
    
  return {"PathFinderEnumArg" + bracket(setup_args) + semicolon};
}


TFBoundedIntParam::TFBoundedIntParam(std::string name_, size_t size_)
  : TFBoundedParam(TFPK_BoundedInt, name_, size_) {}

std::string TFBoundedIntParam::type() const {
  return "size_t";
}
std::string TFBoundedIntParam::initializer() const {
  return bracket(type()) + bracket(callback_var(name));
}

bool TFBoundedIntParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_BoundedInt;
}


TFBoolParam::TFBoolParam(std::string name_)
  : TFBoundedParam(TFPK_Bool, name_, std::vector<std::string>({"false", "true"})) {}

std::string TFBoolParam::type() const {
  return "bool";
}
std::string TFBoolParam::initializer() const {
  return bracket(type()) + bracket(callback_var(name));
}

bool TFBoolParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Bool;
}


TFStringPieceParam::TFStringPieceParam(std::string name_)
  : TFBoundedParam(TFPK_StringPiece, name_, "string_dict().size()") {}

std::string TFStringPieceParam::type() const {
  return "StringPiece";
}
std::string TFStringPieceParam::initializer() const {
  return "get_string" + bracket(callback_var(name));
}

bool TFStringPieceParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_StringPiece;
}


TFDataFormatParam::TFDataFormatParam(std::string name_)
  : TFBoundedParam(TFPK_DataFormat, name_, "dataformat_dict().size()") {}

std::string TFDataFormatParam::type() const {
  return "StringPiece";
}
std::string TFDataFormatParam::initializer() const {
  return "get_dataformat" + bracket(callback_var(name));
}

bool TFDataFormatParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_DataFormat;
}


TFPaddingParam::TFPaddingParam(std::string name_)
  : TFBoundedParam(TFPK_Padding, name_, "padding_dict().size()") {}

std::string TFPaddingParam::type() const {
  return "StringPiece";
}
std::string TFPaddingParam::initializer() const {
  return "get_padding" + bracket(callback_var(name));
}

bool TFPaddingParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Padding;
}


TFBFloatParam::TFBFloatParam(std::string name_)
  : TFBoundedParam(TFPK_Float, name_, "bfloat_dict().size()") {}

std::string TFBFloatParam::type() const {
  return "__bf16";
}
std::string TFBFloatParam::initializer() const {
  return "get_bfloat" + bracket(callback_var(name));
}

bool TFBFloatParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_BFloat;
}


TFHalfParam::TFHalfParam(std::string name_)
  : TFBoundedParam(TFPK_Float, name_, "half_dict().size()") {}

std::string TFHalfParam::type() const {
  return "__fp16";
}
std::string TFHalfParam::initializer() const {
  return "get_half" + bracket(callback_var(name));
}

bool TFHalfParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Half;
}


TFFloatParam::TFFloatParam(std::string name_)
  : TFBoundedParam(TFPK_Float, name_, "float_dict().size()") {}

std::string TFFloatParam::type() const {
  return "float";
}
std::string TFFloatParam::initializer() const {
  return "get_float" + bracket(callback_var(name));
}

bool TFFloatParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Float;
}


TFDoubleParam::TFDoubleParam(std::string name_)
  : TFBoundedParam(TFPK_Double, name_, "double_dict().size()") {}

std::string TFDoubleParam::type() const {
  return "double";
}
std::string TFDoubleParam::initializer() const {
  return "get_double" + bracket(callback_var(name));
}

bool TFDoubleParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Double;
}


TFBasicDtypeParam::TFBasicDtypeParam(std::string name_)
  : TFBoundedParam(TFPK_BasicDtype, name_, "dtype_dict().size()") {}

std::string TFBasicDtypeParam::type() const {
  return "DataType";
}
std::string TFBasicDtypeParam::initializer() const {
  return "get_dtype" + bracket(callback_var(name));
}

bool TFBasicDtypeParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_BasicDtype;
}


TFExtendedDtypeParam::TFExtendedDtypeParam(std::string name_)
  : TFBoundedParam(TFPK_ExtendedDtype, name_, "qdtype_dict().size()") {}

std::string TFExtendedDtypeParam::type() const {
  return "DataType";
}
std::string TFExtendedDtypeParam::initializer() const {
  return "get_qdtype" + bracket(callback_var(name));
}

bool TFExtendedDtypeParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_ExtendedDtype;
}


TFDtypeParam::TFDtypeParam(std::string name_): TFParam(TFPK_Dtype, name_) {
  basic = std::make_unique<TFBasicDtypeParam>(name_);
  extended = std::make_unique<TFExtendedDtypeParam>(name_);
}

std::string TFDtypeParam::type() const {
  return "DataType";
}
std::string TFDtypeParam::initializer() const {
  return
    get_fuzz_target_type() == FTT_Quantization ?
    extended->initializer() :
    basic->initializer();
}

std::vector<std::string> TFDtypeParam::gen_arg_setup() const {
  return
    get_fuzz_target_type() == FTT_Quantization ?
    extended->gen_arg_setup() :
    basic->gen_arg_setup();
}
void TFDtypeParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  if (get_fuzz_target_type() == FTT_Quantization) {
    extended->resolve_name_conflict(names_seen);
  } else {
    basic->resolve_name_conflict(names_seen);
  }
}

bool TFDtypeParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Dtype;
}


TFVariantParam::TFVariantParam(
  std::string name_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFBoundedParam(TFPK_Variant, name_, get_names(params_))
{
  params = std::move(params_);
}
void TFVariantParam::set_default(Expr* default_expr) {
  for (auto& param: params)
    param->set_default(default_expr);
}

std::string TFVariantParam::type() const {
  return name + "_t";
}
std::string TFVariantParam::initializer() const {
  return name + square(callback_var(name));
}

std::vector<std::string> TFVariantParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, TFBoundedParam::gen_arg_setup());
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFVariantParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFVariantParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFVariantParam::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& param: params)
    concat(ignore_conds, param->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TFVariantParam::gen_arg_initialization() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  concat(arg_initialization, gen_typedef());
  concat(arg_initialization, gen_vector());
  arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}
void TFVariantParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFParam::resolve_name_conflict(names_seen);
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}

bool TFVariantParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Variant;
}

std::vector<std::string> TFVariantParam::gen_typedef() const {
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
std::vector<std::string> TFVariantParam::gen_vector() const {
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


TFUnfixedArrayParam::TFUnfixedArrayParam(
  TFParamKind kind_,
  std::string name_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFParam(kind_, name_)
{
  assert(TFPK_UnfixedArray_First <= get_kind() && get_kind() <= TFPK_UnfixedArray_Last);
  params = std::move(params_);
  size = std::make_unique<TFBoundedIntParam>(name + "_size", params.size() + 1);
}
TFUnfixedArrayParam::TFUnfixedArrayParam(
  TFParamKind kind_,
  std::string name_,
  std::string base_type_str_)
  : TFParam(kind_, name_), base_type_str(base_type_str_)
{
  assert(TFPK_UnfixedArray_First <= get_kind() && get_kind() <= TFPK_UnfixedArray_Last);
}

std::string TFUnfixedArrayParam::var() const {
  //if (!stable())
  //  return "";

  return name;
}

std::vector<std::string> TFUnfixedArrayParam::gen_arg_setup() const {
  if (!stable())
    return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, size->gen_arg_setup());
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFUnfixedArrayParam::gen_hard_constraint() const {
  if (!stable())
    return {};

  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFUnfixedArrayParam::gen_soft_constraint() const {
  if (!stable())
    return {};

  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFUnfixedArrayParam::gen_arg_initialization() const {
  //if (!stable())
  //  return {};

  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TFUnfixedArrayParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  //if (!stable())
  //  return;

  TFParam::resolve_name_conflict(names_seen);
  if (stable()) {
    size->resolve_name_conflict(names_seen);
    for (auto& param: params)
      param->resolve_name_conflict(names_seen);
  }
}

bool TFUnfixedArrayParam::stable() const {
  return !params.empty(); //&& params[0]->stable();
}
std::string TFUnfixedArrayParam::base_type() const {
  if (!stable())
    return base_type_str;

  return params[0]->type();
}


TFVectorParam::TFVectorParam(std::string name_, std::vector<std::unique_ptr<TFParam>> params_)
  : TFUnfixedArrayParam(TFPK_Vector, name_, std::move(params_)) {}
TFVectorParam::TFVectorParam(std::string name_, std::string base_type_str_)
  : TFUnfixedArrayParam(TFPK_Vector, name_, base_type_str_) {}

std::string TFVectorParam::type() const {
  return "std::vector<" + TFUnfixedArrayParam::base_type() + ">";
}
std::string TFVectorParam::initializer() const {
  if (!stable())
    return type() + "({})";

  return "vector_init<" + params[0]->type() + ">" + bracket(size->expr() + comma + to_string(params));
}

bool TFVectorParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Vector;
}


TFArraySliceParam::TFArraySliceParam(std::string name_, std::vector<std::unique_ptr<TFParam>> params_)
  : TFParam(TFPK_ArraySlice, name_)
{
  vec = std::make_unique<TFVectorParam>(name + "_vec", std::move(params_));
}
TFArraySliceParam::TFArraySliceParam(std::string name_, std::string base_type_str_)
  : TFParam(TFPK_ArraySlice, name_)
{
  vec = std::make_unique<TFVectorParam>(name + "_vec", base_type_str_);
}

std::string TFArraySliceParam::type() const {
  return "gtl::ArraySlice<" + vec->base_type() + ">";
}
std::string TFArraySliceParam::var() const {
  return name;
}
std::string TFArraySliceParam::initializer() const {
  return type() + bracket(vec->expr());
}

std::vector<std::string> TFArraySliceParam::gen_arg_setup() const {
  return vec->gen_arg_setup();
}
std::vector<std::string> TFArraySliceParam::gen_hard_constraint() const {
  return vec->gen_hard_constraint();
}
std::vector<std::string> TFArraySliceParam::gen_soft_constraint() const {
  return vec->gen_soft_constraint();
}
std::vector<std::string> TFArraySliceParam::gen_arg_initialization() const {
  std::vector<std::string> arg_initialization;
  concat(arg_initialization, vec->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TFArraySliceParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFParam::resolve_name_conflict(names_seen);
  vec->resolve_name_conflict(names_seen);
}

bool TFArraySliceParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_ArraySlice;
}


TFFixedArrayParam::TFFixedArrayParam(
  TFParamKind kind_,
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFParam(kind_, name_)
{
  assert(TFPK_FixedArray_First <= get_kind() && get_kind() <= TFPK_FixedArray_Last);
  size = size_;
  params = std::move(params_);
  assert(params.size() == size);
}

std::string TFFixedArrayParam::var() const {
  return name;
}

std::vector<std::string> TFFixedArrayParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFFixedArrayParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& param: params)
    concat(hard_ctrs, param->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFFixedArrayParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& param: params)
    concat(soft_ctrs, param->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFFixedArrayParam::gen_arg_initialization() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TFFixedArrayParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFParam::resolve_name_conflict(names_seen);
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}


TFTupleParam::TFTupleParam(
  std::string name_,
  size_t size_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFFixedArrayParam(TFPK_Tuple, name_, size_, std::move(params_)) {}

std::string TFTupleParam::type() const {
  std::string type_str = "std::tuple<";
  for (size_t i = 0; i < params.size(); i++) {
    type_str += params[i]->type();
    if (i != params.size() - 1)
      type_str += comma;
  }
  type_str += ">";

  return type_str;
}
std::string TFTupleParam::initializer() const {
  return type() + bracket(to_string(params));
}

bool TFTupleParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Tuple;
}

TFPairParam::TFPairParam(
  std::string name_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFFixedArrayParam(TFPK_Pair, name_, 2, std::move(params_)) {}

std::string TFPairParam::type() const {
  return "std::pair<" + params[0]->type() + comma + params[1]->type() + ">";
}
std::string TFPairParam::initializer() const {
  return type() + bracket(to_string(params));
}

bool TFPairParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Pair;
}

TFTensorParam::TFTensorParam(std::string name_): TFParam(TFPK_Tensor, name_) {
  dtype = std::make_unique<TFDtypeParam>(name + "_dtype");
  rank = std::make_unique<TFBoundedIntParam>(name + "_rank", MAX_RANK + 1);
  for (size_t i = 0; i < MAX_RANK; i++)
    dims.push_back(std::make_unique<TFIntParam>(name + "_" + std::to_string(i), "long"));
  for (auto&& dim: dims)
    dim->set_default(1);
}

std::string TFTensorParam::type() const { return "Input"; }
std::string TFTensorParam::var() const { return name; }
std::string TFTensorParam::initializer() const {
  std::string args =
    scope_var + comma + dtype->expr() + comma + rank->expr() + comma + to_string(dims);
  return "tf_tensor" + bracket(args);
}

std::vector<std::string> TFTensorParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, dtype->gen_arg_setup());
  concat(arg_setup, rank->gen_arg_setup());
  for (auto& dim: dims)
    concat(arg_setup, dim->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFTensorParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& dim: dims)
    concat(hard_ctrs, dim->gen_soft_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFTensorParam::gen_input_pass_condition() const {
  return {"is_too_big" + bracket(rank->expr() + comma + to_string(dims))};
}
std::vector<std::string> TFTensorParam::gen_arg_initialization() const {
  return {type() + space + var() + assign + initializer() + semicolon + newline};
}
void TFTensorParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFParam::resolve_name_conflict(names_seen);
  dtype->resolve_name_conflict(names_seen);
  rank->resolve_name_conflict(names_seen);
  for (auto& dim: dims)
    dim->resolve_name_conflict(names_seen);
}

bool TFTensorParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Tensor;
}


/* TFOptionalParam::TFOptionalParam(std::string name_, std::unique_ptr<TFParam> param_)
  : TFParam(TFPK_Optional, name_)
{
  has_value = std::make_unique<TFBoolParam>(name + "_hasValue");
  param = std::move(param_);
}
TFOptionalParam::TFOptionalParam(std::string name_, std::string base_type_str_)
  : TFParam(TFPK_Optional, name_), base_type_str(base_type_str_) {}
void TFOptionalParam::set_default(Expr* default_expr) {
  if (!stable())
    return;
  
  param->set_default(default_expr);
}

std::string TFOptionalParam::type() const {
  return "c10::optional<" + base_type() + ">";
}
std::string TFOptionalParam::var() const {
  //if (!stable())
  //  return "";

  return name;
}
std::string TFOptionalParam::initializer() const {
  if (!stable())
    //return "c10::nullopt";
    return type() + bracket("c10::nullopt");

  return has_value->expr() + " ? " + type() + bracket(param->expr()) +  " : " + "c10::nullopt";
}

std::vector<std::string> TFOptionalParam::gen_arg_setup() const {
  if (!stable())
    return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, has_value->gen_arg_setup());
  concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFOptionalParam::gen_hard_constraint() const {
  if (!stable())
    return {};

  return param->gen_hard_constraint();
}
std::vector<std::string> TFOptionalParam::gen_soft_constraint() const {
  if (!stable())
    return {};

  return param->gen_soft_constraint();
}
std::vector<std::string> TFOptionalParam::gen_input_pass_condition() const {
  if (!stable())
    return {};

  return param->gen_input_pass_condition();
}
std::vector<std::string> TFOptionalParam::gen_arg_initialization() const {
  //if (!stable())
  //  return {};

  std::vector<std::string> arg_initialization;
  if (stable())
    concat(arg_initialization, param->gen_arg_initialization());
  arg_initialization.push_back(type() + space + var() + assign + initializer() + semicolon + newline);
  return arg_initialization;
}
void TFOptionalParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  //if (!stable())
  //  return;

  TFParam::resolve_name_conflict(names_seen);
  if (stable()) {
    has_value->resolve_name_conflict(names_seen);
    param->resolve_name_conflict(names_seen);
  }
}

bool TFOptionalParam::stable() const {
  return param != nullptr; //&& param->stable();
}

std::string TFOptionalParam::base_type() const {
  if (!stable())
    return base_type_str;

  return param->type();
}

bool TFOptionalParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_Optional;
} */


TFAPIAttrsParam::TFAPIAttrsParam(
  std::string name_,
  std::string api_attrs_class_name_,
  std::vector<std::tuple<std::string, std::unique_ptr<TFBoolParam>, std::unique_ptr<TFParam>>> setters_)
  : TFParam(TFPK_APIAttrs, name_)
{
  api_attrs_class_name = api_attrs_class_name_;
  setters = std::move(setters_);
}

std::string TFAPIAttrsParam::type() const {
  return api_attrs_class_name;
}
std::string TFAPIAttrsParam::var() const {
  return name;
}
std::string TFAPIAttrsParam::initializer() const {
  assert(false);
}

std::vector<std::string> TFAPIAttrsParam::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    concat(arg_setup, setter_flag->gen_arg_setup());
    concat(arg_setup, setter_param->gen_arg_setup());
  }
  return arg_setup;
}
std::vector<std::string> TFAPIAttrsParam::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    concat(hard_ctrs, setter_flag->gen_hard_constraint());
    concat(hard_ctrs, setter_param->gen_hard_constraint());
  }
  return hard_ctrs;
}
std::vector<std::string> TFAPIAttrsParam::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    concat(soft_ctrs, setter_flag->gen_soft_constraint());
    concat(soft_ctrs, setter_param->gen_soft_constraint());
  }
  return soft_ctrs;
}
std::vector<std::string> TFAPIAttrsParam::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    concat(ignore_conds, setter_flag->gen_input_pass_condition());
    concat(ignore_conds, setter_param->gen_input_pass_condition());
  }
  return ignore_conds;
}
std::vector<std::string> TFAPIAttrsParam::gen_arg_initialization() const {
  std::vector<std::string> arg_initialization;
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    concat(arg_initialization, setter_flag->gen_arg_initialization());
    concat(arg_initialization, setter_param->gen_arg_initialization());
  }
  concat(arg_initialization, gen_api_attrs_init());
  if (!arg_initialization.empty())
    arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}
void TFAPIAttrsParam::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFParam::resolve_name_conflict(names_seen);
  for (auto& t: setters) {
    //auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    setter_flag->resolve_name_conflict(names_seen);
    setter_param->resolve_name_conflict(names_seen);
  }
}

bool TFAPIAttrsParam::classof(const TFParam *param) {
  return param->get_kind() == TFPK_APIAttrs;
}

std::vector<std::string> TFAPIAttrsParam::gen_api_attrs_init() const {
  std::vector<std::string> api_attrs_init;

  api_attrs_init.push_back("auto " + name + assign + api_attrs_class_name + "();");
  for (auto& t: setters) {
    auto& setter_name = std::get<0>(t);
    auto& setter_flag = std::get<1>(t);
    auto& setter_param = std::get<2>(t);
    api_attrs_init.push_back("if (" + setter_flag->expr() + ")");
    api_attrs_init.push_back("  " + name + assign + name + "." + setter_name + bracket(setter_param->expr()) + semicolon);
  }
  return api_attrs_init;
}
