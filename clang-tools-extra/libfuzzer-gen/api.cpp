#include "api.h"

const std::string tensor_method_self_var = "self";

TorchAPI::TorchAPI(std::string api_name_): api_name(api_name_) {}
std::string TorchAPI::gen_fuzz_target(FuzzTargetType ftt) {
  set_fuzz_target_type(ftt);
  resolve_name_conflict();
  std::vector<std::string> lines;
  concat(lines, header());
  //concat(lines, setup());
  concat(lines, callback());
  if (get_gen_covrunner())
    concat(lines, footer());
  return join_strs(lines, newline);
}
void TorchAPI::resolve_name_conflict() {
  std::set<std::string> names_seen = {callback_input_var};
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}
std::vector<std::string> TorchAPI::arg_setup_code() const {
  std::vector<std::string> arg_setups;
  size_t index = 0;
  for (auto& param: params)
    index = param->gen_arg_setup(arg_setups, index);
  return arg_setups;
}
std::vector<std::string> TorchAPI::hard_constraint_code() const {
  std::vector<std::string> hard_constraint;
  for (auto& param: params) {
    for (auto& hard_ctr: param->gen_hard_constraint()) {
      hard_constraint.push_back("if (!(" + hard_ctr + "))");
      hard_constraint.push_back("  return -1;");
    }
  }
  return hard_constraint;
}
std::vector<std::string> TorchAPI::soft_constraint_code() const {
  std::vector<std::string> soft_constraint;
  for (auto& param: params)
    concat(soft_constraint, "  ", param->gen_soft_constraint(), comma);

  if (soft_constraint.empty())
    return {};

  soft_constraint.insert(soft_constraint.begin(), "PathFinderAddSoftConstraint({");
  soft_constraint.push_back("});");
  return soft_constraint;
}
std::vector<std::string> TorchAPI::input_pass_condition_code() const {
  std::vector<std::string> input_pass_condition;
  for (auto& param: params) {
    for (auto& input_pass_cond: param->gen_input_pass_condition()) {
      input_pass_condition.push_back("if (" + input_pass_cond + ")");
      input_pass_condition.push_back("  return -1;");
    }
  }
  return input_pass_condition;
}
std::vector<std::string> TorchAPI::arg_initialization_code() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  return arg_initialization;
}

std::vector<std::string> TorchAPI::header() const {
  std::vector<std::string> lines = {
    "#include <stdint.h>",
    "#include <stddef.h>",
    "#include <c10/util/irange.h>",
    "#include <cassert>",
    "#include <cstdlib>",
    "#include <torch/torch.h>",
    "#include \"fuzzer_util.h\"",
  };
  if (get_gen_covrunner())
    lines.push_back("#include <string>");
  lines.push_back("");
  lines.push_back("using namespace fuzzer_util;\n");
  return lines;
}
std::vector<std::string> TorchAPI::setup() const {
  std::vector<std::string> setup_code;

  setup_code.push_back("void PathFinderSetup() {");

  concat(setup_code, "  ", arg_setup_code());
  concat(setup_code, "  ", hard_constraint_code());
  concat(setup_code, "  ", soft_constraint_code());

  setup_code.push_back("}\n");

  return setup_code;
}
std::vector<std::string> TorchAPI::callback() const {
  std::vector<std::string> callback_code;

  if (get_gen_covrunner()) {
    callback_code.push_back(
      "int run_one_unit(const uint8_t *Data, size_t Size) {");
  } else {
    callback_code.push_back(
      "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {");
  }

  std::vector<std::string> arg_setups = arg_setup_code();
  callback_code.push_back(
    "  if (Size < " + std::to_string(arg_setups.size()) + " * sizeof (size_t)) return -1;");
  callback_code.push_back(
    "  const int* args = reinterpret_cast<const int*>(Data);");
  concat(callback_code, "  ", arg_setups, ";");

  callback_code.push_back("");
  concat(callback_code, "  ", hard_constraint_code());

  callback_code.push_back("");
  concat(callback_code, "  ", input_pass_condition_code());

  callback_code.push_back("");
  callback_code.push_back(
    "  torch::set_num_threads(1);\n");

  callback_code.push_back("\n  try {");
  concat(callback_code, "    ", arg_initialization_code());
  concat(callback_code, "    ", api_call_code());
  callback_code.push_back("  } catch (c10::Error& e) {");
  callback_code.push_back("    return abort_if_pytorch_internal_assertion_failed(e.what());");
  callback_code.push_back("  } catch (std::exception& e) {");
  callback_code.push_back("    return abort_if_not_expected_exception(e.what());");
  callback_code.push_back("  }\n");
  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}
std::vector<std::string> TorchAPI::footer() const {
  return {
    "int main(int argc, char* argv[]) {",
    "  std::string corpus_path = std::string(argv[1]);",
    "  std::string itv_mode = std::string(argv[2]);",
    "  size_t itv_start = atoi(argv[3]);",
    "  size_t itv_end = atoi(argv[4]);\n",

    "  std::vector<std::string> seed_paths = get_seed_paths(corpus_path, itv_mode, itv_start, itv_end);",
    "  for (auto& seed_path: seed_paths) {",
    "    std::cout << seed_path << std::endl;",
    "    Unit unit = FileToVector(seed_path);",
    "    run_one_unit(unit.data(), unit.size());",
    "  }\n",

    "  return 0;",
    "}\n",
  };
}


TorchFunction::TorchFunction(
  std::string func_name,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchAPI(func_name)
{
  params = std::move(params_);
}

std::vector<std::string> TorchFunction::api_call_code() const {
  std::string api_call = api_name + "(";
  for (size_t i = 0; i < params.size(); i++) {
    api_call += params[i]->expr();
    if (i != params.size() - 1)
      api_call += comma;
  }
  api_call += ");";
  
  return {api_call};
}


TorchTensorMethod::TorchTensorMethod(
  std::string method_name,
  std::unique_ptr<TorchTensorParam> self_,
  std::vector<std::unique_ptr<TorchParam>> params_)
  : TorchAPI(method_name)
{
  self = self_.get();
  params.push_back(std::move(self_));
  for (auto& param: params_)
    params.push_back(std::move(param));
}

std::vector<std::string> TorchTensorMethod::api_call_code() const {
  std::string api_call = tensor_method_self_var + "." + api_name + "(";
  for (size_t i = 1; i < params.size(); i++) {
    api_call += params[i]->expr();
    if (i != params.size() - 1)
      api_call += comma;
  }
  api_call += ");";
  
  return {api_call};
}


TorchModule::TorchModule(
  std::string module_name,
  std::unique_ptr<TorchParam> module_dtype_,
  std::vector<std::unique_ptr<TorchParam>> ctor_params_,
  std::vector<std::unique_ptr<TorchParam>> forward_params_)
  : TorchAPI(module_name)
{
  module_dtype = module_dtype_.get();
  params.push_back(std::move(module_dtype_));
  for (auto& param: forward_params_) {
    forward_params.push_back(param.get());
    params.push_back(std::move(param));
  }
  for (auto& param: ctor_params_) {
    ctor_params.push_back(param.get());
    params.push_back(std::move(param));
  }
}

std::vector<std::string> TorchModule::api_call_code() const {
  const std::string module_var = "module";

  std::string module_init =
    "auto " + module_var + assign + api_name + "(";
  for (size_t i = 0; i < ctor_params.size(); i++) {
    module_init += ctor_params[i]->expr();
    if (i != ctor_params.size() - 1)
      module_init += comma;
  }
  module_init += ");\n";
  
  std::string forward_call = module_var + "->forward(";
  for (size_t i = 0; i < forward_params.size(); i++) {
    forward_call += forward_params[i]->expr();
    if (i != forward_params.size() - 1)
      forward_call += comma;
  }
  forward_call += ");";

  return {
    module_init,
    module_var + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    forward_call,
  };
}
