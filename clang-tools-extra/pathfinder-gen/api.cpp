#include "api.h"

const std::string tensor_method_self_var = "self";

TorchAPI::TorchAPI(std::string api_name_): api_name(api_name_) {}
std::string TorchAPI::gen_fuzz_target(FuzzTargetType ftt) {
  set_fuzz_target_type(ftt);
  resolve_name_conflict();
  std::vector<std::string> lines;
  concat(lines, header());
  concat(lines, setup());
  concat(lines, callback());
  concat(lines, footer());
  return join_strs(lines, newline);
}
void TorchAPI::resolve_name_conflict() {
  std::set<std::string> names_seen = {callback_input_var};
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}
std::vector<std::string> TorchAPI::arg_setup_code() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchAPI::hard_constraint_code() const {
  std::vector<std::string> hard_constraint;
  for (auto& param: params)
    concat(hard_constraint, "  ", param->gen_hard_constraint(), comma);

  if (hard_constraint.empty())
    return {};

  hard_constraint.insert(hard_constraint.begin(), "PathFinderAddHardConstraint({");
  hard_constraint.push_back("});");
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
  for (auto& param: params)
    concat(
      input_pass_condition,
      "PathFinderPassIf(", param->gen_input_pass_condition(), ");");
  return input_pass_condition;
}
std::vector<std::string> TorchAPI::arg_initialization_code() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  return arg_initialization;
}

std::vector<std::string> TorchAPI::header() const {
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

  callback_code.push_back(
    "int PathFinderTestOneInput(const pathfinder::Input& " + callback_input_var + ") {");
  callback_code.push_back(
    "  torch::set_num_threads(1);\n");

  concat(callback_code, "  ", input_pass_condition_code());

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
    "}  // extern \"C\"\n",
    "int main(int argc, char **argv) {",
    "  pathfinder::parse_arg(argc, argv);",
    "  return pathfinder::driver(PathFinderTestOneInput);",
    "}\n",
  };
}


TorchFunction::TorchFunction(
  std::string func_name,
  std::vector<std::unique_ptr<TorchParam>> params_,
  bool is_void_)
  : TorchAPI(func_name)
{
  params = std::move(params_);
  is_void = is_void_;
}

std::vector<std::string> TorchFunction::api_call_code() const {
  std::vector<std::unique_ptr<TorchParam>> gpu_params;

  std::vector<std::string> all_tensors;
  for(auto& param: params){
    concat(all_tensors, param->all_tensors());
  }
  std::string init_all_tensors = "std::vector<torch::Tensor> all_tensors = {";
  for(size_t i = 0; i < all_tensors.size(); i++){
    init_all_tensors += all_tensors[i];
    if(i != all_tensors.size() - 1){
      init_all_tensors += comma;
    }
  }
  init_all_tensors += "};";
  
  for(auto& param: params){
    gpu_params.push_back(param->to_cuda());
  }
  
  std::vector<std::string> cuda_param_init;
  for(auto& cuda_param: gpu_params){
    concat(cuda_param_init, cuda_param->gen_arg_initialization());
  }

  std::string gpu_api_call = api_name + "(";
  for (size_t i = 0; i < params.size(); i++) {
    gpu_api_call += gpu_params[i]->expr();
    if (i != params.size() - 1)
      gpu_api_call += comma;
  }
  gpu_api_call += ")";

  std::string api_call = api_name + "(";
  for (size_t i = 0; i < params.size(); i++) {
    api_call += params[i]->expr();
    if (i != params.size() - 1)
      api_call += comma;
  }
  api_call += ")";

  std::vector<std::string> result;

  std::vector<std::string> cpu_api_call = !is_void ? 
  std::vector<std::string> { "PathFinderExecuteTarget(",
    "  auto result_cpu = " + api_call + ");\n",
  } : std::vector<std::string> {
    "PathFinderExecuteTarget(",
    "  " + api_call + ");\n",
  };

  concat(result, cuda_param_init);
  concat(result, cpu_api_call);

  std::vector<std::string> res_compare = !is_void ? std::vector<std::string> {
    "auto result_gpu = " + gpu_api_call + ";\n",
    "bool is_cons = true;\n",
    "for(int i = 0; i < 5 && is_cons; i++){",
    "  auto result_cpu2 = " + api_call + ";",
    "  is_cons = is_equal(result_cpu, result_cpu2);",
    "}\n",
    init_all_tensors,
    "bool ill_cond = false;",
    "for(auto t: all_tensors){",
    "  if(is_ill_conditioned(t)){",
    "    ill_cond = true;",
    "    break;}",
    "}\n",
    "if(is_cons && !ill_cond && !is_close(result_cpu, result_gpu)) {",
    "  torch::save(all_tensors, \"/root/pytorch/experiment_result/" + api_name + ".pt\");",
    "  return pathfinder::PATHFINDER_UNEXPECTED_EXCEPTION;", "}\n"
  } : std::vector<std::string>{
    gpu_api_call + ";\n"
  };

  concat(result, res_compare);

  return result;
}


TorchTensorMethod::TorchTensorMethod(
  std::string method_name,
  std::unique_ptr<TorchTensorParam> self_,
  std::vector<std::unique_ptr<TorchParam>> params_,
  bool is_void_)
  : TorchAPI(method_name)
{
  self = self_.get();
  params.push_back(std::move(self_));
  for (auto& param: params_)
    params.push_back(std::move(param));
  is_void = is_void_;
}

std::vector<std::string> TorchTensorMethod::api_call_code() const {

  std::vector<std::string> all_tensors;
  for(auto& param: params){
    concat(all_tensors, param->all_tensors());
  }
  std::string init_all_tensors = "std::vector<torch::Tensor> all_tensors = {";
  for(size_t i = 0; i < all_tensors.size(); i++){
    init_all_tensors += all_tensors[i];
    if(i != all_tensors.size() - 1){
      init_all_tensors += comma;
    }
  }
  init_all_tensors += "};";

  //initializing gpu_params
  std::vector<std::unique_ptr<TorchParam>> gpu_params;
  for(auto& param: params){
    gpu_params.push_back(param->to_cuda());
  }

  std::string tensor_method_self_var_1 = "torch::Tensor self_1 = self.clone();\n";

  std::string tensor_method_self_var_2 = "torch::Tensor self_2 = self.clone();\n";

  std::vector<std::string> cuda_param_init;
  cuda_param_init.push_back(tensor_method_self_var_1);
  cuda_param_init.push_back(tensor_method_self_var_2);
  for(auto& cuda_param: gpu_params){
    concat(cuda_param_init, cuda_param->gen_arg_initialization());
  }

  std::string gpu_api_call = tensor_method_self_var + "_cuda." + api_name + "(";
  for (size_t i = 1; i < params.size(); i++) {
    gpu_api_call += gpu_params[i]->expr();
    if (i != params.size() - 1)
      gpu_api_call += comma;
  }
  gpu_api_call += ")";

  std::string api_call = api_name + "(";
  for (size_t i = 1; i < params.size(); i++) {
    api_call += params[i]->expr();
    if (i != params.size() - 1)
      api_call += comma;
  }
  api_call += ")";

  std::vector<std::string> result;

  std::string api_call1 = tensor_method_self_var + "_1." + api_call;
  
  std::vector<std::string> cpu_api_call = !is_void ? 
  std::vector<std::string> {
    "PathFinderExecuteTarget(",
    "  auto result_cpu = " + api_call1 + ");\n",
  } : std::vector<std::string> {
    "PathFinderExecuteTarget(",
    "  " + api_call1 + ");\n",
  };

  concat(result, cuda_param_init);
  concat(result, cpu_api_call);

  std::string api_call2 = "self_2.clone()." + api_call;

  std::vector<std::string> res_compare = !is_void ? std::vector<std::string>{
    "auto result_gpu = " + gpu_api_call + ";\n",
    "bool is_cons = true;\n",
    "for(int i = 0; i < 5 && is_cons; i++){",
    "  auto result_cpu2 = " + api_call2 + ";",
    "  is_cons = is_equal(result_cpu, result_cpu2);",
    "}\n",
    init_all_tensors,
    "bool ill_cond = false;",
    "for(auto t: all_tensors){",
    "  if(is_ill_conditioned(t)){",
    "    ill_cond = true;",
    "    break;}",
    "}\n",
    "if(is_cons && !ill_cond && !is_close(result_cpu, result_gpu)) {",
    "  torch::save(all_tensors, \"/root/pytorch/experiment_result/" + api_name + ".pt\");",
    "  return pathfinder::PATHFINDER_UNEXPECTED_EXCEPTION;", "}\n"
  } : std::vector<std::string>{
    gpu_api_call + ";\n"
  };

  concat(result, res_compare);

  return result;
}


TorchModule::TorchModule(
  std::string module_name,
  std::unique_ptr<TorchParam> module_dtype_,
  std::vector<std::unique_ptr<TorchParam>> ctor_params_,
  std::vector<std::unique_ptr<TorchParam>> forward_params_,
  bool is_void_)
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
  is_void = is_void_;
}

std::vector<std::string> TorchModule::api_call_code() const {

  std::vector<std::string> all_tensors;
  for(auto& param: ctor_params){
    concat(all_tensors, param->all_tensors());
  }
  for(auto& param: forward_params){
    concat(all_tensors, param->all_tensors());
  }
  std::string init_all_tensors = "std::vector<torch::Tensor> all_tensors = {";
  for(size_t i = 0; i < all_tensors.size(); i++){
    init_all_tensors += all_tensors[i];
    if(i != all_tensors.size() - 1){
      init_all_tensors += comma;
    }
  }
  init_all_tensors += "};";

  const std::string module_var_gpu = "module_gpu";

  std::string gpu_module_init =
    "auto " + module_var_gpu + assign + api_name + "(";
  
  std::vector<std::unique_ptr<TorchParam>> gpu_ctor_params;
  for(auto& param: ctor_params){
    gpu_ctor_params.push_back(param->to_cuda());
  }
  std::vector<std::unique_ptr<TorchParam>> gpu_forward_params;
  for(auto& param: forward_params){
    gpu_forward_params.push_back(param->to_cuda());
  }

  //initialize cuda params
  std::vector<std::string> cuda_param_init;
  for(auto& param: gpu_ctor_params){
    concat(cuda_param_init, param->gen_arg_initialization());
  }
  for(auto& param: gpu_forward_params){
    concat(cuda_param_init, param->gen_arg_initialization());
  }

  for (size_t i = 0; i < ctor_params.size(); i++) {
    gpu_module_init += gpu_ctor_params[i]->expr();
    if (i != ctor_params.size() - 1)
      gpu_module_init += comma;
  }
  gpu_module_init += ");\n";

  std::string gpu_forward_call = module_var_gpu + "->forward(";
  for (size_t i = 0; i < forward_params.size(); i++) {
    gpu_forward_call += gpu_forward_params[i]->expr();
    if (i != forward_params.size() - 1)
      gpu_forward_call += comma;
  }
  gpu_forward_call += ")";

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
  forward_call += ")";

  std::vector<std::string> result;

  std::vector<std::string> cpu_api_call = !is_void ? 
  std::vector<std::string>{
    "PathFinderExecuteTarget(",
    "  " + module_init,
    "  " + module_var + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    "  auto result_cpu = " + forward_call + ");\n",
  } : std::vector<std::string>{
    "PathFinderExecuteTarget(",
    "  " + module_init,
    "  " + module_var + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    "  " + forward_call + ");\n",
  };

  concat(result, cuda_param_init);
  concat(result, cpu_api_call);

  std::vector<std::string> res_compare = !is_void ? 
  std::vector<std::string>{
    gpu_module_init,
    module_var_gpu + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    "auto result_gpu = " + gpu_forward_call + semicolon + newline,
    "bool is_cons = true;\n",
    "for(int i = 0; i < 5 && is_cons; i++){",
    "  auto result_cpu2 = " + forward_call + semicolon + ";",
    "  is_cons = is_equal(result_cpu, result_cpu2);",
    "}\n",
    init_all_tensors,
    "bool ill_cond = false;",
    "for(auto t: all_tensors){",
    "  if(is_ill_conditioned(t)){",
    "    ill_cond = true;",
    "    break;}",
    "}\n",
    "if(is_cons && !ill_cond && !is_close(result_cpu, result_gpu)) {",
    "  torch::save(all_tensors, \"/root/pytorch/experiment_result/" + api_name + ".pt\");",
    "  return pathfinder::PATHFINDER_UNEXPECTED_EXCEPTION;", "}\n"
  } : std::vector<std::string>{
    gpu_module_init,
    module_var_gpu + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    gpu_forward_call + semicolon + newline,
  };

  concat(result, res_compare);

  return result;

}
