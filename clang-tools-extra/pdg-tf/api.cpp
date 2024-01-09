#include "api.h"

const std::string tensor_method_self_var = "self";

TFAPI::TFAPI(std::string api_name_): api_name(api_name_) {}
std::string TFAPI::gen_fuzz_target(FuzzTargetType ftt) {
  set_fuzz_target_type(ftt);
  resolve_name_conflict();
  std::vector<std::string> lines;
  concat(lines, header());
  concat(lines, setup());
  concat(lines, callback());
  concat(lines, footer());
  return join_strs(lines, newline);
}
void TFAPI::resolve_name_conflict() {
  std::set<std::string> names_seen = {callback_input_var};
  for (auto& param: params)
    param->resolve_name_conflict(names_seen);
}
std::vector<std::string> TFAPI::arg_setup_code() const {
  std::vector<std::string> arg_setup;
  for (auto& param: params)
    concat(arg_setup, param->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFAPI::hard_constraint_code() const {
  std::vector<std::string> hard_constraint;
  for (auto& param: params)
    concat(hard_constraint, "  ", param->gen_hard_constraint(), comma);

  if (hard_constraint.empty())
    return {};

  hard_constraint.insert(hard_constraint.begin(), "PathFinderAddHardConstraint({");
  hard_constraint.push_back("});");
  return hard_constraint;
}
std::vector<std::string> TFAPI::soft_constraint_code() const {
  std::vector<std::string> soft_constraint;
  for (auto& param: params)
    concat(soft_constraint, "  ", param->gen_soft_constraint(), comma);

  if (soft_constraint.empty())
    return {};

  soft_constraint.insert(soft_constraint.begin(), "PathFinderAddSoftConstraint({");
  soft_constraint.push_back("});");
  return soft_constraint;
}
std::vector<std::string> TFAPI::input_pass_condition_code() const {
  std::vector<std::string> input_pass_condition;
  for (auto& param: params)
    concat(
      input_pass_condition,
      "PathFinderPassIf(", param->gen_input_pass_condition(), ");");
  if (!input_pass_condition.empty())
    input_pass_condition.back() += "\n";
  return input_pass_condition;
}
std::vector<std::string> TFAPI::arg_initialization_code() const {
  std::vector<std::string> arg_initialization;
  for (auto& param: params)
    concat(arg_initialization, param->gen_arg_initialization());
  return arg_initialization;
}

std::vector<std::string> TFAPI::header() const {
  return {
    "#include \"tensorflow/cc/framework/scope.h\"",
    "#include \"tensorflow/core/graph/graph.h\"",
    "#include \"tensorflow/core/public/session.h\"",
    "#include \"tensorflow/cc/ops/array_ops.h\"",
    "#include \"tensorflow/cc/ops/standard_ops.h\"",
    "#include \"pathfinder.h\"",
    "#include \"fuzzer_util.h\"\n",

    "using namespace tensorflow;\n",
    "using namespace fuzzer_util;\n",

    "extern \"C\" {\n",
  };
}
std::vector<std::string> TFAPI::setup() const {
  std::vector<std::string> setup_code;

  setup_code.push_back("void PathFinderSetup() {");

  concat(setup_code, "  ", arg_setup_code());
  concat(setup_code, "  ", hard_constraint_code());
  concat(setup_code, "  ", soft_constraint_code());

  setup_code.push_back("}\n");

  return setup_code;
}
std::vector<std::string> TFAPI::callback() const {
  std::vector<std::string> callback_code;

  callback_code.push_back(
    "int PathFinderTestOneInput(const pathfinder::Input& " + callback_input_var + ") {");
  std::vector<std::string> basic_setup = {
    "SessionOptions options;",
    "ConfigProto & config = options.config;",
    "config.set_inter_op_parallelism_threads(1);",
    "config.set_intra_op_parallelism_threads(1);",
    "config.set_use_per_session_threads(false);",
    "std::unique_ptr<tensorflow::Session>",
    "  session(tensorflow::NewSession(options));\n",
  };
  concat(callback_code, "  ", basic_setup);

  concat(callback_code, "  ", input_pass_condition_code());

  concat(callback_code, "  ", arg_initialization_code());
  concat(callback_code, "  ", api_call_code());

  callback_code.push_back("  GraphDef graph_def;");
  callback_code.push_back("  TF_CHECK_OK(root.ToGraphDef(&graph_def));");
  callback_code.push_back("  Status status = session->Create(graph_def);");
  callback_code.push_back("  if (!status.ok()) {");
  callback_code.push_back("    LOG(FATAL) << \"Could not create session: \" << status.message();");
  callback_code.push_back("  }\n");

  callback_code.push_back("  std::vector<Tensor> outputs;\n");

  callback_code.push_back("  PathFinderExecuteTarget(");
  callback_code.push_back("    status = session->Run({}, {\"result\"}, {\"result\"}, &outputs));\n");

  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}
std::vector<std::string> TFAPI::footer() const {
  return {
    "}  // extern \"C\"\n",
    "int main(int argc, char **argv) {",
    "  pathfinder::parse_arg(argc, argv);",
    "  return pathfinder::driver(PathFinderTestOneInput);",
    "}\n",
  };
}


TFFunction::TFFunction(
  std::string func_name,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFAPI(func_name)
{
  params = std::move(params_);
}

std::vector<std::string> TFFunction::api_call_code() const {
  std::string api_call = api_name + "(";
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


TFTensorMethod::TFTensorMethod(
  std::string method_name,
  std::unique_ptr<TFTensorParam> self_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : TFAPI(method_name)
{
  self = self_.get();
  params.push_back(std::move(self_));
  for (auto& param: params_)
    params.push_back(std::move(param));
}

std::vector<std::string> TFTensorMethod::api_call_code() const {
  std::string api_call = tensor_method_self_var + "." + api_name + "(";
  for (size_t i = 1; i < params.size(); i++) {
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


TFModule::TFModule(
  std::string module_name,
  //std::unique_ptr<TFParam> module_dtype_,
  std::vector<std::unique_ptr<TFParam>> ctor_params_)
  : TFAPI(module_name)
{
  //module_dtype = module_dtype_.get();
  //params.push_back(std::move(module_dtype_));
  for (auto& param: ctor_params_) {
    ctor_params.push_back(param.get());
    params.push_back(std::move(param));
  }
}

std::vector<std::string> TFModule::api_call_code() const {
  const std::string module_var = "module";

  std::string module_init =
    "auto " + module_var + assign + api_name + "(";
  for (size_t i = 0; i < ctor_params.size(); i++) {
    module_init += ctor_params[i]->expr();
    if (i != ctor_params.size() - 1)
      module_init += comma;
  }
  module_init += ");\n";
  
  /* std::string forward_call = module_var + "->forward(";
  for (size_t i = 0; i < forward_params.size(); i++) {
    forward_call += forward_params[i]->expr();
    if (i != forward_params.size() - 1)
      forward_call += comma;
  }
  forward_call += ")"; */

  return {
    module_init,
    /* module_var + "->to" + bracket(module_dtype->expr()) + semicolon + newline,
    "PathFinderExecuteTarget(",
    "  " + forward_call + ");", */
  };
  return {};
}
