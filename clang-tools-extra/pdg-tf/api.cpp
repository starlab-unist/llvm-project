#include "api.h"

static const std::string target_var = "target";
static const std::string sessionoptions_var = "options";
static const std::string config_var = "config";
static const std::string session_var = "session";
static const std::string graphdef_var = "graph_def";
static const std::string status_var = "status";
static const std::string outputs_var = "outputs";

TFAPI::TFAPI(
  std::string api_name_,
  std::vector<std::unique_ptr<TFParam>> params_)
  : api_name(api_name_), params(std::move(params_)) {}
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
  std::set<std::string> names_seen = {callback_input_var, scope_var, target_var};
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
    "#include \"tensorflow/core/kernels/pathfinder/fuzzer_util.h\"",
    "#include \"src/pathfinder.h\"\n",

    "using namespace tensorflow;",
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
    "SessionOptions " + sessionoptions_var + ";",
    "ConfigProto & " + config_var + " = " + sessionoptions_var + ".config;",
    config_var + ".set_inter_op_parallelism_threads(1);",
    config_var + ".set_intra_op_parallelism_threads(1);",
    config_var + ".set_use_per_session_threads(false);",
    "std::unique_ptr<tensorflow::Session>",
    "  " + session_var + "(tensorflow::NewSession(" + sessionoptions_var + "));\n",
  };
  concat(callback_code, "  ", basic_setup);

  concat(callback_code, "  ", input_pass_condition_code());

  concat(callback_code, "  ", arg_initialization_code());

  callback_code.push_back("  PathFinderExecuteTarget(");
  concat(callback_code, "    ", api_call_code());

  callback_code.push_back("    GraphDef " + graphdef_var + ";");
  callback_code.push_back("    Status " + status_var + " = " + scope_var + ".ToGraphDef(&" + graphdef_var + ");");
  callback_code.push_back("    if (!" + status_var + ".ok())");
  callback_code.push_back("      return -2;\n");
  callback_code.push_back("    " + status_var + " = " + session_var + "->Create(" + graphdef_var + ");");
  callback_code.push_back("    if (!" + status_var + ".ok())");
  callback_code.push_back("      return -2;\n");

  callback_code.push_back("    std::vector<Tensor> " + outputs_var + ";");

  
  callback_code.push_back("    " + status_var + " = " + session_var + "->Run({}, {\"" + target_var + "\"}, {}, &" + outputs_var + ");");
  callback_code.push_back("    if (!" + status_var + ".ok())");
  callback_code.push_back("      return -2;");
  callback_code.push_back("  );\n");

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

std::vector<std::string> TFAPI::api_call_code() const {
  std::string module_init =
    "auto " + target_var + assign + api_name + "(";
  assert(!params.empty() && params[0]->get_kind() == TFParam::TFPK_Scope);
  module_init += params[0]->expr() + ".WithOpName(\"" + target_var + "\")";
  for (size_t i = 1; i < params.size(); i++)
    module_init += comma + params[i]->expr();
  module_init += ");\n";

  return {module_init};
}
