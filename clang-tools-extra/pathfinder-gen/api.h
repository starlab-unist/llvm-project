#ifndef PATHFINDER_GEN_API
#define PATHFINDER_GEN_API

#include "param.h"

using namespace llvm;
using namespace clang;

class TorchAPI {
  public:
    TorchAPI(std::string api_name_): api_name(api_name_) {}
    std::string gen_fuzz_target() const {
      std::vector<std::string> lines;
      concat(lines, header());
      concat(lines, setup());
      concat(lines, callback());
      concat(lines, footer());
      return join_strs(lines, newline);
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
    virtual std::vector<std::string> api_call_code() const = 0;

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

      return setup_code;
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

      return callback_code;
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
    virtual std::vector<std::string> api_call_code() const override {
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

    virtual std::vector<std::string> api_call_code() const override {
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

#endif
