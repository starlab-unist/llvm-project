#ifndef PATHFINDER_GEN_API
#define PATHFINDER_GEN_API

#include "param.h"

using namespace llvm;
using namespace clang;

class TorchAPI {
  public:
    TorchAPI(std::string api_name_);
    std::string gen_fuzz_target() const;
  protected:
    std::string api_name;
    std::vector<std::unique_ptr<TorchParam>> params;
  private:
    std::vector<std::string> arg_setup_code() const;
    std::vector<std::string> hard_constraint_code() const;
    std::vector<std::string> soft_constraint_code() const;
    std::vector<std::string> input_pass_condition_code() const;
    std::vector<std::string> arg_initialization_code() const;
    virtual std::vector<std::string> api_call_code() const = 0;

    std::vector<std::string> header() const;
    std::vector<std::string> setup() const;
    std::vector<std::string> callback() const;
    std::vector<std::string> footer() const;
};

class TorchFunction: public TorchAPI {
  public:
    TorchFunction(
      std::string func_name,
      std::vector<std::unique_ptr<TorchParam>> params_);
  private:
    virtual std::vector<std::string> api_call_code() const override;
};

class TorchModule: public TorchAPI {
  public:
    TorchModule(
      std::string module_name,
      std::unique_ptr<TorchParam> module_dtype_,
      std::vector<std::unique_ptr<TorchParam>> ctor_params_,
      std::vector<std::unique_ptr<TorchParam>> forward_params_);
  private:
    TorchParam* module_dtype;
    std::vector<TorchParam*> ctor_params;
    std::vector<TorchParam*> forward_params;

    virtual std::vector<std::string> api_call_code() const override;
};

#endif
