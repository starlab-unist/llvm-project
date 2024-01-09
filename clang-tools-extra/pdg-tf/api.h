#ifndef PATHFINDER_GEN_API
#define PATHFINDER_GEN_API

#include "param.h"

using namespace llvm;
using namespace clang;

class TFAPI {
  public:
    TFAPI(std::string api_name_);
    std::string gen_fuzz_target(FuzzTargetType ftt);
  protected:
    std::string api_name;
    std::vector<std::unique_ptr<TFParam>> params;
  private:
    void resolve_name_conflict();
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

class TFFunction: public TFAPI {
  public:
    TFFunction(
      std::string func_name,
      std::vector<std::unique_ptr<TFParam>> params_);
  private:
    virtual std::vector<std::string> api_call_code() const override;
};

extern const std::string tensor_method_self_var;

class TFTensorMethod: public TFAPI {
  public:
    TFTensorMethod(
      std::string method_name,
      std::unique_ptr<TFTensorParam> self_,
      std::vector<std::unique_ptr<TFParam>> params_);
  private:
    TFParam* self;

    virtual std::vector<std::string> api_call_code() const override;
};

class TFModule: public TFAPI {
  public:
    TFModule(
      std::string module_name,
      //std::unique_ptr<TFParam> module_dtype_,
      std::vector<std::unique_ptr<TFParam>> ctor_params_);
  private:
    TFParam* module_dtype;
    std::vector<TFParam*> ctor_params;

    virtual std::vector<std::string> api_call_code() const override;
};

#endif
