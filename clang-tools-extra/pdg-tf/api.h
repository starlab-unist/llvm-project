#ifndef PATHFINDER_GEN_API
#define PATHFINDER_GEN_API

#include "param.h"

using namespace llvm;
using namespace clang;

class TFAPI {
  public:
    TFAPI(
      std::string api_name_,
      std::vector<std::unique_ptr<TFParam>> params_);
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
    std::vector<std::string> api_call_code() const;

    std::vector<std::string> header() const;
    std::vector<std::string> setup() const;
    std::vector<std::string> callback() const;
    std::vector<std::string> footer() const;
};

#endif
