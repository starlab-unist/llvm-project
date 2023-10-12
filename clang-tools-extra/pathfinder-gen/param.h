#ifndef PATHFINDER_GEN_PARAM
#define PATHFINDER_GEN_PARAM

#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace clang;

enum ParamType {
  INT,
  BOOL,
  FLOAT,
  DTYPE,
  ENUM,
  TENSOR,
  INTVECTOR,
  FLOATVECTOR,
  INTARRAYREF,
  EXPANDINGARRAY,
  EXPANDINGARRAYWITHOPTIONALELEM,
  OPTIONAL,
  VARIANT,
  MAP,
};

class Param {
  public:
    Param(ParamType ptype_, std::string name_);
    Param(ParamType ptype_, std::string name_, long size);
    Param(ParamType ptype_, std::string name_, std::unique_ptr<Param> base_);
    Param(ParamType ptype_, std::string name_, std::vector<std::unique_ptr<Param>> enums_, std::unique_ptr<Param> expandingarray_);
    Param(
      ParamType ptype_,
      std::string name_,
      std::string map_name_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params_,
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries_);
    std::pair<size_t, size_t> set_offset(size_t enum_offset, size_t int_offset);
    void set_default(std::string param_name, const CXXRecordDecl* cdecl);
    void setup_arg(std::vector<std::string>& args);
    void constraint(std::vector<std::string>& hard_ctrs, std::vector<std::string>& soft_ctrs, bool is_module);
    std::tuple<std::vector<std::string>, std::vector<std::string>, std::string, std::vector<std::string>> to_code(std::string api_name, bool is_module);
    bool is_map();
  private:
    std::string type_str();
    std::string array_str();

    ParamType ptype;
    std::string name;
    size_t enum_offset_start;
    size_t enum_offset_size = 0;
    size_t int_offset_start;
    size_t int_offset_size = 0;

    // INT
    Optional<size_t> default_int = None;

    // TENSOR
    std::string name_dtype;
    std::string name_rank;
    std::vector<std::string> name_dim;

    // ARRAY, VECTOR
    size_t array_size;
    std::string name_size;
    std::vector<std::string> name_elem;

    // OPTIONAL
    std::unique_ptr<Param> base;

    // VARIANT
    std::vector<std::unique_ptr<Param>> enums;
    std::unique_ptr<Param> expandingarray;

    // MAP
    std::string map_name;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params;
    std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries;

    friend size_t set_idx(std::vector<std::unique_ptr<Param>>& params);
    friend std::string gen_api_call(std::string api_name, std::vector<std::unique_ptr<Param>>& params, bool is_module, size_t num_input_tensor);
    friend void gen_pathfinder_fuzz_target(
      std::string target_api_name,
      std::vector<std::unique_ptr<Param>>& params,
      std::ostream& os);
    friend Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx);
    friend Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx);
};

std::string gen_torch_function_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params);
std::string gen_torch_module_pathfinder(std::string api_name, std::vector<std::unique_ptr<Param>>& params, size_t num_input_tensor);

#endif
