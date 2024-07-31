#ifndef PATHFINDER_GEN_PARAM
#define PATHFINDER_GEN_PARAM

#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Casting.h"
#include "utils.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace clang;

enum FuzzTargetType {
  FTT_Basic,
  FTT_Sparse,
};
void set_fuzz_target_type(FuzzTargetType ftt_);

extern const size_t MAX_VECTOR_SIZE;
extern const size_t MAX_ARRAYREF_SIZE;

const std::string symbolic_int_var = "sym_int_arg";
const std::string callback_input_var = "args";

void set_function_mode();
void set_module_mode();
bool is_module_mode();

class TorchParam {
  public:
    enum TorchParamKind {
      TPK_Enum,

      TPK_Int,
      TPK_SymInt,
      TPK_UnsignedInt,
      TPK_Dtype,
      
      // TorchBoundedParam
      TPK_Null,
      TPK_BoundedInt,
      TPK_Bool,
      TPK_String,
      TPK_BFloat,
      TPK_Half,
      TPK_Float,
      TPK_Double,
      TPK_MemoryFormat,
      TPK_Layout,
      TPK_Device,
      TPK_BasicDtype,
      TPK_ScalarDtype,
      TPK_SparseDtype,
      TPK_Variant,
      TPK_Bounded_First = TPK_Null,
      TPK_Bounded_Last = TPK_Variant,

      // TorchUnfixedArrayParam
      TPK_Vector,
      TPK_ArrayRef,
      TPK_OptionalArrayRef,
      TPK_UnfixedArray_First = TPK_Vector,
      TPK_UnfixedArray_Last = TPK_OptionalArrayRef,

      // TorchfixedArrayParam
      TPK_ExpandingArray,
      TPK_ExpandingArrayWithOptionalElem,
      TPK_Tuple,
      TPK_Pair,
      TPK_FixedArray_First = TPK_ExpandingArray,
      TPK_FixedArray_Last = TPK_Pair,

      TPK_Tensor,
      TPK_Scalar,
      TPK_Optional,
      TPK_APIOptions,
    };

    TorchParam(TorchParamKind kind_, std::string name_): kind(kind_), name(name_) {}
    virtual void set_default(Expr* default_expr) {}

    virtual std::string type() const = 0;
    virtual std::string var() const { return ""; }
    virtual std::string initializer() const = 0;
    std::string expr() const {
      return var() != "" ? var() : initializer();
    };

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const { return index; }
    virtual std::vector<std::string> gen_hard_constraint() const { return {}; }
    virtual std::vector<std::string> gen_soft_constraint() const { return {}; }
    virtual std::vector<std::string> gen_input_pass_condition() const { return {}; }
    virtual std::vector<std::string> gen_arg_initialization() const { return {}; }
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) {
      name = unique_name(name, names_seen);
    }
    
    virtual bool stable() const { return true; }
    std::string get_name() const { return name; }
    TorchParamKind get_kind() const { return kind; }
  private:
    const TorchParamKind kind;
  protected:
    std::string name;
};

class TorchIntParam: public TorchParam {
  public:
    TorchIntParam(std::string name_, std::string range_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TorchParam *param);
  private:
    std::string range;
    Optional<int> default_value = None;
};

class TorchSymIntParam: public TorchParam {
  public:
    TorchSymIntParam(std::string name_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchIntParam> intparam;
};

class TorchUnsignedIntParam: public TorchParam {
  public:
    TorchUnsignedIntParam(std::string name_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(unsigned int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TorchParam *param);
  private:
    Optional<unsigned int> default_value = None;
};

class TorchBoundedParam: public TorchParam {
  public:
    TorchBoundedParam(TorchParamKind kind_, std::string name_, const std::vector<std::string>& value_names_);
    TorchBoundedParam(TorchParamKind kind_, std::string name_, size_t size_);
    TorchBoundedParam(TorchParamKind kind_, std::string name_, const std::string& value_list_var_);

    virtual std::string type() const override = 0;
    virtual std::string initializer() const override = 0;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
  protected:
    std::vector<std::string> value_names;
    Optional<size_t> size;
    std::string value_list_var;
};

class TorchNullParam: public TorchBoundedParam {
  public:
    TorchNullParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchBoundedIntParam: public TorchBoundedParam {
  public:
    TorchBoundedIntParam(std::string name_, size_t size_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchBoolParam: public TorchBoundedParam {
  public:
    TorchBoolParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchStringParam: public TorchBoundedParam {
  public:
    TorchStringParam(std::string name_, bool is_view_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
  private:
    bool is_view;
};

class TorchBFloatParam: public TorchBoundedParam {
  public:
    TorchBFloatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchHalfParam: public TorchBoundedParam {
  public:
    TorchHalfParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchFloatParam: public TorchBoundedParam {
  public:
    TorchFloatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchDoubleParam: public TorchBoundedParam {
  public:
    TorchDoubleParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchMemoryFormatParam: public TorchBoundedParam {
  public:
    TorchMemoryFormatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchLayoutParam: public TorchBoundedParam {
  public:
    TorchLayoutParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchDeviceParam: public TorchBoundedParam {
  public:
    TorchDeviceParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchBasicDtypeParam: public TorchBoundedParam {
  public:
    TorchBasicDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchScalarDtypeParam: public TorchBoundedParam {
  public:
    TorchScalarDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchSparseDtypeParam: public TorchBoundedParam {
  public:
    TorchSparseDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchDtypeParam: public TorchParam {
  public:
    TorchDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchBasicDtypeParam> basic;
    std::unique_ptr<TorchSparseDtypeParam> sparse;
};

class TorchVariantParam: public TorchBoundedParam {
  public:
    TorchVariantParam(
      std::string name_,
      std::vector<std::unique_ptr<TorchParam>> params_);
    virtual void set_default(Expr* default_expr) override;

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TorchParam *param);
  private:
    std::vector<std::unique_ptr<TorchParam>> params;

    std::vector<std::string> gen_typedef() const;
    std::vector<std::string> gen_vector() const;
};

class TorchEnumParam: public TorchParam {
  public:
    TorchEnumParam(std::string name_, std::string enum_name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
  private:
    std::string enum_name;
};

class TorchUnfixedArrayParam: public TorchParam {
  public:
    TorchUnfixedArrayParam(
      TorchParamKind kind_,
      std::string name_,
      std::vector<std::unique_ptr<TorchParam>> params_);
    TorchUnfixedArrayParam(
      TorchParamKind kind_,
      std::string name_,
      std::string base_type_str_);

    virtual std::string type() const override = 0;
    virtual std::string var() const override;
    virtual std::string initializer() const override = 0;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    virtual bool stable() const override;
    std::string base_type() const;
  protected:
    std::unique_ptr<TorchBoundedIntParam> size;
    std::vector<std::unique_ptr<TorchParam>> params;
    std::string base_type_str;
};

class TorchVectorParam: public TorchUnfixedArrayParam {
  public:
    TorchVectorParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_);
    TorchVectorParam(std::string name_, std::string base_type_str_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
  private:
    std::string base_type_str;
};

/* class TorchArrayRefParam: public TorchUnfixedArrayParam {
  public:
    TorchArrayRefParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
}; */

class TorchArrayRefParam: public TorchParam {
  public:
    TorchArrayRefParam(std::string name_, std::vector<std::unique_ptr<TorchParam>> params_);
    TorchArrayRefParam(std::string name_, std::string base_type_str_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    //virtual bool stable() const override;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchVectorParam> vec;
};

class TorchFixedArrayParam: public TorchParam {
  public:
    TorchFixedArrayParam(
      TorchParamKind kind_,
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TorchParam>> params_);

    virtual std::string type() const override = 0;
    virtual std::string var() const override;
    virtual std::string initializer() const override = 0;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;
  protected:
    size_t size;
    std::vector<std::unique_ptr<TorchParam>> params;
};

class TorchExpandingArrayParam: public TorchFixedArrayParam {
  public:
    TorchExpandingArrayParam(
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TorchParam>> params_);
    virtual void set_default(Expr* default_expr) override;

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchExpandingArrayWithOptionalElemParam: public TorchFixedArrayParam {
  public:
    TorchExpandingArrayWithOptionalElemParam(
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TorchParam>> params_);
    virtual void set_default(Expr* default_expr) override;

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
  private:
    std::string base_type() const;
};

class TorchTupleParam: public TorchFixedArrayParam {
  public:
    TorchTupleParam(
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TorchParam>> params_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchPairParam: public TorchFixedArrayParam {
  public:
    TorchPairParam(
      std::string name_,
      std::vector<std::unique_ptr<TorchParam>> params_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TorchParam *param);
};

class TorchTensorParam: public TorchParam {
  public:
    TorchTensorParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchDtypeParam> dtype;
    std::unique_ptr<TorchLayoutParam> layout;
    std::unique_ptr<TorchBoundedIntParam> rank;
    std::vector<std::unique_ptr<TorchIntParam>> dims;
};

class TorchScalarParam: public TorchParam {
  public:
    TorchScalarParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchScalarDtypeParam> dtype;
    std::unique_ptr<TorchIntParam> intValue;
    std::unique_ptr<TorchUnsignedIntParam> uintValue;
    std::unique_ptr<TorchBFloatParam> bfloatValue;
    std::unique_ptr<TorchHalfParam> halfValue;
    std::unique_ptr<TorchFloatParam> floatValue;
    std::unique_ptr<TorchDoubleParam> doubleValue;
    std::unique_ptr<TorchHalfParam> realValue32;
    std::unique_ptr<TorchHalfParam> imaginaryValue32;
    std::unique_ptr<TorchFloatParam> realValue64;
    std::unique_ptr<TorchFloatParam> imaginaryValue64;
    std::unique_ptr<TorchDoubleParam> realValue128;
    std::unique_ptr<TorchDoubleParam> imaginaryValue128;
    std::unique_ptr<TorchBoolParam> boolValue;
    std::vector<TorchParam*> params;
};

class TorchOptionalParam: public TorchParam {
  public:
    TorchOptionalParam(std::string name_, std::unique_ptr<TorchParam> param_);
    TorchOptionalParam(std::string name_, std::string base_type_str_);
    virtual void set_default(Expr* default_expr) override;

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    virtual bool stable() const override;
    std::string base_type() const;

    static bool classof(const TorchParam *param);
  private:
    std::unique_ptr<TorchBoolParam> has_value;
    std::unique_ptr<TorchParam> param;
    std::string base_type_str;
};



class TorchAPIOptionsParam: public TorchParam {
  public:
    TorchAPIOptionsParam(
      std::string name_,
      std::string api_optons_class_name_,
      std::vector<std::unique_ptr<TorchParam>> ctor_params_,
      std::vector<std::unique_ptr<TorchParam>> member_params_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual size_t gen_arg_setup(std::vector<std::string>& arg_setups, size_t index) const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TorchParam *param);
  private:
    std::string api_optons_class_name;
    std::vector<std::unique_ptr<TorchParam>> ctor_params;
    std::vector<std::unique_ptr<TorchParam>> member_params;
    std::vector<std::string> member_param_setters;

    std::vector<std::string> gen_member_param_set() const;

    std::vector<std::string> gen_api_options_init() const;
};

#endif
