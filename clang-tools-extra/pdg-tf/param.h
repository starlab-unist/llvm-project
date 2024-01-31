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
  FTT_Quantization,
  FTT_Sparse,
};
void set_fuzz_target_type(FuzzTargetType ftt_);

extern const size_t MAX_VECTOR_SIZE;
extern const size_t MAX_ARRAYREF_SIZE;

const std::string symbolic_int_var = "sym_int_arg";
const std::string callback_input_var = "x";
const std::string scope_var = "scope";

class TFParam {
  public:
    enum TFParamKind {
      TFPK_Scope,

      TFPK_Int,
      TFPK_SymInt,
      TFPK_UnsignedInt,
      TFPK_Dtype,
      TFPK_InputList,
      
      // TFBoundedParam
      TFPK_Null,
      TFPK_BoundedInt,
      TFPK_Bool,
      TFPK_StringPiece,
      TFPK_DataFormat,
      TFPK_Padding,
      TFPK_BFloat,
      TFPK_Half,
      TFPK_Float,
      TFPK_Double,
      TFPK_BasicDtype,
      TFPK_ExtendedDtype,
      TFPK_Variant,
      TFPK_Bounded_First = TFPK_Null,
      TFPK_Bounded_Last = TFPK_Variant,

      // TFUnfixedArrayParam
      TFPK_Vector,
      TFPK_UnfixedArray_First = TFPK_Vector,
      TFPK_UnfixedArray_Last = TFPK_Vector,

      TFPK_ArraySlice,

      // TFfixedArrayParam
      TFPK_Array,
      TFPK_Tuple,
      TFPK_Pair,
      TFPK_FixedArray_First = TFPK_Array,
      TFPK_FixedArray_Last = TFPK_Pair,

      TFPK_PartialTensorShape,
      TFPK_Tensor,
      TFPK_Input,
      TFPK_APIAttrs,
    };

    TFParam(TFParamKind kind_, std::string name_): kind(kind_), name(name_) {}
    virtual void set_default(Expr* default_expr) {}

    virtual std::string type() const = 0;
    virtual std::string var() const { return ""; }
    virtual std::string initializer() const = 0;
    std::string expr() const {
      return var() != "" ? var() : initializer();
    };

    virtual std::vector<std::string> gen_arg_setup() const { return {}; }
    virtual std::vector<std::string> gen_hard_constraint() const { return {}; }
    virtual std::vector<std::string> gen_soft_constraint() const { return {}; }
    virtual std::vector<std::string> gen_input_pass_condition() const { return {}; }
    virtual std::vector<std::string> gen_arg_initialization() const { return {}; }
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) {
      name = unique_name(name, names_seen);
    }
    
    virtual bool stable() const { return true; }
    std::string get_name() const { return name; }
    TFParamKind get_kind() const { return kind; }
  private:
    const TFParamKind kind;
  protected:
    std::string name;
};

class TFScopeParam: public TFParam {
  public:
    TFScopeParam();

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;
};

class TFIntParam: public TFParam {
  public:
    TFIntParam(std::string name_, std::string range_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TFParam *param);
  private:
    std::string range;
    Optional<int> default_value = std::nullopt;
};

class TFSymIntParam: public TFParam {
  public:
    TFSymIntParam(std::string name_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFIntParam> intparam;
};

class TFUnsignedIntParam: public TFParam {
  public:
    TFUnsignedIntParam(std::string name_);
    virtual void set_default(Expr* default_expr) override;
    void set_default(unsigned int value);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;

    static bool classof(const TFParam *param);
  private:
    Optional<unsigned int> default_value = std::nullopt;
};

class TFBoundedParam: public TFParam {
  public:
    TFBoundedParam(TFParamKind kind_, std::string name_, const std::vector<std::string>& value_names_);
    TFBoundedParam(TFParamKind kind_, std::string name_, size_t size_);
    TFBoundedParam(TFParamKind kind_, std::string name_, const std::string& value_list_var_);

    virtual std::string type() const override = 0;
    virtual std::string initializer() const override = 0;

    virtual std::vector<std::string> gen_arg_setup() const override;
  protected:
    std::vector<std::string> value_names;
    Optional<size_t> size;
    std::string value_list_var;
};

class TFNullParam: public TFBoundedParam {
  public:
    TFNullParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFBoundedIntParam: public TFBoundedParam {
  public:
    TFBoundedIntParam(std::string name_, size_t size_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFBoolParam: public TFBoundedParam {
  public:
    TFBoolParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFStringPieceParam: public TFBoundedParam {
  public:
    TFStringPieceParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFDataFormatParam: public TFBoundedParam {
  public:
    TFDataFormatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFPaddingParam: public TFBoundedParam {
  public:
    TFPaddingParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFStringParam: public TFBoundedParam {
  public:
    TFStringParam(std::string name_, bool is_view_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
  private:
    bool is_view;
};

class TFBFloatParam: public TFBoundedParam {
  public:
    TFBFloatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFHalfParam: public TFBoundedParam {
  public:
    TFHalfParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFFloatParam: public TFBoundedParam {
  public:
    TFFloatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFDoubleParam: public TFBoundedParam {
  public:
    TFDoubleParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFMemoryFormatParam: public TFBoundedParam {
  public:
    TFMemoryFormatParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFLayoutParam: public TFBoundedParam {
  public:
    TFLayoutParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFDeviceParam: public TFBoundedParam {
  public:
    TFDeviceParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFBasicDtypeParam: public TFBoundedParam {
  public:
    TFBasicDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFExtendedDtypeParam: public TFBoundedParam {
  public:
    TFExtendedDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFDtypeParam: public TFParam {
  public:
    TFDtypeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFBasicDtypeParam> basic;
    std::unique_ptr<TFExtendedDtypeParam> extended;
};

class TFVariantParam: public TFBoundedParam {
  public:
    TFVariantParam(
      std::string name_,
      std::vector<std::unique_ptr<TFParam>> params_);
    virtual void set_default(Expr* default_expr) override;

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::vector<std::unique_ptr<TFParam>> params;

    std::vector<std::string> gen_typedef() const;
    std::vector<std::string> gen_vector() const;
};

class TFEnumParam: public TFParam {
  public:
    TFEnumParam(std::string name_, std::string enum_name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
  private:
    std::string enum_name;
};

class TFUnfixedArrayParam: public TFParam {
  public:
    TFUnfixedArrayParam(
      TFParamKind kind_,
      std::string name_,
      std::vector<std::unique_ptr<TFParam>> params_);
    TFUnfixedArrayParam(
      TFParamKind kind_,
      std::string name_,
      std::string base_type_str_);

    virtual std::string type() const override = 0;
    virtual std::string var() const override;
    virtual std::string initializer() const override = 0;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    virtual bool stable() const override;
    std::string base_type() const;
  protected:
    std::unique_ptr<TFBoundedIntParam> size;
    std::vector<std::unique_ptr<TFParam>> params;
    std::string base_type_str;
};

class TFVectorParam: public TFUnfixedArrayParam {
  public:
    TFVectorParam(std::string name_, std::vector<std::unique_ptr<TFParam>> params_);
    TFVectorParam(std::string name_, std::string base_type_str_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
  private:
    std::string base_type_str;
};

class TFFixedArrayParam: public TFParam {
  public:
    TFFixedArrayParam(
      TFParamKind kind_,
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TFParam>> params_);

    virtual std::string type() const override = 0;
    virtual std::string var() const override;
    virtual std::string initializer() const override = 0;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;
  protected:
    size_t size;
    std::vector<std::unique_ptr<TFParam>> params;
};

class TFArrayParam: public TFFixedArrayParam {
  public:
    TFArrayParam(
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TFParam>> params_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_initialization() const override;

    static bool classof(const TFParam *param);

    std::string base_type() const;
};

class TFTupleParam: public TFFixedArrayParam {
  public:
    TFTupleParam(
      std::string name_,
      size_t size_,
      std::vector<std::unique_ptr<TFParam>> params_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFPairParam: public TFFixedArrayParam {
  public:
    TFPairParam(
      std::string name_,
      std::vector<std::unique_ptr<TFParam>> params_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    static bool classof(const TFParam *param);
};

class TFArraySliceParam: public TFParam {
  public:
    TFArraySliceParam(std::string name_, std::vector<std::unique_ptr<TFParam>> params_);
    TFArraySliceParam(std::string name_, std::string base_type_str_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    //virtual bool stable() const override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFBoundedIntParam> size;
    std::unique_ptr<TFArrayParam> array;
    std::string base_type_str;
};

class TFInputListParam: public TFParam {
  public:
    TFInputListParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFArraySliceParam> inputlist;
};

class TFPartialTensorShapeParam: public TFParam {
  public:
    TFPartialTensorShapeParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFBoundedIntParam> rank;
    std::vector<std::unique_ptr<TFIntParam>> dims;
};

class TFTensorParam: public TFParam {
  public:
    TFTensorParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFDtypeParam> dtype;
    std::unique_ptr<TFBoundedIntParam> rank;
    std::vector<std::unique_ptr<TFIntParam>> dims;
};

class TFInputParam: public TFParam {
  public:
    TFInputParam(std::string name_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::unique_ptr<TFDtypeParam> dtype;
    std::unique_ptr<TFBoundedIntParam> rank;
    std::unique_ptr<TFIntParam> intval;
    std::unique_ptr<TFFloatParam> floatval1;
    std::unique_ptr<TFFloatParam> floatval2;
    std::unique_ptr<TFBoolParam> boolval;
    std::unique_ptr<TFBoundedIntParam> intvec_size;
    std::vector<std::unique_ptr<TFIntParam>> intvec_base;
    std::vector<std::unique_ptr<TFIntParam>> dims;
};

class TFAPIAttrsParam: public TFParam {
  public:
    TFAPIAttrsParam(
      std::string name_,
      std::string api_attrs_class_name_,
      std::vector<std::tuple<std::string, std::unique_ptr<TFParam>>> setters_);

    virtual std::string type() const override;
    virtual std::string var() const override;
    virtual std::string initializer() const override;

    virtual std::vector<std::string> gen_arg_setup() const override;
    virtual std::vector<std::string> gen_hard_constraint() const override;
    virtual std::vector<std::string> gen_soft_constraint() const override;
    virtual std::vector<std::string> gen_input_pass_condition() const override;
    virtual std::vector<std::string> gen_arg_initialization() const override;
    virtual void resolve_name_conflict(std::set<std::string>& names_seen) override;

    static bool classof(const TFParam *param);
  private:
    std::string api_attrs_class_name;
    std::vector<std::tuple<std::string, std::unique_ptr<TFParam>>> setters;

    std::vector<std::string> gen_api_attrs_init() const;
};


#endif
