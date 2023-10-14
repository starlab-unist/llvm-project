#include "parse.h"
#include "param.h"
#include "utils.h"
#include <iostream>
#include <fstream>

bool option_class_done;

std::unique_ptr<TorchParam> parseBuiltin(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "int" || t == "long") {
        torch_param = std::make_unique<TorchIntParam>(name);
      } else if (t == "bool") {
        torch_param = std::make_unique<TorchBoolParam>(name);
      } else {
        assert(false);
      }
    } else if (builtin->isFloatingPoint()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "float" || t == "double") {
        torch_param = std::make_unique<TorchFloatParam>(name);
      } else {
        assert(false);
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> parseDtype(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "ScalarType")
      torch_param = std::make_unique<TorchDtypeParam>(name);

  return torch_param;
}

std::unique_ptr<TorchParam> parseEnum(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;
  static std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0)
      torch_param = std::make_unique<TorchParam>(ENUM, rtype->getDecl()->getNameAsString());
  }

  return torch_param;
}

Optional<std::unique_ptr<TorchParam>> parseVector(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          if (targs.size() == 1) {
            if (auto p = parseTorchParam(targs[0].getAsType(), "", Ctx)) {
              if (p->ptype == INT)
                return std::make_unique<TorchParam>(INTVECTOR, name);
              if (p->ptype == FLOAT)
                return std::make_unique<TorchParam>(FLOATVECTOR, name);
            }
          }
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<TorchParam>> parseTensor(clang::QualType t, std::string name) {
  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor")
          return std::make_unique<TorchParam>(TENSOR, name);
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor")
        return std::make_unique<TorchParam>(TENSOR, name);
  }

  return None;
}

Optional<std::unique_ptr<TorchParam>> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx) {
  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      if (ctsdecl->getNameAsString() == "ArrayRef") {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type);
        auto p = parseTorchParam(targ[0].getAsType(), "",  Ctx);
        if (p->ptype == INT)
          return std::make_unique<TorchParam>(INTARRAYREF, name);
      }
    }
  }
  
  return None;
}

Optional<std::unique_ptr<TorchParam>> parseExpandingArray(clang::QualType t, std::string name, ASTContext &Ctx) {
  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  const SubstTemplateTypeParmType* sttptype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "ExpandingArray") {
      auto targ = tstype->template_arguments();
      if (targ.size() != 1)
        return None;
      int64_t expandingarray_size =
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
      return std::make_unique<TorchParam>(EXPANDINGARRAY, name, expandingarray_size);
    }
  } else if ((sttptype = dyn_cast<SubstTemplateTypeParmType>(t))) {
    assert(sttptype->isSugared());
    if (const auto* rtype2 = dyn_cast<RecordType>(sttptype->getReplacementType())) {
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype2->getDecl())) {
        if (ctsdecl->getNameAsString() == "ExpandingArray") {
          const auto& targ = ctsdecl->getTemplateArgs();
          assert(targ.size() == 2);
          assert(targ[0].getKind() == TemplateArgument::ArgKind::Integral);
          int64_t expandingarray_size =
            targ[0].getAsIntegral().getExtValue();
          return std::make_unique<TorchParam>(EXPANDINGARRAY, name, expandingarray_size);
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<TorchParam>> parseExpandingArrayWithOptionalElem(clang::QualType t, std::string name, ASTContext &Ctx) {
  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  const SubstTemplateTypeParmType* sttptype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "ExpandingArrayWithOptionalElem") {
      auto targ = tstype->template_arguments();
      if (targ.size() != 1)
        return None;
      int64_t expandingarray_size =
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
      return std::make_unique<TorchParam>(EXPANDINGARRAYWITHOPTIONALELEM, name, expandingarray_size);
    }
  } else if ((sttptype = dyn_cast<SubstTemplateTypeParmType>(t))) {
    assert(sttptype->isSugared());
    if (const auto* rtype2 = dyn_cast<RecordType>(sttptype->getReplacementType())) {
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype2->getDecl())) {
        if (ctsdecl->getNameAsString() == "ExpandingArrayWithOptionalElem") {
          const auto& targ = ctsdecl->getTemplateArgs();
          assert(targ.size() == 2);
          assert(targ[0].getKind() == TemplateArgument::ArgKind::Integral);
          int64_t expandingarray_size =
            targ[0].getAsIntegral().getExtValue();
          return std::make_unique<TorchParam>(EXPANDINGARRAYWITHOPTIONALELEM, name, expandingarray_size);
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<TorchParam>> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx) {
  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "optional") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          auto p = parseTorchParam(targ[0].getAsType(), name, Ctx);
          if (p != nullptr)
            return std::make_unique<TorchParam>(OPTIONAL, name + "_opt", std::move(p));
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<TorchParam>> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx) {
  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "variant") {
          auto targs = tstype->template_arguments();
          std::vector<std::unique_ptr<TorchParam>> enums;
          std::unique_ptr<TorchParam> expandingarray;
          for (auto targ: targs) {
            auto p = parseTorchParam(targ.getAsType(), name + "_array", Ctx);
            if (p->ptype == ENUM) {
              enums.push_back(std::move(p));
            } else if (p->ptype == EXPANDINGARRAY || p->ptype == EXPANDINGARRAYWITHOPTIONALELEM) {
              expandingarray = std::move(p);
            } else {
              assert(false);
            }
          }
          assert(enums.size() > 0);
          return std::make_unique<TorchParam>(VARIANT, name, std::move(enums), std::move(expandingarray));
        }
      }
    }
  }

  return None;
}

std::string get_specialized_name(const ClassTemplateSpecializationDecl* ctsdecl) {
  std::string name = ctsdecl->getQualifiedNameAsString() + "<";

  const auto& targs = ctsdecl->getTemplateArgs();
  for (size_t i = 0; i < targs.size(); i++) {
    if (targs[i].getKind() == TemplateArgument::ArgKind::Type) {
      auto t = targs[i].getAsType();
      if (const auto* rtype = t->getAs<RecordType>()) {
        if (const auto* ctsdecl2 = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl()))
          name += get_specialized_name(ctsdecl2);
      } else if (t->getAs<BuiltinType>() != nullptr) {
        name += t.getAsString();
      } else {
        assert(false);
      }
    } else if (targs[i].getKind() == TemplateArgument::ArgKind::Integral) {
      name += std::to_string(targs[i].getAsIntegral().getExtValue());
    } else {
      assert(false);
    }

    if (i != targs.size()-1)
      name += ",";
  }
  name += ">";

  return name;
}

bool include(std::vector<std::pair<std::string,std::unique_ptr<TorchParam>>>& vec, std::string name) {
  for (auto&& p: vec) {
    if (p.first == name)
      return true;
  }
  return false;
}

void push_back_unique(std::vector<std::pair<std::string,std::unique_ptr<TorchParam>>>& vec, std::string name, std::unique_ptr<TorchParam> param) {
  if (!include(vec, name))
    vec.push_back({name, std::move(param)});
}

Expr* get_default_expr(std::string param_name, const CXXRecordDecl* cdecl) {
  for (auto field: cdecl->fields()) {
    std::string field_name = field->getNameAsString();
    if (param_name.length() + 1 == field_name.length() &&
        field_name.compare(0, field_name.length(), param_name + "_") == 0) {
      return field->getInClassInitializer()->IgnoreUnlessSpelledInSource();
    }
  }
  return nullptr;
}

Optional<std::unique_ptr<TorchParam>> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx) {
  if (option_class_done) return None;

  static std::string option_suffix = "Options";
  std::string api_option_name;
  const RecordType* rtype;
  if (const auto* tdtype = dyn_cast<TypedefType>(t)) {
    api_option_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      option_class_done = true;
      assert(tdtype->isSugared());

      const TemplateSpecializationType* tstype;
      if (const auto* tdtype2 = dyn_cast<TypedefType>(tdtype->desugar())) {
        assert(tdtype2->isSugared());
        tstype = dyn_cast<TemplateSpecializationType>(tdtype2->desugar());
      } else {
        tstype = dyn_cast<TemplateSpecializationType>(tdtype->desugar());
      }

      if (tstype != nullptr) {
        assert(tstype->isSugared());
        rtype = dyn_cast<RecordType>(tstype->desugar());
      } else {
        rtype = dyn_cast<RecordType>(tdtype->desugar());
      }
    }
  } else if ((rtype = dyn_cast<RecordType>(t))) {
    api_option_name = rtype->getDecl()->getQualifiedNameAsString();
    if (api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      option_class_done = true;
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl()))
        api_option_name = get_specialized_name(ctsdecl);
    }
  }

  if (rtype == nullptr || !option_class_done) return None;

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  assert(cdecl != nullptr);

  std::vector<std::pair<std::string,std::unique_ptr<TorchParam>>> ctor_params;
  std::vector<std::pair<std::string,std::unique_ptr<TorchParam>>> entries;
  std::vector<std::string> ctor_param_names;
  for (auto method: cdecl->methods()) {
    if (const auto* cxxconstructordecl = dyn_cast<CXXConstructorDecl>(method)) {
      bool is_special_ctor =
        cxxconstructordecl->isDefaultConstructor() ||
        cxxconstructordecl->isCopyOrMoveConstructor() ||
        cxxconstructordecl->isSpecializationCopyingObject() ||
        cxxconstructordecl->isInheritingConstructor();
      if (!is_special_ctor)
        for (const auto* param: cxxconstructordecl->parameters())
          ctor_param_names.push_back(param->getNameAsString());
      continue;
    }
    std::string param_name = method->getNameAsString();
    if (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator())
      continue;
    if (method->parameters().size() != 1)
      continue;
    bool duplicate = false;
    for (auto&& p: entries) {
      if (p.first == param_name) {
        duplicate = true;
        break;
      }
    }
    if (duplicate)
      continue;

    std::unique_ptr<TorchParam> param = parseTorchParam(method->parameters()[0]->getType(), param_name, Ctx);
    if (param != nullptr) {
      /* if (param->ptype == TENSOR)
        param = std::make_unique<TorchParam>(
          OPTIONAL,
          param_name + "_opt",
          std::move(param)); */
      param->set_default(get_default_expr(param_name, cdecl));

      if (include(ctor_param_names, param_name))
        push_back_unique(ctor_params, param_name, std::move(param));
      else
        push_back_unique(entries, param_name, std::move(param));
    }
  }
  return
    std::make_unique<TorchParam>(
      MAP,
      name,
      api_option_name,
      std::move(ctor_params),
      std::move(entries));
}

std::unique_ptr<TorchParam> parseTorchParam(clang::QualType t, std::string name, ASTContext &Ctx) {
  // Base case
  if (auto builtin_opt = parseBuiltin(t, name, Ctx))
    return std::move(builtin_opt.getValue());
  if (auto dtype_opt = parseDtype(t, name, Ctx))
    return std::move(dtype_opt.getValue());
  if (auto enum_opt = parseEnum(t, Ctx))
    return std::move(enum_opt.getValue());
  if (auto int_vec_opt = parseVector(t, name, Ctx))
    return std::move(int_vec_opt.getValue());
  if (auto tensor_opt = parseTensor(t, name))
    return std::move(tensor_opt.getValue());
  if (auto intarrayref_opt = parseIntArrayRef(t, name, Ctx))
    return std::move(intarrayref_opt.getValue());
  if (auto expandingarray_opt = parseExpandingArray(t, name, Ctx))
    return std::move(expandingarray_opt.getValue());
  if (auto expandingarraywithoptionalelem_opt = parseExpandingArrayWithOptionalElem(t, name, Ctx))
    return std::move(expandingarraywithoptionalelem_opt.getValue());

  // Recursive case
  if (auto optional_opt = parseOptional(t, name, Ctx))
    return std::move(optional_opt.getValue());
  if (auto variant_opt = parseVariant(t, name, Ctx))
    return std::move(variant_opt.getValue());
  if (auto api_option_opt = parseMAP(t, name, Ctx))
    return std::move(api_option_opt.getValue());

  // Simplifying sugars
  if (t->isLValueReferenceType()) {
    return parseTorchParam(t->getAs<LValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (t->isRValueReferenceType()) {
    return parseTorchParam(t->getAs<RValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      return parseTorchParam(elaborated->desugar(), name, Ctx);
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    if (tdtype->isSugared())
      return parseTorchParam(tdtype->desugar(), name, Ctx);
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared())
      return parseTorchParam(tstype->desugar(), name, Ctx);
  }

  return nullptr;
}
