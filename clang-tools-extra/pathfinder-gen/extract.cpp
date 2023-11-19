#include "extract.h"
#include "utils.h"
#include <iostream>
#include <fstream>

bool option_class_done;

std::unique_ptr<TorchParam> extractTorchBuiltin(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "char" || t == "short" || t == "int" || t == "long") {
        torch_param = std::make_unique<TorchIntParam>(name);
      } else if (startswith(t, "unsigned")) {
        torch_param = std::make_unique<TorchUnsignedIntParam>(name);
      } else if (t == "bool") {
        torch_param = std::make_unique<TorchBoolParam>(name);
      }
    } else if (builtin->isFloatingPoint()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "float") {
        torch_param = std::make_unique<TorchFloatParam>(name);
      } else if (t == "double") {
        torch_param = std::make_unique<TorchDoubleParam>(name);
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchDtype(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "ScalarType")
      torch_param = std::make_unique<TorchDtypeParam>(name);

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchEnum(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;
  static const std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0)
      torch_param =
        std::make_unique<TorchEnumParam>(
          name,
          qualified_name.substr(enum_prefix.size(), qualified_name.size() - enum_prefix.size()));
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchVector(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          if (targs.size() == 1) {
            std::vector<std::unique_ptr<TorchParam>> params;
            for (size_t i = 0; i < MAX_VECTOR_SIZE; i++) {
              std::string param_name = name + "_" + std::to_string(i);
              std::unique_ptr<TorchParam> param =
                extractTorchParam(targs[0].getAsType(), param_name, Ctx);
              if (param == nullptr)
                return nullptr;
              params.push_back(std::move(param));
            }
            torch_param = std::make_unique<TorchVectorParam>(name, std::move(params));
          } else {
            assert(false);
          }
        }
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchTensor(clang::QualType t, std::string name,  ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor")
          torch_param = std::make_unique<TorchTensorParam>(name);
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor")
        torch_param = std::make_unique<TorchTensorParam>(name);
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchScalar(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
    if ((rtype = dyn_cast<RecordType>(t)) && rtype->getDecl()->getNameAsString() == "Scalar") {
      torch_param = std::make_unique<TorchScalarParam>(name);
    }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchArrayRef(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      if (ctsdecl->getNameAsString() == "ArrayRef") {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type); 
        std::vector<std::unique_ptr<TorchParam>> params;
        for (size_t i = 0; i < MAX_ARRAYREF_SIZE; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TorchParam> param =
            extractTorchParam(targ[0].getAsType(), param_name, Ctx);
          if (param == nullptr)
            return nullptr;
          params.push_back(std::move(param));
        }
        torch_param = std::make_unique<TorchArrayRefParam>(name, std::move(params));
      }
    }
  }
  
  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchOptionalArrayRef(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "OptionalArrayRef") {
      auto targ = tstype->template_arguments();
      assert(targ.size() == 1);
      std::vector<std::unique_ptr<TorchParam>> params;
      for (size_t i = 0; i < MAX_ARRAYREF_SIZE; i++) {
        std::unique_ptr<TorchParam> param =
          extractTorchParam(targ[0].getAsType(), name + "_" + std::to_string(i), Ctx);
        if (param == nullptr)
          return nullptr;
        params.push_back(std::move(param));
      }
      std::unique_ptr<TorchParam> arrayref =
        std::make_unique<TorchArrayRefParam>(name + "_arrayref", std::move(params));
      torch_param = std::make_unique<TorchOptionalParam>(name, std::move(arrayref));
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchExpandingArray(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  const SubstTemplateTypeParmType* sttptype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "ExpandingArray") {
      auto targ = tstype->template_arguments();
      if (targ.size() != 1 && targ.size() != 2)
        return nullptr;
      int64_t expandingarray_size =
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
      assert(expandingarray_size > 0);
      std::vector<std::unique_ptr<TorchParam>> params;
      if (targ.size() == 1){
        for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TorchParam> param = std::make_unique<TorchIntParam>(param_name);
          params.push_back(std::move(param));
        }
      }
      else if (targ.size() == 2){
        for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TorchParam> param =
            extractTorchParam(targ[1].getAsType(), param_name, Ctx);
          if (param == nullptr)
            return nullptr;
          params.push_back(std::move(param));
        }
      }
      torch_param =
        std::make_unique<TorchExpandingArrayParam>(name, expandingarray_size, std::move(params));
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
          assert(expandingarray_size > 0);
          std::vector<std::unique_ptr<TorchParam>> params;
          for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
            std::string param_name = name + "_" + std::to_string(i);
            std::unique_ptr<TorchParam> param =
              extractTorchParam(targ[1].getAsType(), param_name, Ctx);
            if (param == nullptr)
              return nullptr;
            params.push_back(std::move(param));
          }
          torch_param =
            std::make_unique<TorchExpandingArrayParam>(name, expandingarray_size, std::move(params));
        }
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchExpandingArrayWithOptionalElem(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  const SubstTemplateTypeParmType* sttptype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "ExpandingArrayWithOptionalElem") {
      auto targ = tstype->template_arguments();
      if (targ.size() != 1)
        return nullptr;
      int64_t expandingarray_size =
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
      assert(expandingarray_size > 0);
      std::vector<std::unique_ptr<TorchParam>> params;
      for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
        std::string param_name = name + "_" + std::to_string(i);
        std::unique_ptr<TorchParam> param = std::make_unique<TorchIntParam>(param_name);
        std::unique_ptr<TorchParam> optional_param =
          std::make_unique<TorchOptionalParam>(param_name, std::move(param));
        params.push_back(std::move(optional_param));
      }
      torch_param =
        std::make_unique<TorchExpandingArrayWithOptionalElemParam>(
          name, expandingarray_size, std::move(params));
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
          assert(expandingarray_size > 0);
          std::vector<std::unique_ptr<TorchParam>> params;
          for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
            std::string param_name = name + "_" + std::to_string(i);
            std::unique_ptr<TorchParam> param =
              extractTorchParam(targ[1].getAsType(), param_name + "_base", Ctx);
            if (param == nullptr)
              return nullptr;
            std::unique_ptr<TorchParam> optional_param =
              std::make_unique<TorchOptionalParam>(param_name, std::move(param));
            params.push_back(std::move(optional_param));
          }
          torch_param =
            std::make_unique<TorchExpandingArrayWithOptionalElemParam>(
              name, expandingarray_size, std::move(params));
        }
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchTuple(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "tuple") {
      auto targs = tstype->template_arguments();
      int64_t tuple_size = targs.size();
      std::vector<std::unique_ptr<TorchParam>> params;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TorchParam> param =
          extractTorchParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
        if (param == nullptr)
          return nullptr;
        params.push_back(std::move(param));
      }
      assert(params.size() > 0);
      torch_param =
        std::make_unique<TorchTupleParam>(name, tuple_size, std::move(params));
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchPair(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "pair") {
      auto targs = tstype->template_arguments();
      assert(targs.size() == 2);
      std::vector<std::unique_ptr<TorchParam>> params;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TorchParam> param =
          extractTorchParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
        if (param == nullptr)
          return nullptr;
        params.push_back(std::move(param));
      }
      assert(params.size() > 0);
      torch_param =
        std::make_unique<TorchPairParam>(name, std::move(params));
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchOptional(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "optional") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          std::unique_ptr<TorchParam> param =
            extractTorchParam(targ[0].getAsType(), name + "_base", Ctx);
          if (param == nullptr)
            return std::make_unique<TorchOptionalParam>(name, std::move(std::make_unique<TorchNullParam>(name + "_null")));
          torch_param = std::make_unique<TorchOptionalParam>(name, std::move(param));
        }
      }
    }
  }

  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchVariant(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TorchParam> torch_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "variant") {
          auto targs = tstype->template_arguments();
          std::vector<std::unique_ptr<TorchParam>> params;
          for (size_t i = 0; i < targs.size(); i++) {
            std::unique_ptr<TorchParam> param =
              extractTorchParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
            if (param == nullptr)
              return nullptr;
            params.push_back(std::move(param));
          }
          assert(params.size() > 0);
          torch_param = std::make_unique<TorchVariantParam>(name, std::move(params));
        }
      }
    }
  }

  return torch_param;
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

std::unique_ptr<TorchParam> extractTorchAPIOptions(clang::QualType t, std::string name, ASTContext &Ctx) {
  if (option_class_done) return nullptr;

  static std::string option_suffix = "Options";
  std::string api_options_class_name;
  const RecordType* rtype;
  if (const auto* tdtype = dyn_cast<TypedefType>(t)) {
    api_options_class_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (api_options_class_name.length() > option_suffix.length() &&
        api_options_class_name.compare(
          api_options_class_name.length() - option_suffix.length(),
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
    api_options_class_name = rtype->getDecl()->getQualifiedNameAsString();
    if (api_options_class_name.length() > option_suffix.length() &&
        api_options_class_name.compare(
          api_options_class_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      option_class_done = true;
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl()))
        api_options_class_name = get_specialized_name(ctsdecl);
    }
  }

  if (rtype == nullptr || !option_class_done) return nullptr;

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  assert(cdecl != nullptr);

  std::vector<std::unique_ptr<TorchParam>> ctor_params;
  std::vector<std::unique_ptr<TorchParam>> member_params;
  std::set<std::string> ctor_param_names;
  std::set<std::string> ctor_params_seen;
  std::set<std::string> member_params_seen;

  for (auto method: cdecl->methods()) {
    if (const auto* cxxconstructordecl = dyn_cast<CXXConstructorDecl>(method)) {
      bool is_special_ctor =
        cxxconstructordecl->isDefaultConstructor() ||
        cxxconstructordecl->isCopyOrMoveConstructor() ||
        cxxconstructordecl->isSpecializationCopyingObject() ||
        cxxconstructordecl->isInheritingConstructor();
      if (!is_special_ctor)
        for (const auto* param: cxxconstructordecl->parameters())
          ctor_param_names.insert(param->getNameAsString());
      continue;
    }
    std::string param_name = method->getNameAsString();
    if (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator())
      continue;
    if (method->parameters().size() != 1)
      continue;
    bool duplicate = false;
    for (auto&& p: member_params) {
      if (p->get_name() == param_name) {
        duplicate = true;
        break;
      }
    }
    if (duplicate)
      continue;

    std::unique_ptr<TorchParam> param = extractTorchParam(method->parameters()[0]->getType(), param_name, Ctx);
    if (param == nullptr) {
      std::cerr <<
        "WARNING: Parsing fail on param `" << param_name << "` in `" << api_options_class_name << "`.\n" <<
        "         Type `" << method->parameters()[0]->getType().getAsString() << "` is not supported." << std::endl;
      continue;
    }
    param->set_default(get_default_expr(param_name, cdecl));

    if (ctor_param_names.find(param_name) != ctor_param_names.end()) {
      if (ctor_params_seen.find(param_name) == ctor_params_seen.end()) {
        ctor_params.push_back(std::move(param));
        ctor_params_seen.insert(param_name);
      }
    } else {
      if (member_params_seen.find(param_name) == member_params_seen.end()) {
        member_params.push_back(std::move(param));
        member_params_seen.insert(param_name);
      }
    }
  }
  std::unique_ptr<TorchParam> torch_param =
    std::make_unique<TorchAPIOptionsParam>(
      name,
      api_options_class_name,
      std::move(ctor_params),
      std::move(member_params));
  return torch_param;
}

std::unique_ptr<TorchParam> extractTorchParam(clang::QualType t, std::string name, ASTContext &Ctx) {
  // Base case
  if (auto builtin_param = extractTorchBuiltin(t, name, Ctx))
    return builtin_param;
  if (auto dtype_param = extractTorchDtype(t, name, Ctx))
    return dtype_param;
  if (auto enum_param = extractTorchEnum(t, name, Ctx))
    return enum_param;
  if (auto vector_param = extractTorchVector(t, name, Ctx))
    return vector_param;
  if (auto tensor_param = extractTorchTensor(t, name, Ctx))
    return tensor_param;
  if (auto scalar_param = extractTorchScalar(t, name, Ctx))
    return scalar_param;
  if (auto arrayref_param = extractTorchArrayRef(t, name, Ctx))
    return arrayref_param;
  if (auto optional_array_param = extractTorchOptionalArrayRef(t, name, Ctx))
    return optional_array_param;
  if (auto expandingarray_param = extractTorchExpandingArray(t, name, Ctx))
    return expandingarray_param;
  if (auto expandingarraywithoptionalelem_param = extractTorchExpandingArrayWithOptionalElem(t, name, Ctx))
    return expandingarraywithoptionalelem_param;
  if (auto tuple_param = extractTorchTuple(t, name, Ctx))
    return tuple_param;
  if (auto pair_param = extractTorchPair(t, name, Ctx))
    return pair_param;

  // Recursive case
  if (auto optional_param = extractTorchOptional(t, name, Ctx))
    return optional_param;
  if (auto variant_param = extractTorchVariant(t, name, Ctx))
    return variant_param;
  if (auto api_options_param = extractTorchAPIOptions(t, name, Ctx))
    return api_options_param;

  // Simplifying sugars
  if (t->isLValueReferenceType()) {
    return extractTorchParam(t->getAs<LValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (t->isRValueReferenceType()) {
    return extractTorchParam(t->getAs<RValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      return extractTorchParam(elaborated->desugar(), name, Ctx);
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    if (tdtype->isSugared())
      return extractTorchParam(tdtype->desugar(), name, Ctx);
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared())
      return extractTorchParam(tstype->desugar(), name, Ctx);
  }

  return nullptr;
}
