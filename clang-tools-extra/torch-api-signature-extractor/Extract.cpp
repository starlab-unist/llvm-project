#include "Extract.h"
#include "Utils.h"
#include <iostream>
#include <fstream>

bool option_class_done;

std::unique_ptr<TorchType> extractTorchBuiltin(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "char" || t == "short" || t == "int" || t == "long") {
        torch_type = std::make_unique<TorchIntType>(t);
      } else if (startswith(t, "unsigned")) {
        torch_type = std::make_unique<TorchUnsignedIntType>();
      } else if (t == "bool") {
        torch_type = std::make_unique<TorchBoolType>();
      }
    } else if (builtin->isFloatingPoint()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "float") {
        torch_type = std::make_unique<TorchFloatType>();
      } else if (t == "double") {
        torch_type = std::make_unique<TorchDoubleType>();
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchString(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        std::string type_name = rtype->getDecl()->getNameAsString();
        if (type_name == "basic_string") {
          torch_type = std::make_unique<TorchStringType>(false);
        } else if (type_name == "basic_string_view") {
          torch_type = std::make_unique<TorchStringType>(true);
        }
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchMemoryFormat(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "MemoryFormat")
      torch_type = std::make_unique<TorchMemoryFormatType>();

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchLayout(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "Layout")
      torch_type = std::make_unique<TorchLayoutType>();

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchDevice(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* rtype = t->getAs<RecordType>())
    if (rtype->getDecl()->getQualifiedNameAsString() == "c10::Device")
      torch_type = std::make_unique<TorchDeviceType>();

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchDtype(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "ScalarType")
      torch_type = std::make_unique<TorchDtypeType>();

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchEnum(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;
  static const std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0)
      torch_type =
        std::make_unique<TorchEnumType>(
          qualified_name.substr(enum_prefix.size(), qualified_name.size() - enum_prefix.size()));
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchVector(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          assert(targs.size() == 1);
          std::unique_ptr<TorchType> value_type =
            extractTorchType(targs[0].getAsType(), Ctx);
          if (value_type == nullptr) {
            std::cerr <<
              "WARNING: While extracting `vector` type, failed to extract elem type `" <<
              targs[0].getAsType().getAsString() << "`." << std::endl;
            torch_type = std::make_unique<TorchVectorType>(targs[0].getAsType().getAsString());
          } else {
            torch_type = std::make_unique<TorchVectorType>(std::move(value_type));
          }
        }
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchTensor(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor")
          torch_type = std::make_unique<TorchTensorType>();
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor")
        torch_type = std::make_unique<TorchTensorType>();
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchScalar(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  const RecordType* rtype;
    if ((rtype = dyn_cast<RecordType>(t)) && rtype->getDecl()->getNameAsString() == "Scalar") {
      torch_type = std::make_unique<TorchScalarType>();
    }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchArrayRef(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      if (ctsdecl->getNameAsString() == "ArrayRef") {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type); 
        std::unique_ptr<TorchType> value_type = extractTorchType(targ[0].getAsType(), Ctx);
        if (value_type == nullptr) {
          std::cerr <<
            "WARNING: While extracting `ArrayRef` type , failed to extract elem type `" <<
            targ[0].getAsType().getAsString() << "`." << std::endl;
          torch_type = std::make_unique<TorchArrayRefType>(targ[0].getAsType().getAsString());
        } else {
          torch_type = std::make_unique<TorchArrayRefType>(std::move(value_type));
        }
      }
    }
  }
  
  return torch_type;
}

std::unique_ptr<TorchType> extractTorchOptionalArrayRef(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "OptionalArrayRef") {
      auto targ = tstype->template_arguments();
      assert(targ.size() == 1);
      std::unique_ptr<TorchType> value_type = extractTorchType(targ[0].getAsType(), Ctx);
      if (value_type == nullptr) {
        std::cerr <<
          "WARNING: While extracting `OptionalArrayRef` type, failed to extract elem type `" <<
          targ[0].getAsType().getAsString() << "`." << std::endl;
        torch_type =
          std::make_unique<TorchOptionalArrayRefType>(targ[0].getAsType().getAsString());
      } else {
        torch_type =
          std::make_unique<TorchOptionalArrayRefType>(std::move(value_type));
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchExpandingArray(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

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
      std::unique_ptr<TorchType> value_type;
      if (targ.size() == 1) {
        value_type = std::make_unique<TorchIntType>("long");
      } else if (targ.size() == 2) {
        value_type = extractTorchType(targ[1].getAsType(), Ctx);
        if (value_type == nullptr)
          return nullptr;
      }
      torch_type =
        std::make_unique<TorchExpandingArrayType>(expandingarray_size, std::move(value_type));
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
          std::unique_ptr<TorchType> value_type = extractTorchType(targ[1].getAsType(), Ctx);
          if (value_type == nullptr)
            return nullptr;
          torch_type =
            std::make_unique<TorchExpandingArrayType>(expandingarray_size, std::move(value_type));
        }
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchExpandingArrayWithOptionalElem(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

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
      std::unique_ptr<TorchType> value_type = std::make_unique<TorchIntType>("long");
      torch_type =
        std::make_unique<TorchExpandingArrayWithOptionalElemType>(
          expandingarray_size, std::move(value_type));
    }
  } else if ((sttptype = dyn_cast<SubstTemplateTypeParmType>(t))) {
    assert(sttptype->isSugared());
    if (const auto* rtype2 = dyn_cast<RecordType>(sttptype->getReplacementType())) {
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype2->getDecl())) {
        if (ctsdecl->getNameAsString() == "ExpandingArrayWithOptionalElem") {
          const auto& targ = ctsdecl->getTemplateArgs();
          assert(targ.size() == 2);
          assert(targ[0].getKind() == TemplateArgument::ArgKind::Integral);
          int64_t expandingarray_size = targ[0].getAsIntegral().getExtValue();
          assert(expandingarray_size > 0);
          std::unique_ptr<TorchType> value_type =
            extractTorchType(targ[1].getAsType(), Ctx);
          if (value_type == nullptr)
            return nullptr;
          torch_type =
            std::make_unique<TorchExpandingArrayWithOptionalElemType>(
              expandingarray_size, std::move(value_type));
        }
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchTuple(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "tuple") {
      auto targs = tstype->template_arguments();
      std::vector<std::unique_ptr<TorchType>> ttypes;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TorchType> ttype =
          extractTorchType(targs[i].getAsType(), Ctx);
        if (ttype == nullptr)
          return nullptr;
        ttypes.push_back(std::move(ttype));
      }
      assert(ttypes.size() > 0);
      torch_type =
        std::make_unique<TorchTupleType>(std::move(ttypes));
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchPair(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "pair") {
      auto targs = tstype->template_arguments();
      assert(targs.size() == 2);
      std::vector<std::unique_ptr<TorchType>> ttypes;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TorchType> ttype =
          extractTorchType(targs[i].getAsType(), Ctx);
        if (ttype == nullptr)
          return nullptr;
        ttypes.push_back(std::move(ttype));
      }
      assert(ttypes.size() > 0);
      torch_type =
        std::make_unique<TorchPairType>(std::move(ttypes));
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchSymint(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (rtype->getDecl()->getNameAsString() == "SymInt")
      torch_type = std::make_unique<TorchSymIntType>();
  }
  
  return torch_type;
}

std::unique_ptr<TorchType> extractTorchOptional(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "optional") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          std::unique_ptr<TorchType> value_type =
            extractTorchType(targ[0].getAsType(), Ctx);
          if (value_type == nullptr) {
            std::cerr <<
              "WARNING: While extracting `optional` type, failed to extract base type `" <<
              targ[0].getAsType().getAsString() << "`." << std::endl;
            torch_type = std::make_unique<TorchOptionalType>(targ[0].getAsType().getAsString());
          } else {
            torch_type = std::make_unique<TorchOptionalType>(std::move(value_type));
          }
        }
      }
    }
  }

  return torch_type;
}

std::unique_ptr<TorchType> extractTorchVariant(clang::QualType t, ASTContext &Ctx) {
  std::unique_ptr<TorchType> torch_type;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "variant") {
          auto targs = tstype->template_arguments();
          std::vector<std::unique_ptr<TorchType>> ttypes;
          for (size_t i = 0; i < targs.size(); i++) {
            std::unique_ptr<TorchType> ttype =
              extractTorchType(targs[i].getAsType(), Ctx);
            if (ttype == nullptr)
              return nullptr;
            ttypes.push_back(std::move(ttype));
          }
          assert(ttypes.size() > 0);
          torch_type = std::make_unique<TorchVariantType>(std::move(ttypes));
        }
      }
    }
  }

  return torch_type;
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

clang::QualType get_base(clang::QualType t, ASTContext &Ctx) {
  clang::QualType before = t;
  clang::QualType after;

  while (true) {
    after =
      before
        .getUnqualifiedType()
        .getDesugaredType(Ctx)
        .getNonReferenceType();
    if (before == after)
      break;
    before = after;
  }

  return after;
}

std::unique_ptr<TorchType> extractTorchAPIOptions(clang::QualType t, ASTContext &Ctx) {
  if (option_class_done) return nullptr;

  static std::string option_suffix = "Options";
  std::string class_name;
  const RecordType* rtype;
  if (const auto* tdtype = dyn_cast<TypedefType>(t)) {
    class_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (class_name.length() > option_suffix.length() &&
        class_name.compare(
          class_name.length() - option_suffix.length(),
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
    class_name = rtype->getDecl()->getQualifiedNameAsString();
    if (class_name.length() > option_suffix.length() &&
        class_name.compare(
          class_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      option_class_done = true;
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl()))
        class_name = get_specialized_name(ctsdecl);
    }
  }

  if (rtype == nullptr || !option_class_done) return nullptr;

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  assert(cdecl != nullptr);

  std::vector<TorchParam> ctor_params;
  std::vector<TorchParam> member_params;
  std::set<std::string> ctor_param_names;
  std::set<std::string> member_params_seen;

  bool default_ctor_available = false;
  for (auto ctordecl: cdecl->ctors())
    if (ctordecl->isDefaultConstructor())
      default_ctor_available = true;

  if (!default_ctor_available) {
    for (auto ctordecl: cdecl->ctors()) {
      bool is_special_ctor =
        ctordecl->isDefaultConstructor() ||
        ctordecl->isCopyOrMoveConstructor() ||
        ctordecl->isSpecializationCopyingObject() ||
        ctordecl->isInheritingConstructor();
      if (!is_special_ctor) {
        std::vector<TorchParam> ctor_params_;
        std::set<std::string> ctor_param_names_;
        bool extract_succeed = true;
        for (const auto* param: ctordecl->parameters()) {
          std::string param_name = param->getNameAsString();
          std::unique_ptr<TorchType> ctor_param_type = extractTorchType(param->getType(), Ctx);
          if (ctor_param_type == nullptr) {
            std::cerr <<
              "WARNING: Parsing fail on param `" << param_name << "` in `" << class_name << "`.\n" <<
              "         Type `" << param->getType().getAsString() << "` is not supported." << std::endl;
            extract_succeed = false;
            break;
          }
          ctor_param_type->set_default(get_default_expr(param_name, cdecl));
          ctor_params_.push_back({param_name, std::move(ctor_param_type)});
          ctor_param_names_.insert(param_name);
        }
        if (extract_succeed) {
          ctor_params = std::move(ctor_params_);
          ctor_param_names = ctor_param_names_;
          break;
        }
      }
    }
    if (ctor_params.empty())
      return nullptr;
  }

  for (auto method: cdecl->methods()) {
    if (dyn_cast<CXXConstructorDecl>(method) != nullptr)
      continue;

    std::string param_name = method->getNameAsString();
    if (ctor_param_names.find(param_name) != ctor_param_names.end())
      continue;
    if (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator())
      continue;
    if (method->parameters().size() != 1)
      continue;
    if (get_base(method->getReturnType(), Ctx) != get_base(t, Ctx))
      continue;

    std::unique_ptr<TorchType> param_type = extractTorchType(method->parameters()[0]->getType(), Ctx);
    if (param_type == nullptr) {
      std::cerr <<
        "WARNING: Parsing fail on param `" << param_name << "` in `" << class_name << "`.\n" <<
        "         Type `" << method->parameters()[0]->getType().getAsString() << "` is not supported." << std::endl;
      continue;
    }
    param_type->set_default(get_default_expr(param_name, cdecl));

    bool insert = true;
    for (size_t i = 0; i < member_params.size(); i++) {
      if (member_params[i].first == param_name) {
        insert = false;
        if (!member_params[i].second->precise()) {
          member_params[i] = {param_name, std::move(param_type)};
          break;
        }
      }
    }
    if (insert)
      member_params.push_back({param_name, std::move(param_type)});
  }
  std::unique_ptr<TorchType> torch_type =
    std::make_unique<TorchAPIOptionsType>(
      class_name,
      std::move(ctor_params),
      std::move(member_params));
  return torch_type;
}

std::unique_ptr<TorchType> extractTorchType(clang::QualType t, ASTContext &Ctx) {
  // Base case
  if (auto builtin_param = extractTorchBuiltin(t, Ctx))
    return builtin_param;
  if (auto string_param = extractTorchString(t, Ctx))
    return string_param;
  if (auto memory_format_param = extractTorchMemoryFormat(t, Ctx))
    return memory_format_param;
  if (auto layout_param = extractTorchLayout(t, Ctx))
    return layout_param;
  if (auto device_param = extractTorchDevice(t, Ctx))
    return device_param;
  if (auto dtype_param = extractTorchDtype(t, Ctx))
    return dtype_param;
  if (auto enum_param = extractTorchEnum(t, Ctx))
    return enum_param;
  if (auto vector_param = extractTorchVector(t, Ctx))
    return vector_param;
  if (auto tensor_param = extractTorchTensor(t, Ctx))
    return tensor_param;
  if (auto scalar_param = extractTorchScalar(t, Ctx))
    return scalar_param;
  if (auto arrayref_param = extractTorchArrayRef(t, Ctx))
    return arrayref_param;
  if (auto optional_array_param = extractTorchOptionalArrayRef(t, Ctx))
    return optional_array_param;
  if (auto expandingarray_param = extractTorchExpandingArray(t, Ctx))
    return expandingarray_param;
  if (auto expandingarraywithoptionalelem_param = extractTorchExpandingArrayWithOptionalElem(t, Ctx))
    return expandingarraywithoptionalelem_param;
  if (auto tuple_param = extractTorchTuple(t, Ctx))
    return tuple_param;
  if (auto pair_param = extractTorchPair(t, Ctx))
    return pair_param;
  if (auto symint_param = extractTorchSymint(t, Ctx))
    return symint_param;

  // Recursive case
  if (auto optional_param = extractTorchOptional(t, Ctx))
    return optional_param;
  if (auto variant_param = extractTorchVariant(t, Ctx))
    return variant_param;
  if (auto api_options_param = extractTorchAPIOptions(t, Ctx))
    return api_options_param;

  // Simplifying sugars
  if (t->isLValueReferenceType()) {
    return extractTorchType(t->getAs<LValueReferenceType>()->getPointeeType(), Ctx);
  } else if (t->isRValueReferenceType()) {
    return extractTorchType(t->getAs<RValueReferenceType>()->getPointeeType(), Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      return extractTorchType(elaborated->desugar(), Ctx);
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    if (tdtype->isSugared())
      return extractTorchType(tdtype->desugar(), Ctx);
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared())
      return extractTorchType(tstype->desugar(), Ctx);
  }

  return nullptr;
}
