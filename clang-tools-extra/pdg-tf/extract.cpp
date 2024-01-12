#include "extract.h"
#include "utils.h"
#include <iostream>
#include <fstream>

std::unique_ptr<TFParam> extractTFScope(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = t->getAs<RecordType>())
    if (rtype->getDecl()->getQualifiedNameAsString() == "tensorflow::Scope") {
      tf_param = std::make_unique<TFScopeParam>();
    }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFBuiltin(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "char" || t == "short" || t == "int" || t == "long") {
        tf_param = std::make_unique<TFIntParam>(name, t);
      } else if (startswith(t, "unsigned")) {
        tf_param = std::make_unique<TFUnsignedIntParam>(name);
      } else if (t == "bool") {
        tf_param = std::make_unique<TFBoolParam>(name);
      }
    } else if (builtin->isFloatingPoint()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "float") {
        tf_param = std::make_unique<TFFloatParam>(name);
      } else if (t == "double") {
        tf_param = std::make_unique<TFDoubleParam>(name);
      }
    }
  }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFStringPiece(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* tdtype = dyn_cast<TypedefType>(t))
    if (tdtype->getDecl()->getNameAsString() == "StringPiece") {
      if (name == "padding") {
        tf_param = std::make_unique<TFPaddingParam>(name);
      } else {
        tf_param = std::make_unique<TFStringPieceParam>(name);
      }
    }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFDtype(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "DataType")
      tf_param = std::make_unique<TFDtypeParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFVector(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          if (targs.size() == 1) {
            std::vector<std::unique_ptr<TFParam>> params;
            for (size_t i = 0; i < MAX_VECTOR_SIZE; i++) {
              std::string param_name = name + "_" + std::to_string(i);
              std::unique_ptr<TFParam> param =
                extractTFParam(targs[0].getAsType(), param_name, Ctx);
              if (param == nullptr)
                break;
              params.push_back(std::move(param));
            }
            if (params.empty()) {
              std::cerr <<
                "WARNING: While extracting `vector` param `" << name << "`, failed to extract base type `" <<
                targs[0].getAsType().getAsString() << "`.\n" <<
                "         param `" << name << "` will be fixed to empty vector." << std::endl;
              tf_param = std::make_unique<TFVectorParam>(name, targs[0].getAsType().getAsString());
            } else {
              tf_param = std::make_unique<TFVectorParam>(name, std::move(params));
            }
          } else {
            assert(false);
          }
        }
      }
    }
  }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFTensor(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = t->getAs<RecordType>())
    if (rtype->getDecl()->getQualifiedNameAsString() == "tensorflow::Input")
      tf_param = std::make_unique<TFTensorParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFArraySlice(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  const auto* tstype = dyn_cast<TemplateSpecializationType>(t);
  if (tstype == nullptr)
    return nullptr;
  const auto* rtype = dyn_cast<RecordType>(tstype->desugar());
  if (rtype == nullptr)
    return nullptr;
  if (rtype->getDecl()->getNameAsString() != "Span")
    return nullptr;

  auto targs = tstype->template_arguments();
  assert(targs.size() == 1);
  assert(targs[0].getKind() == TemplateArgument::ArgKind::Type);
  const auto* sttpt = dyn_cast<SubstTemplateTypeParmType>(targs[0].getAsType());
  if (sttpt == nullptr)
    return nullptr;

  const auto* tatd = dyn_cast<TypeAliasTemplateDecl>(sttpt->getAssociatedDecl());
  if (tatd->getNameAsString() != "ArraySlice")
    return nullptr;

  auto base_type = sttpt->getReplacementType();

  std::vector<std::unique_ptr<TFParam>> params;
  for (size_t i = 0; i < MAX_ARRAYREF_SIZE; i++) {
    std::string param_name = name + "_" + std::to_string(i);
    std::unique_ptr<TFParam> param =
      extractTFParam(base_type, param_name, Ctx);
    if (param == nullptr)
      break;
    params.push_back(std::move(param));
  }
  if (params.empty()) {
    std::cerr <<
      "WARNING: While extracting `ArraySlice` param `" << name << "`, failed to extract base type `" <<
      base_type.getAsString() << "`.\n" <<
      "         param `" << name << "` will be fixed to empty ArraySlice." << std::endl;
    tf_param = std::make_unique<TFArraySliceParam>(name, base_type.getAsString());
  } else {
    tf_param = std::make_unique<TFArraySliceParam>(name, std::move(params));
  }
  
  return tf_param;
}

std::unique_ptr<TFParam> extractTFTuple(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "tuple") {
      auto targs = tstype->template_arguments();
      int64_t tuple_size = targs.size();
      std::vector<std::unique_ptr<TFParam>> params;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TFParam> param =
          extractTFParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
        if (param == nullptr)
          return nullptr;
        params.push_back(std::move(param));
      }
      assert(params.size() > 0);
      tf_param =
        std::make_unique<TFTupleParam>(name, tuple_size, std::move(params));
    }
  }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFPair(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "pair") {
      auto targs = tstype->template_arguments();
      assert(targs.size() == 2);
      std::vector<std::unique_ptr<TFParam>> params;
      for (size_t i = 0; i < targs.size(); i++) {
        std::unique_ptr<TFParam> param =
          extractTFParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
        if (param == nullptr)
          return nullptr;
        params.push_back(std::move(param));
      }
      assert(params.size() > 0);
      tf_param =
        std::make_unique<TFPairParam>(name, std::move(params));
    }
  }

  return tf_param;
}

/* std::unique_ptr<TFParam> extractTFOptional(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "optional") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          std::unique_ptr<TFParam> param =
            extractTFParam(targ[0].getAsType(), name + "_base", Ctx);
          if (param == nullptr) {
            std::cerr <<
              "WARNING: While extracting `optional` param `" << name << "`, failed to extract base type `" <<
              targ[0].getAsType().getAsString() << "`.\n" <<
              "         param `" << name << "` will be fixed to `nullopt`." << std::endl;
            tf_param = std::make_unique<TFOptionalParam>(name, targ[0].getAsType().getAsString());
          } else {
            tf_param = std::make_unique<TFOptionalParam>(name, std::move(param));
          }
        }
      }
    }
  }

  return tf_param;
} */

std::unique_ptr<TFParam> extractTFVariant(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "variant") {
          auto targs = tstype->template_arguments();
          std::vector<std::unique_ptr<TFParam>> params;
          for (size_t i = 0; i < targs.size(); i++) {
            std::unique_ptr<TFParam> param =
              extractTFParam(targs[i].getAsType(), name + "_" + std::to_string(i), Ctx);
            if (param == nullptr)
              return nullptr;
            params.push_back(std::move(param));
          }
          assert(params.size() > 0);
          tf_param = std::make_unique<TFVariantParam>(name, std::move(params));
        }
      }
    }
  }

  return tf_param;
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

std::unique_ptr<TFParam> extractTFAPIAttrs(clang::QualType t, std::string name, ASTContext &Ctx) {
  static std::string attrs_suffix = "Attrs";
  std::string api_attrs_class_name;

  const auto* rtype = dyn_cast<RecordType>(t);
  if (rtype == nullptr)
    return nullptr;

  api_attrs_class_name = rtype->getDecl()->getQualifiedNameAsString();
  bool is_attrs =
    api_attrs_class_name.length() > attrs_suffix.length() &&
    api_attrs_class_name.compare(
      api_attrs_class_name.length() - attrs_suffix.length(),
      attrs_suffix.length(), attrs_suffix) == 0;
  if (!is_attrs)
    return nullptr;

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  if (cdecl == nullptr)
    return nullptr;

  //cdecl->dump();

  std::vector<std::tuple<std::string, std::unique_ptr<TFBoolParam>, std::unique_ptr<TFParam>>> setters;

  bool default_ctor_available = false;
  for (auto ctordecl: cdecl->ctors())
    if (ctordecl->isDefaultConstructor())
      default_ctor_available = true;

  if (!default_ctor_available)
    return nullptr;

  for (auto setter: cdecl->methods()) {
    if (dyn_cast<CXXConstructorDecl>(setter) != nullptr)
      continue;

    std::string setter_name = setter->getNameAsString();
    if (setter->isCopyAssignmentOperator() || setter->isMoveAssignmentOperator())
      continue;
    if (setter->parameters().size() != 1)
      continue;
    if (get_base(setter->getReturnType(), Ctx) != get_base(t, Ctx))
      continue;

    std::string dataformat_suffix = "Format";
    bool is_dataformat_param =
      setter_name.length() >= dataformat_suffix.length() &&
      setter_name.compare(
        setter_name.length() - dataformat_suffix.length(),
        dataformat_suffix.length(), dataformat_suffix) == 0;

    auto param_type = setter->parameters()[0]->getType();
    auto param_name = setter->parameters()[0]->getNameAsString();
    std::unique_ptr<TFParam> param = extractTFParam(param_type, param_name, Ctx);
    if (param == nullptr) {
      std::cerr <<
        "WARNING: Parsing fail on param `" << setter_name << "` in `" << api_attrs_class_name << "`.\n" <<
        "         Type `" << param_type.getAsString() << "` is not supported." << std::endl;
      //param_type->dump();
      continue;
    }
    if (is_dataformat_param && param->get_kind() == TFParam::TFPK_StringPiece)
      param = std::make_unique<TFDataFormatParam>(param_name);
    
    setters.push_back(
      std::move(std::make_tuple(setter_name, std::make_unique<TFBoolParam>("set_" + setter_name) ,std::move(param))));
  }

  std::unique_ptr<TFParam> tf_param =
    std::make_unique<TFAPIAttrsParam>(
      name,
      api_attrs_class_name,
      std::move(setters));
  return tf_param;
}

std::unique_ptr<TFParam> extractTFParam(clang::QualType t, std::string name, ASTContext &Ctx) {
  // Base case
  if (auto scope_param = extractTFScope(t, name, Ctx))
    return scope_param;
  if (auto builtin_param = extractTFBuiltin(t, name, Ctx))
    return builtin_param;
  if (auto string_piece_param = extractTFStringPiece(t, name, Ctx))
    return string_piece_param;
  if (auto dtype_param = extractTFDtype(t, name, Ctx))
    return dtype_param;
  if (auto vector_param = extractTFVector(t, name, Ctx))
    return vector_param;
  if (auto tensor_param = extractTFTensor(t, name, Ctx))
    return tensor_param;
  if (auto array_slice_param = extractTFArraySlice(t, name, Ctx))
    return array_slice_param;
  if (auto tuple_param = extractTFTuple(t, name, Ctx))
    return tuple_param;
  if (auto pair_param = extractTFPair(t, name, Ctx))
    return pair_param;

  // Recursive case
  //if (auto optional_param = extractTFOptional(t, name, Ctx))
  //  return optional_param;
  if (auto variant_param = extractTFVariant(t, name, Ctx))
    return variant_param;
  if (auto api_options_param = extractTFAPIAttrs(t, name, Ctx))
    return api_options_param;

  // Simplifying sugars
  if (t->isLValueReferenceType()) {
    return extractTFParam(t->getAs<LValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (t->isRValueReferenceType()) {
    return extractTFParam(t->getAs<RValueReferenceType>()->getPointeeType(), name, Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      return extractTFParam(elaborated->desugar(), name, Ctx);
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    if (tdtype->isSugared())
      return extractTFParam(tdtype->desugar(), name, Ctx);
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared())
      return extractTFParam(tstype->desugar(), name, Ctx);
  }

  return nullptr;
}
