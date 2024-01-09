#include "extract.h"
#include "utils.h"
#include <iostream>
#include <fstream>

bool option_class_done;

std::unique_ptr<TFParam> extractTFScope(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = t->getAs<RecordType>())
    if (rtype->getDecl()->getQualifiedNameAsString() == "tensorflow::Scope") {
      std::cout << "Found Scope" << std::endl;
      tf_param = std::make_unique<TFScopeParam>(name);
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
      tf_param = std::make_unique<TFStringPieceParam>(name);
      std::cout << "Found StringPiece" << std::endl;
    }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFString(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        std::string type_name = rtype->getDecl()->getNameAsString();
        if (type_name == "basic_string") {
          tf_param = std::make_unique<TFStringParam>(name, false);
        } else if (type_name == "basic_string_view") {
          tf_param = std::make_unique<TFStringParam>(name, true);
        }
      }
    }
  }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFMemoryFormat(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "MemoryFormat")
      tf_param = std::make_unique<TFMemoryFormatParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFLayout(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "Layout")
      tf_param = std::make_unique<TFLayoutParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFDevice(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = t->getAs<RecordType>())
    if (rtype->getDecl()->getQualifiedNameAsString() == "c10::Device")
      tf_param = std::make_unique<TFDeviceParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFDtype(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* etype = t->getAs<EnumType>())
    if (etype->getDecl()->getNameAsString() == "ScalarType")
      tf_param = std::make_unique<TFDtypeParam>(name);

  return tf_param;
}

std::unique_ptr<TFParam> extractTFEnum(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;
  static const std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0)
      tf_param =
        std::make_unique<TFEnumParam>(
          name,
          qualified_name.substr(enum_prefix.size(), qualified_name.size() - enum_prefix.size()));
  }

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
    if (rtype->getDecl()->getQualifiedNameAsString() == "tensorflow::Input") {
      std::cout << "Found Tensor" << std::endl;
      tf_param = std::make_unique<TFTensorParam>(name);
    }

  return tf_param;
}

/* std::unique_ptr<TFParam> extractTFTensor(clang::QualType t, std::string name,  ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor")
          tf_param = std::make_unique<TFTensorParam>(name);
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor")
        tf_param = std::make_unique<TFTensorParam>(name);
  }

  return tf_param;
} */

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

/* std::unique_ptr<TFParam> extractTFArraySlice(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      //std::cout << "======================================" << std::endl;
      //ctsdecl->dump();
      //std::cout << "======================================" << std::endl;
      //if (ctsdecl->getNameAsString() == "ArraySlice") {
      if (ctsdecl->getNameAsString() == "Span") {
        std::cout << "Found Span" << std::endl;
        ctsdecl->dump();
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type); 
        std::vector<std::unique_ptr<TFParam>> params;
        for (size_t i = 0; i < MAX_ARRAYREF_SIZE; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TFParam> param =
            extractTFParam(targ[0].getAsType(), param_name, Ctx);
          if (param == nullptr)
            break;
          params.push_back(std::move(param));
        }
        if (params.empty()) {
          std::cerr <<
            "WARNING: While extracting `ArrayRef` param `" << name << "`, failed to extract base type `" <<
            targ[0].getAsType().getAsString() << "`.\n" <<
            "         param `" << name << "` will be fixed to empty ArrayRef." << std::endl;
          tf_param = std::make_unique<TFArrayRefParam>(name, targ[0].getAsType().getAsString());
        } else {
          tf_param = std::make_unique<TFArrayRefParam>(name, std::move(params));
        }
      }
    }
  }
  
  return tf_param;
} */

std::unique_ptr<TFParam> extractTFArrayRef(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      if (ctsdecl->getNameAsString() == "ArrayRef") {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type); 
        std::vector<std::unique_ptr<TFParam>> params;
        for (size_t i = 0; i < MAX_ARRAYREF_SIZE; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TFParam> param =
            extractTFParam(targ[0].getAsType(), param_name, Ctx);
          if (param == nullptr)
            break;
          params.push_back(std::move(param));
        }
        if (params.empty()) {
          std::cerr <<
            "WARNING: While extracting `ArrayRef` param `" << name << "`, failed to extract base type `" <<
            targ[0].getAsType().getAsString() << "`.\n" <<
            "         param `" << name << "` will be fixed to empty ArrayRef." << std::endl;
          tf_param = std::make_unique<TFArrayRefParam>(name, targ[0].getAsType().getAsString());
        } else {
          tf_param = std::make_unique<TFArrayRefParam>(name, std::move(params));
        }
      }
    }
  }
  
  return tf_param;
}

std::unique_ptr<TFParam> extractTFExpandingArray(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

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
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).value().getExtValue();
      assert(expandingarray_size > 0);
      std::vector<std::unique_ptr<TFParam>> params;
      if (targ.size() == 1){
        for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TFParam> param = std::make_unique<TFIntParam>(param_name, "long");
          params.push_back(std::move(param));
        }
      }
      else if (targ.size() == 2){
        for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
          std::string param_name = name + "_" + std::to_string(i);
          std::unique_ptr<TFParam> param =
            extractTFParam(targ[1].getAsType(), param_name, Ctx);
          if (param == nullptr)
            return nullptr;
          params.push_back(std::move(param));
        }
      }
      tf_param =
        std::make_unique<TFExpandingArrayParam>(name, expandingarray_size, std::move(params));
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
          std::vector<std::unique_ptr<TFParam>> params;
          for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
            std::string param_name = name + "_" + std::to_string(i);
            std::unique_ptr<TFParam> param =
              extractTFParam(targ[1].getAsType(), param_name, Ctx);
            if (param == nullptr)
              return nullptr;
            params.push_back(std::move(param));
          }
          tf_param =
            std::make_unique<TFExpandingArrayParam>(name, expandingarray_size, std::move(params));
        }
      }
    }
  }

  return tf_param;
}

std::unique_ptr<TFParam> extractTFExpandingArrayWithOptionalElem(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

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
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).value().getExtValue();
      assert(expandingarray_size > 0);
      std::vector<std::unique_ptr<TFParam>> params;
      for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
        std::string param_name = name + "_" + std::to_string(i);
        std::unique_ptr<TFParam> param = std::make_unique<TFIntParam>(param_name, "long");
        std::unique_ptr<TFParam> optional_param =
          std::make_unique<TFOptionalParam>(param_name, std::move(param));
        params.push_back(std::move(optional_param));
      }
      tf_param =
        std::make_unique<TFExpandingArrayWithOptionalElemParam>(
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
          std::vector<std::unique_ptr<TFParam>> params;
          for (size_t i = 0; i < (size_t)expandingarray_size; i++) {
            std::string param_name = name + "_" + std::to_string(i);
            std::unique_ptr<TFParam> param =
              extractTFParam(targ[1].getAsType(), param_name + "_base", Ctx);
            if (param == nullptr)
              return nullptr;
            std::unique_ptr<TFParam> optional_param =
              std::make_unique<TFOptionalParam>(param_name, std::move(param));
            params.push_back(std::move(optional_param));
          }
          tf_param =
            std::make_unique<TFExpandingArrayWithOptionalElemParam>(
              name, expandingarray_size, std::move(params));
        }
      }
    }
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

std::unique_ptr<TFParam> extractTFSymint(clang::QualType t, std::string name, ASTContext &Ctx) {
  std::unique_ptr<TFParam> tf_param;

  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (rtype->getDecl()->getNameAsString() == "SymInt")
      tf_param = std::make_unique<TFSymIntParam>(name);
  }
  
  return tf_param;
}

std::unique_ptr<TFParam> extractTFOptional(clang::QualType t, std::string name, ASTContext &Ctx) {
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
}

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

std::unique_ptr<TFParam> extractTFAPIOptions(clang::QualType t, std::string name, ASTContext &Ctx) {
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

  std::vector<std::unique_ptr<TFParam>> ctor_params;
  std::vector<std::unique_ptr<TFParam>> member_params;
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
        std::vector<std::unique_ptr<TFParam>> ctor_params_;
        std::set<std::string> ctor_param_names_;
        bool extract_succeed = true;
        for (const auto* param: ctordecl->parameters()) {
          std::string param_name = param->getNameAsString();
          std::unique_ptr<TFParam> tf_param = extractTFParam(param->getType(), param_name, Ctx);
          if (tf_param == nullptr) {
            std::cerr <<
              "WARNING: Parsing fail on param `" << param_name << "` in `" << api_options_class_name << "`.\n" <<
              "         Type `" << param->getType().getAsString() << "` is not supported." << std::endl;
            extract_succeed = false;
            break;
          }
          tf_param->set_default(get_default_expr(param_name, cdecl));
          ctor_params_.push_back(std::move(tf_param));
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

    std::unique_ptr<TFParam> param = extractTFParam(method->parameters()[0]->getType(), param_name, Ctx);
    if (param == nullptr) {
      std::cerr <<
        "WARNING: Parsing fail on param `" << param_name << "` in `" << api_options_class_name << "`.\n" <<
        "         Type `" << method->parameters()[0]->getType().getAsString() << "` is not supported." << std::endl;
      continue;
    }
    param->set_default(get_default_expr(param_name, cdecl));

    bool insert = true;
    for (size_t i = 0; i < member_params.size(); i++) {
      if (member_params[i]->get_name() == param_name) {
        insert = false;
        if (!member_params[i]->stable()) {
          member_params[i] = std::move(param);
          break;
        }
      }
    }
    if (insert)
      member_params.push_back(std::move(param));
  }
  std::unique_ptr<TFParam> tf_param =
    std::make_unique<TFAPIOptionsParam>(
      name,
      api_options_class_name,
      std::move(ctor_params),
      std::move(member_params));
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
  if (auto string_param = extractTFString(t, name, Ctx))
    return string_param;
  if (auto memory_format_param = extractTFMemoryFormat(t, name, Ctx))
    return memory_format_param;
  if (auto layout_param = extractTFLayout(t, name, Ctx))
    return layout_param;
  if (auto device_param = extractTFDevice(t, name, Ctx))
    return device_param;
  if (auto dtype_param = extractTFDtype(t, name, Ctx))
    return dtype_param;
  if (auto enum_param = extractTFEnum(t, name, Ctx))
    return enum_param;
  if (auto vector_param = extractTFVector(t, name, Ctx))
    return vector_param;
  if (auto tensor_param = extractTFTensor(t, name, Ctx))
    return tensor_param;
  if (auto arrayref_param = extractTFArraySlice(t, name, Ctx))
    return arrayref_param;
  if (auto arrayref_param = extractTFArrayRef(t, name, Ctx))
    return arrayref_param;
  if (auto expandingarray_param = extractTFExpandingArray(t, name, Ctx))
    return expandingarray_param;
  if (auto expandingarraywithoptionalelem_param = extractTFExpandingArrayWithOptionalElem(t, name, Ctx))
    return expandingarraywithoptionalelem_param;
  if (auto tuple_param = extractTFTuple(t, name, Ctx))
    return tuple_param;
  if (auto pair_param = extractTFPair(t, name, Ctx))
    return pair_param;
  if (auto symint_param = extractTFSymint(t, name, Ctx))
    return symint_param;

  // Recursive case
  if (auto optional_param = extractTFOptional(t, name, Ctx))
    return optional_param;
  if (auto variant_param = extractTFVariant(t, name, Ctx))
    return variant_param;
  if (auto api_options_param = extractTFAPIOptions(t, name, Ctx))
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
