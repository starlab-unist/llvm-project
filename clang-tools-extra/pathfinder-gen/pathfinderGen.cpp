// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"

#include "param.h"
#include <iostream>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;

std::map<std::string, std::vector<size_t>> target_api = {
  {"torch::nn::functional::conv1d",{3,3,1}},
  {"torch::nn::functional::conv2d",{4,4,1}},
  {"torch::nn::functional::conv3d",{5,5,1}},
  {"torch::nn::functional::conv_transpose1d",{3,3,1}},
  {"torch::nn::functional::conv_transpose2d",{4,4,1}},
  {"torch::nn::functional::conv_transpose3d",{5,5,1}},
};
std::string current_target;
std::vector<size_t> current_tensor_rank;
size_t current_tensor_rank_idx;

size_t get_rank() {
  static const size_t DEFAULT_RANK = 3;
  if (current_tensor_rank.size() <= current_tensor_rank_idx) {
    return DEFAULT_RANK;
  } else {
    return current_tensor_rank[current_tensor_rank_idx++];
  }
}

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx, bool is_optional=false);

Optional<std::unique_ptr<Param>> parseBuiltin(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse builtin\n";
  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      //std::cout << "Builtin: " << builtin->getNameAsCString(Ctx.getPrintingPolicy()) << std::endl;
      //t->dump();
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "long") {
        return std::make_unique<Param>(INT);
      } else if (t == "bool") {
        return std::make_unique<Param>(BOOL);
      } else {
        std::cout << "dd: " << t << std::endl;
        assert(false);
      }
    }
    /* if (builtin->isBool())
      return std::make_unique<Param>(BOOL); */
  }
  return None;
}

Optional<std::unique_ptr<Param>> parseEnum(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse enum\n";
  static std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0) {
      //t->dump();
      return std::make_unique<Param>(ENUM, rtype->getDecl()->getNameAsString());
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseIntVector(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse int vector\n";

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          if (targs.size() == 1) {
            auto p = parseTorchParam(targs[0].getAsType(), Ctx);
            if (p->ptype == INT)
              return std::make_unique<Param>(INTVECTOR);
          }
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseTensor(clang::QualType t, bool is_optional) {
  //std::cout << "Parse tensor\n";
  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor") {
          //std::cout << "target_api_name: " << target_api_name << std::endl;
          /* if (current_target == "torch::nn::functional::conv2d") {
            std::cout << "$$$$$$$$$$$$$is true\n";
            if (is_optional) {
              std::cout << "$$$$$$$$$$$$$is true\n";
              return std::make_unique<Param>(TENSOR, 1);
            } else {
              return std::make_unique<Param>(TENSOR, 4);
            }
          } else {
            return std::make_unique<Param>(TENSOR);
          } */
          return std::make_unique<Param>(TENSOR, get_rank());
        }
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor") {
        //std::cout << "target_api_name: " << target_api_name << std::endl;
        /* if (current_target == "torch::nn::functional::conv2d") {
          std::cout << "$$$$$$$$$$$$$is true\n";
          if (is_optional) {
            std::cout << "$$$$$$$$$$$$$is true\n";
            return std::make_unique<Param>(TENSOR, 1);
          } else {
            return std::make_unique<Param>(TENSOR, 4);
          }
        } else {
          return std::make_unique<Param>(TENSOR);
        } */
        return std::make_unique<Param>(TENSOR, get_rank());
      }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseExpandingArray(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse expending array\n";
  if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared()) {
      if (const auto* rtype = tstype->desugar()->getAs<RecordType>()) {
        if (rtype->getDecl()->getNameAsString() == "ExpandingArray") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          int64_t expendingarray_size =
            targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
          //t->dump();
          return std::make_unique<Param>(EXPENDINGARRAY, expendingarray_size);
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse variant\n";
  //t->dump();
  //if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    //if (tstype->isSugared()) {
      //if (const auto* etype = tstype->desugar()->getAs<ElaboratedType>()) {
        //if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
        if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
          //if (const auto* rtype = tstype->desugar()->getAs<RecordType>()) {
          if (tstype->isSugared()) {
            if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
              if (rtype->getDecl()->getNameAsString() == "variant") {
                auto targs = tstype->template_arguments();
                std::vector<std::unique_ptr<Param>> types;
                std::vector<std::string> enum_vec;
                for (auto targ: targs) {
                  auto p = parseTorchParam(targ.getAsType(), Ctx);
                  if (p->ptype == ENUM)
                    enum_vec.push_back(p->enum_name);
                  types.push_back(std::move(p));
                }
                //t->dump();
                return std::make_unique<Param>(VARIANT, std::move(types), enum_vec);
              }
            }
          }
        }
      //}
    //}
  //}

  return None;
}

void push_back_unique(std::vector<std::pair<std::string,std::unique_ptr<Param>>>& vec, std::string name, std::unique_ptr<Param> param) {
  bool exist = false;
  for (auto&& p: vec) {
    if (p.first == name)
      exist = true;
  }

  if (!exist)
    vec.push_back({name, std::move(param)});
}

Optional<std::unique_ptr<Param>> parseAPIOption(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse API Option\n";
  static std::string option_suffix = "Options";
  static std::set<std::string> seen;

  if (const auto* rtype = t->getAs<RecordType>()) {
    auto cdecl = rtype->getAsCXXRecordDecl();
    assert(cdecl != nullptr);
    std::string api_option_name = cdecl->getQualifiedNameAsString();
    if (seen.find(api_option_name) == seen.end() &&
        api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      seen.insert(api_option_name);
      //cdecl->dump();
      //std::cout << "===========================\n";
      /* for (auto field: cdecl->fields()) {
        field->dump();
        field->getInClassInitializer()->dump();
      } */
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> api_option_types;
      for (auto method: cdecl->methods()) {
        if (method->getNameAsString() == cdecl->getNameAsString())
          continue;
        if (method->parameters().size() != 1)
          continue;
        std::string param_name = method->getNameInfo().getAsString();
        std::unique_ptr<Param> param = parseTorchParam(method->parameters()[0]->getType(), Ctx, true);
        if (param != nullptr) {
          param->set_default(param_name, cdecl);
          //api_option_types.push_back({param_name, std::move(param)});
          push_back_unique(api_option_types, param_name, std::move(param));
        }
      }
      return
        std::make_unique<Param>(
          API_OPTION,
          cdecl->getQualifiedNameAsString(),
          std::move(api_option_types));
    }
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    std::string api_option_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (seen.find(api_option_name) == seen.end() &&
        api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      seen.insert(api_option_name);
      assert(tdtype->isSugared());
      auto tstype = tdtype->desugar()->getAs<TemplateSpecializationType>();
      assert(tstype->isSugared());
      auto rtype = tstype->desugar()->getAs<RecordType>();
      auto cdecl = rtype->getAsCXXRecordDecl();
      assert(cdecl != nullptr);
      //cdecl->dump();
      //std::cout << "===========================\n";
      /* for (auto field: cdecl->fields()) {
        field->dump();
        field->getInClassInitializer()->dump();
      } */
      std::vector<std::pair<std::string,std::unique_ptr<Param>>> api_option_types;
      for (auto method: cdecl->methods()) {
        if (method->parameters().size() != 1)
          continue;
        std::string param_name = method->getNameInfo().getAsString();
        std::unique_ptr<Param> param = parseTorchParam(method->parameters()[0]->getType(), Ctx, true);
        if (param != nullptr) {
          param->set_default(param_name, cdecl);
          //api_option_types.push_back({param_name, std::move(param)});
          push_back_unique(api_option_types, param_name, std::move(param));
        }
      }
      return
        std::make_unique<Param>(
          API_OPTION,
          tdtype->getDecl()->getQualifiedNameAsString(),
          std::move(api_option_types));
    }
  }

  return None;
}

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx, bool is_optional) {
  if (auto builtin_opt = parseBuiltin(t, Ctx))
    return std::move(builtin_opt.getValue());
  if (auto enum_opt = parseEnum(t, Ctx))
    return std::move(enum_opt.getValue());
  if (auto int_vec_opt = parseIntVector(t, Ctx))
    return std::move(int_vec_opt.getValue());
  if (auto tensor_opt = parseTensor(t, is_optional))
    return std::move(tensor_opt.getValue());
  if (auto expendingarray_opt = parseExpandingArray(t, Ctx))
    return std::move(expendingarray_opt.getValue());
  if (auto variant_opt = parseVariant(t, Ctx))
    return std::move(variant_opt.getValue());
  if (auto api_option_opt = parseAPIOption(t, Ctx))
    return std::move(api_option_opt.getValue());

  // simplify
  if (t->isLValueReferenceType()) {
    return parseTorchParam(t->getAs<LValueReferenceType>()->getPointeeType(), Ctx, is_optional);
  } else if (t->isRValueReferenceType()) {
    return parseTorchParam(t->getAs<RValueReferenceType>()->getPointeeType(), Ctx, is_optional);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    //std::cout << "elaborated~~\n";
    if (elaborated->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(elaborated->desugar(), Ctx, is_optional);
    }
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    //std::cout << "typedeftype~~\n";
    if (tdtype->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(tdtype->desugar(), Ctx, is_optional);
    }
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    //std::cout << "templateSpecialized~~\n";
    if (tstype->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(tstype->desugar(), Ctx, is_optional);
    }
  }

  //t->dump();
  return nullptr;
}

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    //if (const FunctionDecl *FD = Result.Nodes.getNodeAs<clang::FunctionDecl>("func")){
    //if (Declaration->getQualifiedNameAsString() == target_api) {
    if (target_api.find(Declaration->getQualifiedNameAsString()) != target_api.end()) {
      //FD->dump();
      current_target = Declaration->getQualifiedNameAsString();
      current_tensor_rank = target_api.find(current_target)->second;
      current_tensor_rank_idx = 0;
      auto param_decls = Declaration->parameters();
      std::vector<std::unique_ptr<Param>> params;
      for (auto param_decl: param_decls) {
        //std::cout << "param: " << param_decl->getNameAsString() << std::endl;
        clang::QualType t = param_decl->getType();
        t->dump();
        std::unique_ptr<Param> p = parseTorchParam(t, *Context);
        if (p != nullptr)
          params.push_back(std::move(p));
      }
      //for (auto&& param: params)
      //  std::cout << param->to_string(0) << std::endl;

      //std::cout << "=========================================\n";

      gen_pathfinder_fuzz_target(
        current_target,
        params,
        std::cout);

      std::cout << "=============================================================\n";
    }

    return true;
  }

private:
  ASTContext *Context;
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  explicit FindNamedClassConsumer(ASTContext *Context)
    : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  FindNamedClassVisitor Visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<FindNamedClassConsumer>(&Compiler.getASTContext());
  }
};

static llvm::cl::OptionCategory MyToolCategory("my-tool options");

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  //return Tool.run(std::make_unique<FindNamedClassAction>().get());
  return Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  //if (argc > 1) {
  //  clang::tooling::runToolOnCode(std::make_unique<FindNamedClassAction>(), argv[1]);
  //}
}