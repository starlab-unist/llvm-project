// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "param.h"
#include <iostream>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx);

Optional<std::unique_ptr<Param>> parseBuiltin(clang::QualType t, ASTContext &Ctx) {
  std::cout << "Parse builtin\n";
  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      //std::cout << "Builtin: " << builtin->getNameAsCString(Ctx.getPrintingPolicy()) << std::endl;
      t->dump();
      return std::make_unique<Param>(INT);
    }
    /* if (builtin->isBool())
      return std::make_unique<Param>(BOOL); */
  }
  return None;
}

Optional<std::unique_ptr<Param>> parseEnum(clang::QualType t, ASTContext &Ctx) {
  std::cout << "Parse enum\n";
  static std::string enum_prefix = "torch::enumtype::";

  if (const auto* rtype = t->getAs<RecordType>()) {
    std::string qualified_name = rtype->getDecl()->getQualifiedNameAsString();
    if (qualified_name.compare(0, enum_prefix.size(), enum_prefix) == 0) {
      t->dump();
      return std::make_unique<Param>(ENUM, rtype->getDecl()->getNameAsString());
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseTensor(clang::QualType t) {
  std::cout << "Parse tensor\n";
  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor") {
          t->dump();
          return std::make_unique<Param>(TENSOR);
        }
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor") {
        t->dump();
        return std::make_unique<Param>(TENSOR);
      }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseExpandingArray(clang::QualType t, ASTContext &Ctx) {
  std::cout << "Parse expending array\n";
  if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared()) {
      if (const auto* rtype = tstype->desugar()->getAs<RecordType>()) {
        if (rtype->getDecl()->getNameAsString() == "ExpandingArray") {
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          int64_t expendingarray_size =
            targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
          t->dump();
          return std::make_unique<Param>(EXPENDINGARRAY, expendingarray_size);
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, ASTContext &Ctx) {
  std::cout << "Parse variant\n";
  if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    if (tstype->isSugared()) {
      if (const auto* etype = tstype->desugar()->getAs<ElaboratedType>()) {
        if (const auto* tstype2 = etype->getAs<TemplateSpecializationType>()) {
          if (const auto* rtype = tstype2->desugar()->getAs<RecordType>()) {
            if (rtype->getDecl()->getNameAsString() == "variant") {
              auto targs = tstype2->template_arguments();
              std::vector<std::unique_ptr<Param>> types;
              for (auto targ: targs) {
                types.push_back(std::move(parseTorchParam(targ.getAsType(), Ctx)));
              }
              t->dump();
              return std::make_unique<Param>(VARIANT, std::move(types));
            }
          }
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseAPIOption(clang::QualType t, ASTContext &Ctx) {
  std::cout << "Parse API Option\n";
  static std::string option_suffix = "Options";

  if (const auto* tdtype = t->getAs<TypedefType>()) {
    std::string api_option_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      assert(tdtype->isSugared());
      auto tstype = tdtype->desugar()->getAs<TemplateSpecializationType>();
      assert(tstype->isSugared());
      auto rtype = tstype->desugar()->getAs<RecordType>();
      auto cdecl = rtype->getAsCXXRecordDecl();
      assert(cdecl != nullptr);
      cdecl->dump();
      std::cout << "===========================\n";
      /* for (auto field: cdecl->fields()) {
        field->dump();
        field->getInClassInitializer()->dump();
      } */
      std::map<std::string,std::unique_ptr<Param>> api_option_types;
      for (auto method: cdecl->methods()) {
        if (method->parameters().size() != 1)
          continue;
        std::string param_name = method->getNameInfo().getAsString();
        std::unique_ptr<Param> param = parseTorchParam(method->parameters()[0]->getType(), Ctx);
        if (param != nullptr) {
          param->set_default(param_name, cdecl);
          api_option_types.insert({param_name, std::move(param)});
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

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx) {
  if (auto builtin_opt = parseBuiltin(t, Ctx))
    return std::move(builtin_opt.getValue());
  if (auto enum_opt = parseEnum(t, Ctx))
    return std::move(enum_opt.getValue());
  if (auto tensor_opt = parseTensor(t))
    return std::move(tensor_opt.getValue());
  if (auto expendingarray_opt = parseExpandingArray(t, Ctx))
    return std::move(expendingarray_opt.getValue());
  if (auto variant_opt = parseVariant(t, Ctx))
    return std::move(variant_opt.getValue());
  if (auto api_option_opt = parseAPIOption(t, Ctx))
    return std::move(api_option_opt.getValue());

  // simplify
  if (t->isLValueReferenceType()) {
    return parseTorchParam(t->getAs<LValueReferenceType>()->getPointeeType(), Ctx);
  } else if (t->isRValueReferenceType()) {
    return parseTorchParam(t->getAs<RValueReferenceType>()->getPointeeType(), Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    std::cout << "elaborated~~\n";
    if (elaborated->isSugared()) {
      std::cout << "is sugared~~\n";
      return parseTorchParam(elaborated->desugar(), Ctx);
    }
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    std::cout << "typedeftype~~\n";
    if (tdtype->isSugared()) {
      std::cout << "is sugared~~\n";
      return parseTorchParam(tdtype->desugar(), Ctx);
    }
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    std::cout << "templateSpecialized~~\n";
    if (tstype->isSugared()) {
      std::cout << "is sugared~~\n";
      const auto* rtype = tstype->desugar()->getAs<RecordType>();
      if (rtype == nullptr)
        return parseTorchParam(tstype->desugar(), Ctx);
    }
  }

  t->dump();
  return nullptr;
}

DeclarationMatcher FuncMatcher =
  functionDecl(hasName("torch::nn::functional::conv3d")).bind("func");

class FuncPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) override {
    ASTContext *Context = Result.Context;
    if (const FunctionDecl *FD = Result.Nodes.getNodeAs<clang::FunctionDecl>("func")){
      //FD->dump();
      auto params = FD->parameters();
      for (auto param: params) {
        std::cout << "param: " << param->getNameAsString() << std::endl;
        clang::QualType t = param->getType();
        std::unique_ptr<Param> p = parseTorchParam(t, *Context);
        if (p)
          std::cout << p->to_string(0) << std::endl;




        /* std::cout << "===================================================\n";
        //param->dump();
        //param->getType()->dump();
        if (param->getType()->isLValueReferenceType()) {
          auto lvrt = param->getType()->getAs<LValueReferenceType>();
          lvrt->dump();
          std::cout << "---------------------------------------------------\n";
          if (lvrt->getPointeeType()->isTypedefNameType()) {
            auto tdtype = lvrt->getPointeeType()->getAs<TypedefType>();
            tdtype->dump();
            std::cout << "---------------------------------------------------\n";
            std::cout << tdtype->getDecl()->getNameAsString() << std::endl << tdtype->getDecl()->getQualifiedNameAsString() << std::endl;
            std::cout << "---------------------------------------------------\n";
            if (tdtype->isSugared()) {
              std::cout << "name: " + tdtype->getDecl()->getNameAsString() << std::endl;
              auto tst = tdtype->desugar()->getAs<TemplateSpecializationType>();
              tst->dump();
              std::cout << "---------------------------------------------------\n";
              if (tst->isSugared()) {
                if (tst->desugar()->isRecordType()) {
                  auto rtype = tst->desugar()->getAs<RecordType>();
                  auto cdecl = rtype->getAsCXXRecordDecl();
                  std::cout << "===================================================\n";
                  for (auto method: cdecl->methods()) {
                    if (method->parameters().size() == 0)
                      continue;
                    std::cout << "---------------------------------------------------\n";
                    std::cout << method->getNameInfo().getAsString() << std::endl;
                    for (auto param: method->parameters()) {
                      std::cout << "param: " << param->getNameAsString() << std::endl;
                      clang::QualType t = param->getType();
                      std::unique_ptr<Param> p = parseTorchParam(t, *Context);
                      if (p)
                        std::cout << p->to_string(0) << std::endl;
                    }                    
                  }
                }
              }
            }
          }
        } */
      }
    }
  }
};

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  //LoopPrinter Printer;
  FuncPrinter Printer;
  MatchFinder Finder;
  //Finder.addMatcher(LoopMatcher, &Printer);
  Finder.addMatcher(FuncMatcher, &Printer);

  return Tool.run(newFrontendActionFactory(&Finder).get());
}
