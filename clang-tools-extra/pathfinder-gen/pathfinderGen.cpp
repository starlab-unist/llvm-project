#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "parse.h"
#include "api.h"
#include "utils.h"
#include <iostream>
#include <fstream>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;

std::map<std::string, std::map<std::string, std::string>> generated_code;

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    std::string function_name = Declaration->getNameAsString();
    std::string function_name_qualified = Declaration->getQualifiedNameAsString();

    std::string function_group;
    for (auto& entry: get_torch_function_list()) {
      if (entry.second.find(function_name_qualified) != entry.second.end()) {
        function_group = entry.first;
        break;
      }
    }
    if (function_group == "")
      return true;

    std::cout << "Generating fuzz target of API `" << function_name_qualified << "`..." << std::endl;

    option_class_done = false;
    auto param_decls = Declaration->parameters();
    std::vector<std::unique_ptr<TorchParam>> params;
    for (auto param_decl: param_decls) {
      clang::QualType t = param_decl->getType();
      std::string name = param_decl->getNameAsString();
      std::unique_ptr<TorchParam> p = parseTorchParam(t, name, *Context);
      if (p != nullptr)
        params.push_back(std::move(p));
    }

    if (params.empty()) {
      std::cout << "FAILED: Generating fuzz target of API `" << function_name_qualified << "` failed." << std::endl;
      return true;
    }

    generated_code[function_group][function_name] =
      TorchFunction(function_name_qualified, std::move(params)).gen_fuzz_target();

    return true;
  }

  Optional<std::vector<std::unique_ptr<TorchParam>>> parseModuleCtor(CXXConstructorDecl* ctor, size_t num_input_tensor) {
    if (ctor->isCopyOrMoveConstructor() ||
        ctor->isSpecializationCopyingObject() ||
        ctor->isInheritingConstructor())
      return None;

    option_class_done = false;
    std::vector<std::unique_ptr<TorchParam>> params;
    //for (size_t i = 0; i < num_input_tensor; i++)
    //  params.push_back(std::make_unique<TorchParam>(TENSOR, "tensor" + std::to_string(i)));
    for (const auto* param: ctor->parameters()) {
      clang::QualType t = param->getType();
      std::string name = param->getNameAsString();
      std::unique_ptr<TorchParam> p = parseTorchParam(t, name, *Context);
      if (p != nullptr)
        params.push_back(std::move(p));
    }
    return params;
  }

  std::vector<std::unique_ptr<TorchParam>> pickBest(std::vector<std::vector<std::unique_ptr<TorchParam>>> candidates) {
    assert(!candidates.empty());

    for (auto&& params: candidates)
      for (auto&& param: params)
        if (isa<TorchAPIOptionsParam>(param.get()))
          return std::move(params);

    size_t num_params_best = 0;
    size_t best_idx = 0;
    for (size_t i = 0; i < candidates.size(); i++)
      if (candidates[i].size() > num_params_best) {
        num_params_best = candidates[i].size();
        best_idx = i;
      }

    return std::move(candidates[best_idx]);
  }

  bool VisitCXXRecordDecl(CXXRecordDecl* Declaration) {
    std::string module_name = Declaration->getNameAsString();
    std::string module_name_qualified = Declaration->getQualifiedNameAsString();

    auto torch_module_list = get_torch_module_list();
    if (torch_module_list.find(module_name_qualified) == torch_module_list.end())
      return true;

    std::cout << "Generating fuzz target of API `" << module_name_qualified << "`..." << std::endl;

    assert(!Declaration->bases().empty());
    const auto* elaborated = dyn_cast<ElaboratedType>(Declaration->bases_begin()->getType());
    assert(elaborated != nullptr);
    assert(elaborated->isSugared());
    const auto* tstype = dyn_cast<TemplateSpecializationType>(elaborated->desugar());
    assert(tstype->isSugared());
    assert(tstype->desugar()->getAs<RecordType>()->getDecl()->getNameAsString() == "ModuleHolder");
    auto targs = tstype->template_arguments();
    assert(targs.size() == 1);
    auto* class_decl = dyn_cast<CXXRecordDecl>(targs[0].getAsType()->getAs<RecordType>()->getDecl());

    /* std::vector<std::vector<std::unique_ptr<TorchParam>>> candidates;

    for (auto ctor: class_decl->ctors())
      if (auto parsed = parseModuleCtor(ctor, num_input_tensor(module_name_qualified)))
        candidates.push_back(std::move(parsed.getValue()));
    if (!class_decl->bases().empty()) {
      const TemplateSpecializationType* tstype2;
      if (const auto* etype = dyn_cast<ElaboratedType>(class_decl->bases_begin()->getType())) {
        tstype2 = dyn_cast<TemplateSpecializationType>(etype->desugar());
      } else {
        tstype2 = dyn_cast<TemplateSpecializationType>(class_decl->bases_begin()->getType());
      }

      if (tstype2 != nullptr) {
        assert(tstype2->isSugared());
        if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(tstype2->desugar()->getAs<RecordType>()->getDecl())) {
          if (!ctsdecl->bases().empty() && ctsdecl->bases_begin()->getType()->getAs<RecordType>()->getDecl()->getNameAsString() == "NormImplBase") {
            auto targs = ctsdecl->bases_begin()->getType()->getAs<TemplateSpecializationType>()->template_arguments();
            assert(targs.size() == 3 && targs[2].getKind() == TemplateArgument::ArgKind::Type);
            option_class_done = false;
            std::vector<std::unique_ptr<TorchParam>> params;
            for (size_t i = 0; i < num_input_tensor(module_name_qualified); i++)
              params.push_back(std::make_unique<TorchParam>(TENSOR, "tensor" + std::to_string(i)));
            if (auto p = parseTorchParam(targs[2].getAsType(), "options", *Context))
              params.push_back(std::move(p));
            candidates.push_back(std::move(params));
          } else {
            for (auto ctor: ctsdecl->ctors())
              if (auto parsed = parseModuleCtor(ctor, num_input_tensor(module_name_qualified)))
                candidates.push_back(std::move(parsed.getValue()));
          }
        }
      }
    }

    auto params = pickBest(std::move(candidates));
    if (params.empty()) {
      std::cout << "FAILED: Generating fuzz target of API `" << module_name_qualified << "` failed." << std::endl;
      return true;
    }

    generated_code[torch_module_list_file_name()][module_name] =
      gen_torch_module_pathfinder(module_name_qualified, params, num_input_tensor(module_name_qualified)); */

    return true;
  }

private:
  ASTContext *Context;
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  explicit FindNamedClassConsumer(ASTContext *Context)
    : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  FindNamedClassVisitor Visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
    return std::make_unique<FindNamedClassConsumer>(&Compiler.getASTContext());
  }
};

static llvm::cl::OptionCategory MyToolCategory("my-tool options");

int main(int argc, const char **argv) {
  init_torch_api_list();

  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  write_recursive(generated_code);

  return 0;
}
