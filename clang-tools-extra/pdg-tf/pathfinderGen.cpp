#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "extract.h"
#include "api.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <regex>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;

class Output {
  typedef std::pair<std::string, std::string> api;
  typedef std::map<std::string, std::vector<api>> group;

  public:
    Output() {}
    void add(std::string fuzz_target_type, std::string group_name, std::string api_name, std::string code) {
      if (output.find(fuzz_target_type) == output.end())
        output[fuzz_target_type] = {};
      if (output[fuzz_target_type].find(group_name) == output[fuzz_target_type].end())
        output[fuzz_target_type][group_name] = {};
      output[fuzz_target_type][group_name].push_back({api_name, code});
    }
    void write() {
      std::set<std::string> names_seen;

      make_dir("generated");
      std::string cmake_contents0;
      for (auto& p0: output) {
        std::string fuzz_target_type = p0.first;
        auto groups = p0.second;
        make_dir("generated/" + fuzz_target_type);
        cmake_contents0 += "add_subdirectory(" + fuzz_target_type + ")\n";
        std::string cmake_contents1;
        for (auto& p1: groups) {
          std::string group_name = p1.first;
          auto apis = p1.second;
          make_dir("generated/" + fuzz_target_type + "/" + group_name);
          cmake_contents1 += "add_subdirectory(" + group_name + ")\n";
          std::string cmake_contents2;
          for (auto& p2: apis) {
            std::string api_name = p2.first;
            auto code = p2.second;
            std::string file_name =
              unique_name(fuzz_target_type + "_" + group_name + "_" + api_name, names_seen) + ".cpp";
            write_file("generated/" + fuzz_target_type + "/" + group_name + "/" + file_name, code);
            cmake_contents2 += "add_pathfinder_fuzz_target(" + strip_ext(file_name) + ")\n";
          }
          write_file("generated/" + fuzz_target_type + "/" + group_name + "/CMakeLists.txt", cmake_contents2);
        }
        write_file("generated/" + fuzz_target_type + "/CMakeLists.txt", cmake_contents1);
      }
      write_file("generated/CMakeLists.txt", cmake_contents0);
    }
  private:
    std::map<std::string, group> output;
    void write_file(std::string file_path, std::string contents) {
      std::ofstream f(file_path);
      if(f.is_open()){
        f << contents;
        f.close();
      } else {
        assert(false);
      }
    }
};

Output output;

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  Optional<std::vector<std::unique_ptr<TFParam>>> extractModuleCtor(CXXConstructorDecl* ctor) {
    if (ctor->isCopyOrMoveConstructor() ||
        ctor->isSpecializationCopyingObject() ||
        ctor->isInheritingConstructor())
      return std::nullopt;
    //std::cout << "==============ctor===============" << std::endl;
    //ctor->dump();

    option_class_done = false;
    std::vector<std::unique_ptr<TFParam>> params;
    for (const auto* param: ctor->parameters()) {
      //std::cout << "--------------param---------------" << std::endl;
      clang::QualType t = param->getType();
      //t->dump();
      std::string name = param->getNameAsString();
      std::unique_ptr<TFParam> p = extractTFParam(t, name, *Context);
      if (p == nullptr) {
        std::cerr <<
          "WARNING: Parsing fail on param `" << name << "`.\n" <<
          "         Type `" << t.getAsString() << "` is not supported." << std::endl;
        return std::nullopt;
      }
      params.push_back(std::move(p));
    }
    return params;
  }

  Optional<std::vector<std::unique_ptr<TFParam>>> extractForward(CXXMethodDecl* forward) {
    std::vector<std::unique_ptr<TFParam>> params;
    for (const auto* param: forward->parameters()) {
      clang::QualType t = param->getType();
      std::string name = param->getNameAsString();
      std::unique_ptr<TFParam> p = extractTFParam(t, name, *Context);
      if (p == nullptr) {
        std::cerr <<
          "WARNING: Parsing fail on param `" << name << "`.\n" <<
          "         Type `" << t.getAsString() << "` is not supported." << std::endl;
        return std::nullopt;
      }
      params.push_back(std::move(p));
    }
    return params;
  }

  std::vector<std::unique_ptr<TFParam>> pickBest(std::vector<std::vector<std::unique_ptr<TFParam>>> ctor_params_candidates) {
    assert(!ctor_params_candidates.empty());

    for (auto&& params: ctor_params_candidates)
      for (auto&& param: params)
        if (isa<TFAPIAttrsParam>(param.get()))
          return std::move(params);

    size_t num_params_best = 0;
    size_t best_idx = 0;
    for (size_t i = 0; i < ctor_params_candidates.size(); i++)
      if (ctor_params_candidates[i].size() > num_params_best) {
        num_params_best = ctor_params_candidates[i].size();
        best_idx = i;
      }

    return std::move(ctor_params_candidates[best_idx]);
  }

  void extractModule(CXXRecordDecl* Declaration) {
    std::string module_name = Declaration->getNameAsString();
    std::string module_name_qualified = Declaration->getQualifiedNameAsString();

    std::cout << "Generating fuzz target of API `" << module_name_qualified << "`..." << std::endl;

    //Declaration->dump();

    std::vector<std::vector<std::unique_ptr<TFParam>>> ctor_params_candidates;

    for (auto ctor: Declaration->ctors()) {
      if (auto extracted = extractModuleCtor(ctor))
        ctor_params_candidates.push_back(std::move(extracted.value()));
    }

    if (ctor_params_candidates.empty()) {
      std::cerr << "FAILED: Generating fuzz target of API `" << module_name_qualified << "` failed." << std::endl;
      return;
    }
    auto ctor_params = pickBest(std::move(ctor_params_candidates));

    TFAPI tf_api(
      module_name_qualified,
      std::move(ctor_params));

    std::string module_group_name = std::regex_replace(tf_module_list_file_name(), std::regex("::"), "_");
    output.add("basic", module_group_name, module_name, tf_api.gen_fuzz_target(FTT_Basic));
    output.add("quantization", module_group_name, module_name, tf_api.gen_fuzz_target(FTT_Quantization));
    output.add("sparse", module_group_name, module_name, tf_api.gen_fuzz_target(FTT_Sparse));
  }

  bool VisitCXXRecordDecl(CXXRecordDecl* Declaration) {
    std::string class_name = Declaration->getNameAsString();

    auto tf_module_list = get_tf_module_list();

    if (tf_module_list.find(Declaration->getQualifiedNameAsString()) != tf_module_list.end())
      extractModule(Declaration);

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
  init_tf_api_list();

  auto ExpectedParser =
      tooling::CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  output.write();

  return 0;
}
