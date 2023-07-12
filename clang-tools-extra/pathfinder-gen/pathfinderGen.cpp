#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "param.h"
#include "util.h"
#include <iostream>
#include <fstream>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;

typedef std::vector<long> tensor_rank;

const std::set<std::vector<long>> same_ranks = {{0,0},{1,1},{2,2},{3,3},{4,4},{5,5}};

std::map<std::string, std::set<tensor_rank>> target_api = {
  {"torch::nn::functional::conv1d",{{3,3,1}}},
  {"torch::nn::functional::conv2d",{{4,4,1}}},
  {"torch::nn::functional::conv3d",{{5,5,1}}},
  {"torch::nn::functional::conv_transpose1d",{{3,3,1}}},
  {"torch::nn::functional::conv_transpose2d",{{4,4,1}}},
  {"torch::nn::functional::conv_transpose3d",{{5,5,1}}},
  {"torch::nn::functional::unfold",{{4}}},
  {"torch::nn::functional::fold",{{4}}},

  {"torch::nn::functional::avg_pool1d",{{3}}},
  {"torch::nn::functional::avg_pool2d",{{4}}},
  {"torch::nn::functional::avg_pool3d",{{5}}},
  {"torch::nn::functional::max_pool1d",{{3}}},
  {"torch::nn::functional::max_pool2d",{{4}}},
  {"torch::nn::functional::max_pool3d",{{5}}},
  {"torch::nn::functional::max_unpool1d",{{3}}},
  {"torch::nn::functional::max_unpool2d",{{4}}},
  {"torch::nn::functional::max_unpool3d",{{5}}},
  {"torch::nn::functional::lp_pool1d",{{3}}},
  {"torch::nn::functional::lp_pool2d",{{4}}},
  {"torch::nn::functional::adaptive_max_pool1d",{{3}}},
  {"torch::nn::functional::adaptive_max_pool2d",{{4}}},
  {"torch::nn::functional::adaptive_max_pool3d",{{5}}},
  {"torch::nn::functional::adaptive_avg_pool1d",{{3}}},
  {"torch::nn::functional::adaptive_avg_pool2d",{{4}}},
  {"torch::nn::functional::adaptive_avg_pool3d",{{5}}},

  {"torch::nn::functional::threshold",{{}}},
  {"torch::nn::functional::threshold_",{{}}},
  {"torch::nn::functional::relu",{{}}},
  {"torch::nn::functional::relu_",{{}}},
  {"torch::nn::functional::hardtanh",{{}}},
  {"torch::nn::functional::hardtanh_",{{}}},
  {"torch::nn::functional::hardswish",{{}}},
  {"torch::nn::functional::relu6",{{}}},
  {"torch::nn::functional::elu",{{}}},
  {"torch::nn::functional::elu_",{{}}},
  {"torch::nn::functional::selu",{{}}},
  {"torch::nn::functional::celu",{{}}},
  {"torch::nn::functional::leaky_relu",{{}}},
  {"torch::nn::functional::leaky_relu_",{{}}},
  {"torch::nn::functional::prelu",{{}}},
  {"torch::nn::functional::rrelu",{{}}},
  {"torch::nn::functional::rrelu_",{{}}},
  {"torch::nn::functional::glu",{{}}},
  {"torch::nn::functional::gelu",{{}}},
  {"torch::nn::functional::logsigmoid",{{}}},
  {"torch::nn::functional::hardshrink",{{}}},
  {"torch::nn::functional::tanhshrink",{{}}},
  {"torch::nn::functional::softsign",{{}}},
  {"torch::nn::functional::softplus",{{}}},
  {"torch::nn::functional::softmin",{{}}},
  {"torch::nn::functional::softmax",{{}}},
  {"torch::nn::functional::softshrink",{{}}},
  {"torch::nn::functional::gumbel_softmax",{{}}},
  {"torch::nn::functional::log_softmax",{{}}},
  {"torch::nn::functional::tanh",{{}}},
  {"torch::nn::functional::sigmoid",{{}}},
  {"torch::nn::functional::hardsigmoid",{{}}},
  {"torch::nn::functional::silu",{{}}},
  {"torch::nn::functional::batch_norm",{{-1,1,1,1,1}}},
  {"torch::nn::functional::instance_norm",{{-1,1,1,1,1}}},
  {"torch::nn::functional::layer_norm",{{}}},
  {"torch::nn::functional::local_response_norm",{{}}},
  {"torch::nn::functional::normalize",{{}}},

  {"torch::nn::functional::linear",{{-1,1,1},{-1,2,1}}},
  {"torch::nn::functional::bilinear",{{-1,-1,3,1}}},

  {"torch::nn::functional::dropout",{{}}},
  {"torch::nn::functional::alpha_dropout",{{}}},
  {"torch::nn::functional::feature_alpha_dropout",{{4},{5}}},
  {"torch::nn::functional::dropout2d",{{3},{4}}},
  {"torch::nn::functional::dropout3d",{{4},{5}}},

  // Tensor element sensitive APIs
  {"torch::nn::functional::embedding",{{-1,2}}},
  {"torch::nn::functional::embedding_bag",{{2,2,1,2},{1,2,1,1}}},
  {"torch::nn::functional::one_hot",{{}}},

  {"torch::nn::functional::pairwise_distance",{{2,2},{1,1}}},
  {"torch::nn::functional::cosine_similarity",{same_ranks}},
  {"torch::nn::functional::pdist",{{2}}},
  {"torch::nn::functional::binary_cross_entropy",{same_ranks}},
  {"torch::nn::functional::binary_cross_entropy_with_logits",{{{0,0,1},{1,1,1},{2,2,1},{3,3,1},{4,4,1},{5,5,1}}}},
  {"torch::nn::functional::poisson_nll_loss",{same_ranks}},
  {"torch::nn::functional::cosine_embedding_loss",{{2,2},{1,1}}},
  {"torch::nn::functional::cross_entropy",{{1,0},{2,1},{3,2}}},
  {"torch::nn::functional::ctc_loss",{{3,2,1,1},{2,1,0,0}}},
  {"torch::nn::functional::hinge_embedding_loss",{same_ranks}},
  {"torch::nn::functional::kl_div",{same_ranks}},
  {"torch::nn::functional::l1_loss",{same_ranks}},
  {"torch::nn::functional::mse_loss",{same_ranks}},
  {"torch::nn::functional::margin_ranking_loss",{{1,1,1},{0,0,0}}},
  {"torch::nn::functional::multilabel_margin_loss",{{1,1},{2,2}}},
  {"torch::nn::functional::multilabel_soft_margin_loss",{{2,2}}},
  {"torch::nn::functional::multi_margin_loss",{{2,1,1},{1,0,1}}},
  {"torch::nn::functional::nll_loss",{{2,1,1},{1,0,1}}},
  {"torch::nn::functional::smooth_l1_loss",{same_ranks}},
  {"torch::nn::functional::soft_margin_loss",{same_ranks}},
  {"torch::nn::functional::triplet_margin_loss",{{2,2,2},{1,1,1}}},
  {"torch::nn::functional::triplet_margin_with_distance_loss",{{1,1,1},{2,2,2},{3,3,3},{4,4,4},{5,5,5}}},
  {"torch::nn::functional::pixel_shuffle",{{3},{4},{5}}},
  {"torch::nn::functional::pixel_unshuffle",{{3},{4},{5}}},
  {"torch::nn::functional::pad",{{}}},
  {"torch::nn::functional::interpolate",{{}}},
  {"torch::nn::functional::upsample",{{}}},
  {"torch::nn::functional::upsample_nearest",{{}}},
  {"torch::nn::functional::upsample_bilinear",{{}}},
  {"torch::nn::functional::grid_sample",{{4,4},{5,5}}},
  {"torch::nn::functional::affine_grid",{{3}}},
};
std::string current_target;
tensor_rank current_tensor_rank;
size_t current_tensor_rank_idx;
bool option_class_done;

/* size_t get_rank() {
  static const size_t DEFAULT_RANK = 3;
  if (current_tensor_rank.size() <= current_tensor_rank_idx) {
    return DEFAULT_RANK;
  } else {
    return current_tensor_rank[current_tensor_rank_idx++];
  }
} */

Optional<size_t> get_rank() {
  //static const size_t DEFAULT_RANK = 3;
  //std::cout << "rank.size(): " << std::to_string(current_tensor_rank.size()) << ", idx: " << std::to_string(current_tensor_rank_idx ) << std::endl;
  if (current_tensor_rank.size() <= current_tensor_rank_idx) {
    return None;
  } else {
    long dim = current_tensor_rank[current_tensor_rank_idx++];
    if (dim == -1)
      return None;
    else
      return (size_t)dim;
  }
}

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx);

Optional<std::unique_ptr<Param>> parseBuiltin(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse builtin\n";
  if (const auto* builtin = t->getAs<BuiltinType>()) {
    if (builtin->isInteger() || builtin->isSignedInteger() || builtin->isUnsignedInteger()) {
      //std::cout << "Builtin: " << builtin->getNameAsCString(Ctx.getPrintingPolicy()) << std::endl;
      //t->dump();
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "int" || t == "long") {
        return std::make_unique<Param>(INT);
      } else if (t == "bool") {
        return std::make_unique<Param>(BOOL);
      } else {
        //std::cout << "dd: " << t << std::endl;
        assert(false);
      }
    } else if (builtin->isFloatingPoint()) {
      std::string t = builtin->getNameAsCString(Ctx.getPrintingPolicy());
      if (t == "float" || t == "double") {
        return std::make_unique<Param>(FLOAT);
      } else {
        //std::cout << "dd: " << t << std::endl;
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

Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse int vector\n";

  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "vector") {
          auto targs = tstype->template_arguments();
          if (targs.size() == 1) {
            //targs[0].getAsType()->dump();
            auto p = parseTorchParam(targs[0].getAsType(), Ctx);
            if (p->ptype == INT)
              return std::make_unique<Param>(INTVECTOR);
            if (p->ptype == FLOAT)
              return std::make_unique<Param>(FLOATVECTOR);
          }
        }
      }
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseTensor(clang::QualType t) {
  //std::cout << "Parse tensor\n";
  if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    if (elaborated->isSugared())
      if (const auto* rtype = elaborated->desugar()->getAs<RecordType>())
        if (rtype->getDecl()->getNameAsString() == "Tensor") {
          return std::make_unique<Param>(TENSOR, get_rank());
        }
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor") {
        return std::make_unique<Param>(TENSOR, get_rank());
      }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseExpandingArray(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "Parse expending array\n";

  const RecordType* rtype;
  const TemplateSpecializationType* tstype;
  const SubstTemplateTypeParmType* sttptype;
  if ((tstype = dyn_cast<TemplateSpecializationType>(t))) {
    assert(tstype->isSugared());
    rtype = dyn_cast<RecordType>(tstype->desugar());
    if (rtype != nullptr && rtype->getDecl()->getNameAsString() == "ExpandingArray") {
      auto targ = tstype->template_arguments();
      assert(targ.size() == 1);
      int64_t expandingarray_size =
        targ[0].getAsExpr()->getIntegerConstantExpr(Ctx).getValue().getExtValue();
      //t->dump();
      return std::make_unique<Param>(EXPANDINGARRAY, expandingarray_size);
    }
  } else if ((sttptype = dyn_cast<SubstTemplateTypeParmType>(t))) {
    assert(sttptype->isSugared());
    //std::cout << "sttptype:\n" << std::endl;
    //sttptype->dump();
    //std::cout << "sttptype->getReplacementType:\n";
    //sttptype->getReplacementType()->dump();
    if (const auto* rtype2 = dyn_cast<RecordType>(sttptype->getReplacementType())) {
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype2->getDecl())) {
        if (//ctsdecl->getNameAsString() == "ExpendingArray" ||
            ctsdecl->getNameAsString() == "ExpandingArrayWithOptionalElem") {
          //ctsdecl->dump();
          const auto& targ = ctsdecl->getTemplateArgs();
          assert(targ.size() == 2);
          //bool isExpr = targ[0].getAsExpr() != nullptr;
          //std::cout << "is expr: " << isExpr << std::endl;
          assert(targ[0].getKind() == TemplateArgument::ArgKind::Integral);
          int64_t expandingarray_size =
            targ[0].getAsIntegral().getExtValue();
          return std::make_unique<Param>(EXPANDINGARRAY, expandingarray_size);
        }
      }
      /* std::cout << "sttptype->getReplacementType()->getDecl(): " << rtype2->getDecl()->getNameAsString() <<  "\n";
      rtype2->getDecl()->dump();
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype2->getDecl())) {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 2);
        targ[0].dump();
        targ[1].dump();
      } */
    }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, ASTContext &Ctx) {
  if (const auto* tstype = dyn_cast<TemplateSpecializationType>(t)) {
    if (tstype->isSugared()) {
      if (const auto* rtype = dyn_cast<RecordType>(tstype->desugar())) {
        if (rtype->getDecl()->getNameAsString() == "optional") {
          //std::cout << "optional:\n";
          auto targ = tstype->template_arguments();
          assert(targ.size() == 1);
          //targ[0].getAsType()->dump();
          auto p = parseTorchParam(targ[0].getAsType(), Ctx);
          //TODO: dtype
          if (p != nullptr)
            return std::make_unique<Param>(OPTIONAL, std::move(p));
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
                //std::vector<std::unique_ptr<Param>> types;
                //std::vector<std::string> enum_vec;
                std::vector<std::unique_ptr<Param>> enums;
                std::unique_ptr<Param> expandingarray;
                for (auto targ: targs) {
                  auto p = parseTorchParam(targ.getAsType(), Ctx);
                  if (p->ptype == ENUM) {
                    enums.push_back(std::move(p));
                  } else if (p->ptype == EXPANDINGARRAY) {
                    expandingarray = std::move(p);
                  } else {
                    assert(false);
                  }
                }
                assert(enums.size() > 0);
                if (expandingarray != nullptr) {
                  return std::make_unique<Param>(VARIANT, std::vector<std::unique_ptr<Param>>(), std::move(expandingarray));
                } else {
                  return std::make_unique<Param>(VARIANT, std::move(enums), nullptr);
                }
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

Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, ASTContext &Ctx) {
  if (option_class_done) return None;

  //std::cout << "Parse API Option\n";
  static std::string option_suffix = "Options";
  //static std::set<std::string> seen;

  std::string api_option_name;
  const RecordType* rtype;
  if (const auto* tdtype = dyn_cast<TypedefType>(t)) {
    api_option_name = tdtype->getDecl()->getQualifiedNameAsString();
    if (api_option_name.length() > option_suffix.length() &&
        api_option_name.compare(
          api_option_name.length() - option_suffix.length(),
          option_suffix.length(), option_suffix) == 0) {
      option_class_done = true;
      //std::cout << "tdtype\n";
      //tdtype->dump();
      assert(tdtype->isSugared());

      const TemplateSpecializationType* tstype;
      if (const auto* tdtype2 = dyn_cast<TypedefType>(tdtype->desugar())) {
        assert(tdtype2->isSugared());
        tstype = dyn_cast<TemplateSpecializationType>(tdtype2->desugar());
      } else {
        tstype = dyn_cast<TemplateSpecializationType>(tdtype->desugar());
      }

      if (tstype != nullptr) {
        //std::cout << "tstype\n";
        //tstype->dump();
        assert(tstype->isSugared());
        rtype = dyn_cast<RecordType>(tstype->desugar());
        //std::cout << "rtype\n";
        //rtype->dump();
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
    }
  }

  if (rtype == nullptr || !option_class_done) return None;

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  assert(cdecl != nullptr);
  //std::cout << "cdecl\n";
  //cdecl->dump();
  std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries;
  for (auto method: cdecl->methods()) {
    std::string param_name = method->getNameAsString();
    if (param_name == cdecl->getNameAsString())
      continue;
    if (param_name == "operator=")
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
    //std::string param_name = method->getNameInfo().getAsString();
    //std::cout << "method name: " << param_name << std::endl;
    //std::cout << "param type:\n" << std::endl;
    //method->parameters()[0]->getType()->dump();
    std::unique_ptr<Param> param = parseTorchParam(method->parameters()[0]->getType(), Ctx);
    if (param != nullptr) {
      if (param->ptype == TENSOR)
        param = std::make_unique<Param>(
          OPTIONAL,
          std::move(param));
      param->set_default(param_name, cdecl);
      //entries.push_back({param_name, std::move(param)});
      push_back_unique(entries, param_name, std::move(param));
    }
  }
  return
    std::make_unique<Param>(
      MAP,
      api_option_name,
      std::move(entries));
}

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx) {
  if (auto builtin_opt = parseBuiltin(t, Ctx))
    return std::move(builtin_opt.getValue());
  if (auto enum_opt = parseEnum(t, Ctx))
    return std::move(enum_opt.getValue());
  if (auto int_vec_opt = parseVector(t, Ctx))
    return std::move(int_vec_opt.getValue());
  if (auto tensor_opt = parseTensor(t))
    return std::move(tensor_opt.getValue());
  if (auto expandingarray_opt = parseExpandingArray(t, Ctx))
    return std::move(expandingarray_opt.getValue());
  if (auto optional_opt = parseOptional(t, Ctx))
    return std::move(optional_opt.getValue());
  if (auto variant_opt = parseVariant(t, Ctx))
    return std::move(variant_opt.getValue());
  if (auto api_option_opt = parseMAP(t, Ctx))
    return std::move(api_option_opt.getValue());

  // simplify
  if (t->isLValueReferenceType()) {
    return parseTorchParam(t->getAs<LValueReferenceType>()->getPointeeType(), Ctx);
  } else if (t->isRValueReferenceType()) {
    return parseTorchParam(t->getAs<RValueReferenceType>()->getPointeeType(), Ctx);
  } else if (const auto* elaborated = t->getAs<ElaboratedType>()) {
    //std::cout << "elaborated~~\n";
    if (elaborated->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(elaborated->desugar(), Ctx);
    }
  } else if (const auto* tdtype = t->getAs<TypedefType>()) {
    //std::cout << "typedeftype~~\n";
    if (tdtype->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(tdtype->desugar(), Ctx);
    }
  } else if (const auto* tstype = t->getAs<TemplateSpecializationType>()) {
    //std::cout << "templateSpecialized~~\n";
    if (tstype->isSugared()) {
      //std::cout << "is sugared~~\n";
      return parseTorchParam(tstype->desugar(), Ctx);
    }
  }

  //t->dump();
  return nullptr;
}

std::vector<std::pair<std::string, std::string>> file_buffer;

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    //if (const FunctionDecl *FD = Result.Nodes.getNodeAs<clang::FunctionDecl>("func")){
    //if (Declaration->getQualifiedNameAsString() == target_api) {
    auto api_it = target_api.find(Declaration->getQualifiedNameAsString());
    if (api_it != target_api.end()) {
      current_target = Declaration->getQualifiedNameAsString();
      std::string current_target_unqualified = Declaration->getNameAsString();
      size_t api_id = 0;
      for (auto rank: api_it->second) {
        current_tensor_rank = rank;
        current_tensor_rank_idx = 0;
        option_class_done = false;
        auto param_decls = Declaration->parameters();
        std::vector<std::unique_ptr<Param>> params;
        for (auto param_decl: param_decls) {
          //std::cout << "param: " << param_decl->getNameAsString() << std::endl;
          clang::QualType t = param_decl->getType();
          //t->dump();
          std::unique_ptr<Param> p = parseTorchParam(t, *Context);
          if (p != nullptr)
            params.push_back(std::move(p));
        }

        std::string filename =
          "pathfinder_fuzz_target_" +
          current_target_unqualified + "_" + std::to_string(api_id) +
          ".cpp";
        std::string code = gen_code(current_target, params);
        file_buffer.push_back(std::make_pair(filename, code));
        api_id++;
      }
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
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  std::string cmake_contents;
  for (auto p: file_buffer) {
    std::string filename = p.first;
    std::cout << filename << std::endl;
    std::string code = p.second;
    std::ofstream writeFile(filename);
    if(writeFile.is_open()){
      writeFile << code;
      writeFile.close();
      cmake_contents += "add_pathfinder_fuzz_target(" + strip_ext(filename) + ")\n";
    } else {
      assert(false);
    }
  }

  std::ofstream writeFile("CMakeLists.txt");
  if(writeFile.is_open()){
    writeFile << cmake_contents;
    writeFile.close();
  } else {
    assert(false);
  }

  return 0;
}