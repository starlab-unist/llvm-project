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

std::set<std::string> functional_common = {
  "torch::nn::functional::conv1d",
  "torch::nn::functional::conv2d",
  "torch::nn::functional::conv3d",
  "torch::nn::functional::conv_transpose1d",
  "torch::nn::functional::conv_transpose2d",
  "torch::nn::functional::conv_transpose3d",
  "torch::nn::functional::unfold",
  "torch::nn::functional::fold",

  "torch::nn::functional::avg_pool1d",
  "torch::nn::functional::avg_pool2d",
  "torch::nn::functional::avg_pool3d",
  "torch::nn::functional::max_pool1d",
  "torch::nn::functional::max_pool2d",
  "torch::nn::functional::max_pool3d",
  "torch::nn::functional::max_unpool1d",
  "torch::nn::functional::max_unpool2d",
  "torch::nn::functional::max_unpool3d",
  "torch::nn::functional::lp_pool1d",
  "torch::nn::functional::lp_pool2d",
  "torch::nn::functional::adaptive_max_pool1d",
  "torch::nn::functional::adaptive_max_pool2d",
  "torch::nn::functional::adaptive_max_pool3d",
  "torch::nn::functional::adaptive_avg_pool1d",
  "torch::nn::functional::adaptive_avg_pool2d",
  "torch::nn::functional::adaptive_avg_pool3d",

  "torch::nn::functional::threshold",
  "torch::nn::functional::threshold_",
  "torch::nn::functional::relu",
  "torch::nn::functional::relu_",
  "torch::nn::functional::hardtanh",
  "torch::nn::functional::hardtanh_",
  "torch::nn::functional::hardswish",
  "torch::nn::functional::relu6",
  "torch::nn::functional::elu",
  "torch::nn::functional::elu_",
  "torch::nn::functional::selu",
  "torch::nn::functional::celu",
  "torch::nn::functional::leaky_relu",
  "torch::nn::functional::leaky_relu_",
  "torch::nn::functional::prelu",
  "torch::nn::functional::rrelu",
  "torch::nn::functional::rrelu_",
  "torch::nn::functional::glu",
  "torch::nn::functional::gelu",
  "torch::nn::functional::logsigmoid",
  "torch::nn::functional::hardshrink",
  "torch::nn::functional::tanhshrink",
  "torch::nn::functional::softsign",
  "torch::nn::functional::softplus",
  "torch::nn::functional::softmin",
  "torch::nn::functional::softmax",
  "torch::nn::functional::softshrink",
  "torch::nn::functional::gumbel_softmax",
  "torch::nn::functional::log_softmax",
  "torch::nn::functional::tanh",
  "torch::nn::functional::sigmoid",
  "torch::nn::functional::hardsigmoid",
  "torch::nn::functional::silu",
  "torch::nn::functional::batch_norm",
  "torch::nn::functional::instance_norm",
  "torch::nn::functional::layer_norm",
  "torch::nn::functional::local_response_norm",
  "torch::nn::functional::normalize",

  "torch::nn::functional::linear",
  "torch::nn::functional::bilinear",

  "torch::nn::functional::dropout",
  "torch::nn::functional::alpha_dropout",
  "torch::nn::functional::feature_alpha_dropout",
  "torch::nn::functional::dropout2d",
  "torch::nn::functional::dropout3d",

  // Tensor element sensitive APIs
  "torch::nn::functional::embedding",
  "torch::nn::functional::embedding_bag",
  "torch::nn::functional::one_hot",

  "torch::nn::functional::pairwise_distance",
  "torch::nn::functional::cosine_similarity",
  "torch::nn::functional::pdist",
  "torch::nn::functional::binary_cross_entropy",
  "torch::nn::functional::binary_cross_entropy_with_logits",
  "torch::nn::functional::poisson_nll_loss",
  "torch::nn::functional::cosine_embedding_loss",
  "torch::nn::functional::cross_entropy",
  "torch::nn::functional::ctc_loss",
  "torch::nn::functional::hinge_embedding_loss",
  "torch::nn::functional::kl_div",
  "torch::nn::functional::l1_loss",
  "torch::nn::functional::mse_loss",
  "torch::nn::functional::margin_ranking_loss",
  "torch::nn::functional::multilabel_margin_loss",
  "torch::nn::functional::multilabel_soft_margin_loss",
  "torch::nn::functional::multi_margin_loss",
  "torch::nn::functional::nll_loss",
  "torch::nn::functional::smooth_l1_loss",
  "torch::nn::functional::soft_margin_loss",
  "torch::nn::functional::triplet_margin_loss",
  "torch::nn::functional::triplet_margin_with_distance_loss",
  "torch::nn::functional::pixel_shuffle",
  "torch::nn::functional::pixel_unshuffle",
  "torch::nn::functional::pad",
  "torch::nn::functional::interpolate",
  "torch::nn::functional::upsample",
  "torch::nn::functional::upsample_nearest",
  "torch::nn::functional::upsample_bilinear",
  "torch::nn::functional::grid_sample",
  "torch::nn::functional::affine_grid",
};

std::set<std::string> functional_additional = {
  "torch::nn::functional::adaptive_max_pool2d_with_indices",
  "torch::nn::functional::fractional_max_pool2d",
  "torch::nn::functional::fractional_max_pool2d_with_indices",
  "torch::nn::functional::fractional_max_pool3d",
  "torch::nn::functional::fractional_max_pool3d_with_indices",
  "torch::nn::functional::group_norm",
  "torch::nn::functional::huber_loss",
  "torch::nn::functional::max_pool1d_with_indices",
  "torch::nn::functional::max_pool2d_with_indices",
  "torch::nn::functional::max_pool3d_with_indices",
  "torch::nn::functional::mish",
  "torch::nn::functional::multi_head_attention_forward",
};

std::set<std::string> at = {
  "at::abs",
  "at::abs_",
  "at::absolute",
  "at::acos",
  "at::acos_",
  "at::abs",
  "at::absolute",
  "at::acos",
  "at::acosh",
  "at::addbmm",
  "at::addcdiv",
  "at::addcmul",
  "at::addmm",
  "at::addmv",
  "at::addr",
  "at::adjoint",
  "at::affine_grid_generator",
  "at::alias",
  "at::align_tensors",
  "at::all",
  "at::allclose",
  "at::alpha_dropout",
  "at::amax",
  "at::amin",
  "at::aminmax",
  "at::angle",
  "at::any",
  "at::arccos",
  "at::arccosh",
  "at::arcsin",
  "at::arcsinh",
  "at::arctan",
  "at::arctan2",
  "at::arctanh",
  "at::argmax",
  "at::argmin",
  "at::argsort",
  "at::argsort",
  "at::argwhere",
  "at::as_strided",
  "at::as_strided_scatter",
  "at::asin",
  "at::asinh",
  "at::atan",
  "at::atan2",
  "at::atanh",
  "at::atleast_1d",
  "at::atleast_2d",
  "at::atleast_3d",
};

std::set<std::string> linalg = {
  "torch::linalg::cholesky",
  "torch::linalg::cholesky_out",
  "torch::linalg::det",
  "torch::linalg::eig",
  "torch::linalg::eig_out",
  //{"torch::linalg::eigh",
  //{"torch::linalg::eigh_out",
  "torch::linalg::eigvals",
  "torch::linalg::eigvals_out",
  //{"torch::linalg::eigvalsh",
  //{"torch::linalg::eigvalsh_out",
  "torch::linalg::householder_product",
  "torch::linalg::householder_product_out",
  "torch::linalg::inv",
  "torch::linalg::inv_out",
  "torch::linalg::ldl_factor_ex",
  "torch::linalg::ldl_factor_ex_out",
  "torch::linalg::ldl_solve",
  "torch::linalg::ldl_solve_out",
  "torch::linalg::linalg_det",
  "torch::linalg::linalg_norm",
  "torch::linalg::linalg_norm_out",
  "torch::linalg::lstsq",
  "torch::linalg::lu",
  "torch::linalg::lu_factor",
  "torch::linalg::lu_factor_out",
  "torch::linalg::lu_out",
  "torch::linalg::matrix_exp",
  "torch::linalg::matrix_norm",
  "torch::linalg::matrix_norm_out",
  "torch::linalg::matrix_power",
  "torch::linalg::matrix_power_out",
  "torch::linalg::matrix_rank",
  "torch::linalg::matrix_rank_out",
  "torch::linalg::multi_dot",
  "torch::linalg::multi_dot_out",
  "torch::linalg::norm",
  "torch::linalg::norm_out",
  "torch::linalg::pinv",
  "torch::linalg::pinv_out",
  "torch::linalg::qr",
  "torch::linalg::qr_out",
  "torch::linalg::slogdet",
  "torch::linalg::slogdet_out",
  "torch::linalg::solve",
  "torch::linalg::solve_ex",
  "torch::linalg::solve_ex_out",
  "torch::linalg::solve_out",
  "torch::linalg::solve_triangular",
  "torch::linalg::solve_triangular_out",
  "torch::linalg::svd",
  "torch::linalg::svd_out",
  "torch::linalg::svdvals",
  "torch::linalg::svdvals_out",
  "torch::linalg::tensorinv",
  "torch::linalg::tensorinv_out",
  "torch::linalg::tensorsolve",
  "torch::linalg::tensorsolve_out",
  "torch::linalg::vector_norm",
  "torch::linalg::vector_norm_out",
};

std::set<std::string> module = {
  // Convolution Layers
  "torch::nn::Conv1d",
  "torch::nn::Conv2d",
  "torch::nn::Conv3d",
  "torch::nn::ConvTranspose1d",
  "torch::nn::ConvTranspose2d",
  "torch::nn::ConvTranspose3d",
  "torch::nn::Unfold",
  "torch::nn::Fold",

  // Pooling Layers
  "torch::nn::MaxPool1d",
  "torch::nn::MaxPool2d",
  "torch::nn::MaxPool3d",
  "torch::nn::MaxUnpool1d",
  "torch::nn::MaxUnpool2d",
  "torch::nn::MaxUnpool3d",
  "torch::nn::AvgPool1d",
  "torch::nn::AvgPool2d",
  "torch::nn::AvgPool3d",
  "torch::nn::FractionalMaxPool2d",
  "torch::nn::FractionalMaxPool3d",
  "torch::nn::LPPool1d",
  "torch::nn::LPPool2d",
  "torch::nn::AdaptiveMaxPool1d",
  "torch::nn::AdaptiveMaxPool2d",
  "torch::nn::AdaptiveMaxPool3d",
  "torch::nn::AdaptiveAvgPool1d",
  "torch::nn::AdaptiveAvgPool2d",
  "torch::nn::AdaptiveAvgPool3d",

  // Padding Layers
  "torch::nn::ReflectionPad1d",
  "torch::nn::ReflectionPad2d",
  "torch::nn::ReflectionPad3d",
  "torch::nn::ReplicationPad1d",
  "torch::nn::ReplicationPad2d",
  "torch::nn::ReplicationPad3d",
  "torch::nn::ZeroPad1d",
  "torch::nn::ZeroPad2d",
  "torch::nn::ZeroPad3d",
  "torch::nn::ConstantPad1d",
  "torch::nn::ConstantPad2d",
  "torch::nn::ConstantPad3d",

  // Non-linear Activations (weighted sum, nonlinearity)
  "torch::nn::ELU",
  "torch::nn::Hardshrink",
  "torch::nn::Hardtanh",
  "torch::nn::LeakyReLU",
  "torch::nn::LogSigmoid",
  //{"torch::nn::MultiheadAttention",
  "torch::nn::PReLU",
  "torch::nn::ReLU",
  "torch::nn::ReLU6",
  "torch::nn::RReLU",
  "torch::nn::SELU",
  "torch::nn::CELU",
  "torch::nn::Sigmoid",
  "torch::nn::SiLU",
  "torch::nn::Mish",
  "torch::nn::Softplus",
  "torch::nn::Softshrink",
  "torch::nn::Softsign",
  "torch::nn::Tanh",
  "torch::nn::Tanhshrink",
  "torch::nn::Threshold",
  "torch::nn::GLU",

  // Non-linear Activations (other)
  "torch::nn::Softmin",
  "torch::nn::Softmax",
  "torch::nn::Softmax2d",
  "torch::nn::LogSoftmax",
  "torch::nn::AdaptiveLogSoftmaxWithLoss",

  // Normalization Layers
  "torch::nn::BatchNorm1d",
  "torch::nn::BatchNorm2d",
  "torch::nn::BatchNorm3d",
  "torch::nn::GroupNorm",
  "torch::nn::InstanceNorm1d",
  "torch::nn::InstanceNorm2d",
  "torch::nn::InstanceNorm3d",
  "torch::nn::LayerNorm",
  "torch::nn::LocalResponseNorm",

  // Recurrent Layers
  "torch::nn::RNN",
  "torch::nn::LSTM",
  "torch::nn::GRU",
  "torch::nn::RNNCell",
  "torch::nn::LSTMCell",
  "torch::nn::GRUCell",

  // Transformer Layers
  //{"torch::nn::Transformer",
  //{"torch::nn::TransformerEncoder",
  //{"torch::nn::TransformerDecoder",
  //{"torch::nn::TransformerEncoderLayer",
  //{"torch::nn::TransformerDecoderLayer",

  // Linear Layers
  "torch::nn::Identity",
  "torch::nn::Linear",
  "torch::nn::Bilinear",

  // Dropout Layers
  "torch::nn::Dropout",
  "torch::nn::Dropout2d",
  "torch::nn::Dropout3d",
  "torch::nn::AlphaDropout",
  "torch::nn::FeatureAlphaDropout",

  // Sparse Layers
  "torch::nn::Embedding",
  "torch::nn::EmbeddingBag",

  // Distance Functions
  "torch::nn::CosineSimilarity",
  "torch::nn::PairwiseDistance",

  // Loss Functions
  "torch::nn::L1Loss",
  "torch::nn::MSELoss",
  "torch::nn::CrossEntropyLoss",
  "torch::nn::CTCLoss",
  "torch::nn::NLLLoss",
  "torch::nn::PoissonNLLLoss",
  "torch::nn::KLDivLoss",
  "torch::nn::BCELoss",
  "torch::nn::BCEWithLogitsLoss",
  "torch::nn::MarginRankingLoss",
  "torch::nn::HingeEmbeddingLoss",
  "torch::nn::MultiLabelMarginLoss",
  "torch::nn::HuberLoss",
  "torch::nn::SmoothL1Loss",
  "torch::nn::SoftMarginLoss",
  "torch::nn::MultiLabelSoftMarginLoss",
  "torch::nn::CosineEmbeddingLoss",
  "torch::nn::MultiMarginLoss",
  "torch::nn::TripletMarginLoss",
  "torch::nn::TripletMarginWithDistanceLoss",
  
  // Vision Layers
  "torch::nn::PixelShuffle",
  "torch::nn::PixelUnshuffle",
  "torch::nn::Upsample",
  
  // Utilities
  "torch::nn::Flatten",
  "torch::nn::Unflatten",
  
  // etc
  "torch::nn::CrossMapLRN2d",
};

size_t num_input_tensor(std::string& module_name) {
  if (endswith(module_name, "Loss")) {
    if (module_name == "torch::nn::CTCLoss") {
      return 4;
    } else if (
        module_name == "torch::nn::TripletMarginLoss" ||
        module_name == "torch::nn::TripletMarginWithDistanceLoss" ||
        module_name == "torch::nn::CosineEmbeddingLoss" ||
        module_name == "torch::nn::MarginRankingLoss") {
      return 3;
    } else {
      return 2;
    }
  } else {
    if (module_name == "torch::nn::LSTM" ||
        module_name == "torch::nn::LSTMCell") {
      return 3;
    } else if (
        module_name == "torch::nn::MaxUnpool1d" ||
        module_name == "torch::nn::MaxUnpool2d" ||
        module_name == "torch::nn::MaxUnpool3d" ||
        module_name == "torch::nn::CosineSimilarity" ||
        module_name == "torch::nn::PairwiseDistance" ||
        module_name == "torch::nn::GRU" ||
        module_name == "torch::nn::GRUCell" ||
        module_name == "torch::nn::RNN" ||
        module_name == "torch::nn::RNNCell" ||
        module_name == "torch::nn::Bilinear") {
      return 2;
    } else {
      return 1;
    }
  }
}

std::string current_target;
bool option_class_done;

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

Optional<std::unique_ptr<Param>> parseDtype(clang::QualType t, ASTContext &Ctx) {
  if (const auto* etype = t->getAs<EnumType>()) {
    //std::cout << "EnumType: " << etype->getDecl()->getNameAsString() << "\n";
    //std::cout << "          " << etype->getDecl()->getQualifiedNameAsString() << "\n";
    //etype->getDecl()->dump();
    if (etype->getDecl()->getNameAsString() == "ScalarType") {
      return std::make_unique<Param>(DTYPE);
    }
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
            if (auto p = parseTorchParam(targs[0].getAsType(), Ctx)) {
              if (p->ptype == INT)
                return std::make_unique<Param>(INTVECTOR);
              if (p->ptype == FLOAT)
                return std::make_unique<Param>(FLOATVECTOR);
            }
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
          return std::make_unique<Param>(TENSOR);
        }
  } else if (const auto* underlying = t.getTypePtrOrNull()) {
    if (const auto* rtype = underlying->getAs<RecordType>())
      if (rtype->getDecl()->getNameAsString() == "Tensor") {
        return std::make_unique<Param>(TENSOR);
      }
  }

  return None;
}

Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, ASTContext &Ctx) {
  if (const auto* rtype = dyn_cast<RecordType>(t)) {
    if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
      if (ctsdecl->getNameAsString() == "ArrayRef") {
        const auto& targ = ctsdecl->getTemplateArgs();
        assert(targ.size() == 1);
        assert(targ[0].getKind() == TemplateArgument::ArgKind::Type);
        auto p = parseTorchParam(targ[0].getAsType(), Ctx);
        if (p->ptype == INT)
          return std::make_unique<Param>(INTARRAYREF);
      }
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
      //rtype->dump();
      auto targ = tstype->template_arguments();
      if (targ.size() != 1)
        return None;
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
        //if (ctsdecl->getNameAsString() == "ExpandingArrayWithOptionalElem") {
        if (ctsdecl->getNameAsString() == "ExpandingArray") {
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

Optional<std::unique_ptr<Param>> parseExpandingArrayWithOptionalElem(clang::QualType t, ASTContext &Ctx) {
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
      return std::make_unique<Param>(EXPANDINGARRAYWITHOPTIONALELEM, expandingarray_size);
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
          return std::make_unique<Param>(EXPANDINGARRAYWITHOPTIONALELEM, expandingarray_size);
        }
      }
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
                  } else if (p->ptype == EXPANDINGARRAY || p->ptype == EXPANDINGARRAYWITHOPTIONALELEM) {
                    expandingarray = std::move(p);
                  } else {
                    assert(false);
                  }
                }
                assert(enums.size() > 0);
                /* if (expandingarray != nullptr) {
                  return std::make_unique<Param>(VARIANT, std::vector<std::unique_ptr<Param>>(), std::move(expandingarray));
                } else {
                  return std::make_unique<Param>(VARIANT, std::move(enums), nullptr);
                } */
                //return std::make_unique<Param>(VARIANT, std::move(enums), nullptr);
                return std::make_unique<Param>(VARIANT, std::move(enums), std::move(expandingarray));
              }
            }
          }
        }
      //}
    //}
  //}

  return None;
}

std::string get_specialized_name(const ClassTemplateSpecializationDecl* ctsdecl) {
  std::cout << "get_specialized_name:\n";
  ctsdecl->dump();
  std::string name = ctsdecl->getQualifiedNameAsString() + "<";

  const auto& targs = ctsdecl->getTemplateArgs();
  for (size_t i = 0; i < targs.size(); i++) {
    if (targs[i].getKind() == TemplateArgument::ArgKind::Type) {
      auto t = targs[i].getAsType();
      std::cout << "t: " << t.getAsString() << std::endl;
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

Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, ASTContext &Ctx) {
  if (option_class_done) return None;

  //std::cout << "Parse API Option\n";
  //t->dump();
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
      if (const auto* ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(rtype->getDecl())) {
        std::string name = get_specialized_name(ctsdecl);
        std::cout << "============================================================\n";
        std::cout << "specialized name: " << name << std::endl;
        std::cout << "============================================================\n";
        api_option_name = name;
      }
    }
  }

  if (rtype == nullptr || !option_class_done) return None;

  //std::cout << "============================================================\n";
  //std::cout << "type: " << t.getAsString() << std::endl;
  //std::cout << "============================================================\n";

  const auto* cdecl = rtype->getAsCXXRecordDecl();
  assert(cdecl != nullptr);
  std::cout << "cdecl:\n";
  cdecl->dump();
  std::vector<std::pair<std::string,std::unique_ptr<Param>>> ctor_params;
  std::vector<std::pair<std::string,std::unique_ptr<Param>>> entries;
  std::vector<std::string> ctor_param_names;
  for (auto method: cdecl->methods()) {
    //method->dump();
    if (const auto* cxxconstructordecl = dyn_cast<CXXConstructorDecl>(method)) {
      bool is_special_ctor =
        cxxconstructordecl->isDefaultConstructor() ||
        cxxconstructordecl->isCopyOrMoveConstructor() ||
        cxxconstructordecl->isSpecializationCopyingObject() ||
        cxxconstructordecl->isInheritingConstructor();
      if (!is_special_ctor) {
        //cxxconstructordecl->dump();
        for (const auto* param: cxxconstructordecl->parameters()) {
          //param->dump();
          //std::cout << param->getNameAsString() << std::endl;
          ctor_param_names.push_back(param->getNameAsString());
        }
      }
      continue;
    }
    std::string param_name = method->getNameAsString();
    //if (param_name == cdecl->getNameAsString())
    //  continue;
    //if (param_name == "operator=")
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
      //std::cout << "cdecl:\n";
      //cdecl->dump();
      param->set_default(param_name, cdecl);
      //entries.push_back({param_name, std::move(param)});
      if (include(ctor_param_names, param_name))
        push_back_unique(ctor_params, param_name, std::move(param));
      else
        push_back_unique(entries, param_name, std::move(param));
    }
  }
  return
    std::make_unique<Param>(
      MAP,
      api_option_name,
      std::move(ctor_params),
      std::move(entries));
}

std::unique_ptr<Param> parseTorchParam(clang::QualType t, ASTContext &Ctx) {
  //std::cout << "parseTorchParam\n";
  //t->dump();
  if (auto builtin_opt = parseBuiltin(t, Ctx))
    return std::move(builtin_opt.getValue());
  if (auto dtype_opt = parseDtype(t, Ctx))
    return std::move(dtype_opt.getValue());
  if (auto enum_opt = parseEnum(t, Ctx))
    return std::move(enum_opt.getValue());
  if (auto int_vec_opt = parseVector(t, Ctx))
    return std::move(int_vec_opt.getValue());
  if (auto tensor_opt = parseTensor(t))
    return std::move(tensor_opt.getValue());
  if (auto intarrayref_opt = parseIntArrayRef(t, Ctx))
    return std::move(intarrayref_opt.getValue());
  if (auto expandingarray_opt = parseExpandingArray(t, Ctx))
    return std::move(expandingarray_opt.getValue());
  if (auto expandingarraywithoptionalelem_opt = parseExpandingArrayWithOptionalElem(t, Ctx))
    return std::move(expandingarraywithoptionalelem_opt.getValue());
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

std::vector<std::pair<std::string, std::string>> functional_common_files_pathfinder;
std::vector<std::pair<std::string, std::string>> functional_additional_files_pathfinder;
std::vector<std::pair<std::string, std::string>> at_files_pathfinder;
std::vector<std::pair<std::string, std::string>> module_files_pathfinder;

std::vector<std::pair<std::string, std::string>> functional_common_files_libfuzzer;

void set_union(std::set<std::string>& orig,std::set<std::string>& other) {
  orig.insert(other.begin(), other.end());
}

bool in(std::string api_name, std::set<std::string> apis) {
  return apis.find(api_name) != apis.end();
}

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    std::set<std::string> target_api;
    set_union(target_api, functional_common);
    set_union(target_api, functional_additional);
    set_union(target_api, at);

    std::string fname = Declaration->getNameAsString();
    std::string fname_qualified = Declaration->getQualifiedNameAsString();

    if (target_api.find(fname_qualified) == target_api.end())
      return true;

    std::cout << "Generating fuzz target of API `" << fname_qualified << "`..." << std::endl;

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

    if (params.empty()) {
      std::cout << "FAILED: Generating fuzz target of API `" << fname_qualified << "` failed." << std::endl;
      return true;
    }

    std::string code_pathfinder = gen_torch_function_pathfinder(fname_qualified, params);
    if (in (fname_qualified, functional_common))
      functional_common_files_pathfinder.push_back(std::make_pair(fname, code_pathfinder));
    else if (in (fname_qualified, functional_additional))
      functional_additional_files_pathfinder.push_back(std::make_pair(fname, code_pathfinder));
    else if (in (fname_qualified, at))
      at_files_pathfinder.push_back(std::make_pair(fname, code_pathfinder));

    std::string code_libfuzzer = gen_torch_function_libfuzzer(fname_qualified, params);
    if (in (fname_qualified, functional_common))
      functional_common_files_libfuzzer.push_back(std::make_pair(fname, code_libfuzzer));

    return true;
  }

  Optional<std::vector<std::unique_ptr<Param>>> parseModuleCtor(CXXConstructorDecl* ctor, size_t num_input_tensor) {
    std::cout << "ctor:\n";
    ctor->dump();

    if (//ctor->isDefaultConstructor() ||
        ctor->isCopyOrMoveConstructor() ||
        ctor->isSpecializationCopyingObject() ||
        ctor->isInheritingConstructor())
      return None;

    option_class_done = false;
    std::vector<std::unique_ptr<Param>> params;
    for (size_t i = 0; i < num_input_tensor; i++)
      params.push_back(std::make_unique<Param>(TENSOR));
    for (const auto* param: ctor->parameters()) {
      clang::QualType t = param->getType();
      //t->dump();
      std::unique_ptr<Param> p = parseTorchParam(t, *Context);
      if (p != nullptr)
        params.push_back(std::move(p));
    }
    return params;
  }

  std::vector<std::unique_ptr<Param>> pickBest(std::vector<std::vector<std::unique_ptr<Param>>> candidates) {
    assert(!candidates.empty());

    for (auto&& params: candidates)
      for (auto&& param: params)
        if (param->is_map())
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
    auto target_api = module;
    auto api_it = target_api.find(Declaration->getQualifiedNameAsString());
    if (api_it == target_api.end())
      return true;

    current_target = Declaration->getQualifiedNameAsString();
    std::cout << current_target << std::endl;
    std::string current_target_unqualified = Declaration->getNameAsString();

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
    std::cout << "========================================================================\n";
    std::cout << "class_decl:\n";
    class_decl->dump();
    std::cout << "========================================================================\n";

    std::vector<std::vector<std::unique_ptr<Param>>> candidates;

    for (auto ctor: class_decl->ctors())
      if (auto parsed = parseModuleCtor(ctor, num_input_tensor(current_target)))
        candidates.push_back(std::move(parsed.getValue()));
    if (!class_decl->bases().empty()) {
      std::cout << "base type:\n";
      class_decl->bases_begin()->getType()->dump();
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
            //std::cout << "normimplbase:\n";
            //ctsdecl->bases_begin()->getType()->getAs<RecordType>()->getDecl()->dump();
            auto targs = ctsdecl->bases_begin()->getType()->getAs<TemplateSpecializationType>()->template_arguments();
            assert(targs.size() == 3 && targs[2].getKind() == TemplateArgument::ArgKind::Type);
            option_class_done = false;
            std::vector<std::unique_ptr<Param>> params;
            for (size_t i = 0; i < num_input_tensor(current_target); i++)
              params.push_back(std::make_unique<Param>(TENSOR));
            if (auto p = parseTorchParam(targs[2].getAsType(), *Context))
              params.push_back(std::move(p));
            candidates.push_back(std::move(params));
          } else {
            std::cout << "ctsdecl:\n";
            ctsdecl->dump();
            for (auto ctor: ctsdecl->ctors())
              if (auto parsed = parseModuleCtor(ctor, num_input_tensor(current_target)))
                candidates.push_back(std::move(parsed.getValue()));
          }
        }
      }
    }

    auto params = pickBest(std::move(candidates));

    std::string code_pathfinder = gen_torch_module_pathfinder(current_target, params, num_input_tensor(current_target));
    module_files_pathfinder.push_back(std::make_pair(current_target_unqualified, code_pathfinder));

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

bool file_exist(std::string filename, bool is_libfuzzer) {
  static std::set<std::string> generated_pathfinder;
  static std::set<std::string> generated_libfuzzer;

  if (is_libfuzzer)
    if (generated_libfuzzer.find(filename) == generated_libfuzzer.end()) {
      generated_libfuzzer.insert(filename);
      return false;
    } else {
      return true;
    }
  else
    if (generated_pathfinder.find(filename) == generated_pathfinder.end()) {
      generated_pathfinder.insert(filename);
      return false;
    } else {
      return true;
    }
}

void make_dir(std::string dir) {
  if (mkdir(dir.c_str(), 0775) != 0) {
    std::cout << "Failed to create directory " << dir << std::endl;
    exit(0);
  }
}

void make_dir_and_write_files(std::string dir, std::vector<std::pair<std::string, std::string>> files, bool is_libfuzzer) {
  if (mkdir(dir.c_str(), 0775) != 0) {
    std::cout << "Failed to create directory " << dir << std::endl;
    exit(0);
  }

  //std::string cmake_contents = "include(../../pathfinder.cmake)\n\n";
  std::string cmake_contents;
  for (auto p: files) {
    std::string target_name = p.first;
    std::string code = p.second;

    size_t id = 0;
    std::string filename = target_name + "_" + std::to_string(id) + ".cpp";
    while (file_exist(filename, is_libfuzzer))
      filename = target_name + "_" + std::to_string(++id) + ".cpp";

    std::ofstream writeFile(dir + "/" + filename);
    if(writeFile.is_open()){
      writeFile << code;
      writeFile.close();
      if (is_libfuzzer)
        cmake_contents += "add_libfuzzer_fuzz_target(" + strip_ext(filename) + ")\n";
      else
        cmake_contents += "add_pathfinder_fuzz_target(" + strip_ext(filename) + ")\n";
    } else {
      assert(false);
    }
  }

  std::ofstream writeFile(dir + "/CMakeLists.txt");
  if(writeFile.is_open()){
    writeFile << cmake_contents;
    writeFile.close();
  } else {
    assert(false);
  }
}

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  make_dir("pathfinder");
  make_dir("pathfinder/generated");
  make_dir_and_write_files("pathfinder/generated/functional_common", functional_common_files_pathfinder, false);
  make_dir_and_write_files("pathfinder/generated/functional_additional", functional_additional_files_pathfinder, false);
  make_dir_and_write_files("pathfinder/generated/at", at_files_pathfinder, false);
  make_dir_and_write_files("pathfinder/generated/module", module_files_pathfinder, false);

  make_dir("libfuzzer");
  make_dir("libfuzzer/generated");
  make_dir_and_write_files("libfuzzer/generated/functional_common", functional_common_files_libfuzzer, true);

  std::string cmake_pathfinder_contents;
  cmake_pathfinder_contents += "add_subdirectory(functional_common)\n";
  cmake_pathfinder_contents += "add_subdirectory(functional_additional)\n";
  cmake_pathfinder_contents += "add_subdirectory(at)\n";
  cmake_pathfinder_contents += "add_subdirectory(module)\n";

  std::string cmake_libfuzzer_contents;
  cmake_libfuzzer_contents += "add_subdirectory(functional_common)\n";

  std::ofstream cmake_pathfinder("pathfinder/generated/CMakeLists.txt");
  std::ofstream cmake_libfuzzer("libfuzzer/generated/CMakeLists.txt");
  if(cmake_pathfinder.is_open() || cmake_libfuzzer.is_open()){
    cmake_pathfinder << cmake_pathfinder_contents;
    cmake_libfuzzer << cmake_libfuzzer_contents;
    cmake_pathfinder.close();
    cmake_libfuzzer.close();
  } else {
    assert(false);
  }

  return 0;
}
