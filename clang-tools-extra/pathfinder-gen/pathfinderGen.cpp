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

std::map<std::string, std::set<tensor_rank>> functional_common = {
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

std::map<std::string, std::set<tensor_rank>> functional_additional = {
  {"torch::nn::functional::_smooth_l1_loss",{{}}},
  {"torch::nn::functional::_unpool_output_size",{{}}},
  {"torch::nn::functional::adaptive_max_pool2d_with_indices",{{4}}},
  {"torch::nn::functional::fractional_max_pool2d",{{4}}},
  {"torch::nn::functional::fractional_max_pool2d_with_indices",{{4}}},
  {"torch::nn::functional::fractional_max_pool3d",{{5}}},
  {"torch::nn::functional::fractional_max_pool3d_with_indices",{{5}}},
  {"torch::nn::functional::group_norm",{{2},{3},{4},{5}}},
  {"torch::nn::functional::huber_loss",{same_ranks}},
  {"torch::nn::functional::max_pool1d_with_indices",{{3}}},
  {"torch::nn::functional::max_pool2d_with_indices",{{4}}},
  {"torch::nn::functional::max_pool3d_with_indices",{{5}}},
  {"torch::nn::functional::mish",{{}}},
  {"torch::nn::functional::multi_head_attention_forward",{{}}},
};

std::map<std::string, std::set<tensor_rank>> at = {
  {"at::abs",{{}}},
  {"at::abs_",{{}}},
  {"at::absolute",{{}}},
  {"at::acos",{{}}},
  {"at::acos_",{{}}},
  {"at::abs",{{}}},
  {"at::absolute",{{}}},
  {"at::acos",{{}}},
  {"at::acosh",{{}}},
  {"at::addbmm",{{}}},
  {"at::addcdiv",{{}}},
  {"at::addcmul",{{}}},
  {"at::addmm",{{}}},
  {"at::addmv",{{}}},
  {"at::addr",{{}}},
  {"at::adjoint",{{}}},
  {"at::affine_grid_generator",{{}}},
  {"at::alias",{{}}},
  {"at::align_tensors",{{}}},
  {"at::all",{{}}},
  {"at::allclose",{{}}},
  {"at::alpha_dropout",{{}}},
  {"at::amax",{{}}},
  {"at::amin",{{}}},
  {"at::aminmax",{{}}},
  {"at::angle",{{}}},
  {"at::any",{{}}},
  {"at::arccos",{{}}},
  {"at::arccosh",{{}}},
  {"at::arcsin",{{}}},
  {"at::arcsinh",{{}}},
  {"at::arctan",{{}}},
  {"at::arctan2",{{}}},
  {"at::arctanh",{{}}},
  {"at::argmax",{{}}},
  {"at::argmin",{{}}},
  {"at::argsort",{{}}},
  {"at::argsort",{{}}},
  {"at::argwhere",{{}}},
  {"at::as_strided",{{}}},
  {"at::as_strided_scatter",{{}}},
  {"at::asin",{{}}},
  {"at::asinh",{{}}},
  {"at::atan",{{}}},
  {"at::atan2",{{}}},
  {"at::atanh",{{}}},
  {"at::atleast_1d",{{}}},
  {"at::atleast_2d",{{}}},
  {"at::atleast_3d",{{}}},
};

std::map<std::string, std::set<tensor_rank>> module = {
  // Convolution Layers
  {"torch::nn::Conv1d",{{2},{3}}},
  {"torch::nn::Conv2d",{{3},{4}}},
  {"torch::nn::Conv3d",{{4},{5}}},
  {"torch::nn::ConvTranspose1d",{{2},{3}}},
  {"torch::nn::ConvTranspose2d",{{3},{4}}},
  {"torch::nn::ConvTranspose3d",{{4},{5}}},
  {"torch::nn::Unfold",{{2},{3},{4}}},
  {"torch::nn::Fold",{{2},{3}}},

  // Pooling Layers
  {"torch::nn::MaxPool1d",{{2},{3}}},
  {"torch::nn::MaxPool2d",{{3},{4}}},
  {"torch::nn::MaxPool3d",{{4},{5}}},
  {"torch::nn::MaxUnpool1d",{{2,2},{3,3}}},
  {"torch::nn::MaxUnpool2d",{{3,3},{4,4}}},
  {"torch::nn::MaxUnpool3d",{{4,4},{5,5}}},
  {"torch::nn::AvgPool1d",{{2},{3}}},
  {"torch::nn::AvgPool2d",{{3},{4}}},
  {"torch::nn::AvgPool3d",{{4},{5}}},
  {"torch::nn::FractionalMaxPool2d",{{3},{4}}},
  {"torch::nn::FractionalMaxPool3d",{{4},{5}}},
  {"torch::nn::LPPool1d",{{2},{3}}},
  {"torch::nn::LPPool2d",{{4}}},
  {"torch::nn::AdaptiveMaxPool1d",{{2},{3}}},
  {"torch::nn::AdaptiveMaxPool2d",{{3},{4}}},
  {"torch::nn::AdaptiveMaxPool3d",{{4},{5}}},
  {"torch::nn::AdaptiveAvgPool1d",{{2},{3}}},
  {"torch::nn::AdaptiveAvgPool2d",{{3},{4}}},
  {"torch::nn::AdaptiveAvgPool3d",{{4},{5}}},

  // Padding Layers
  {"torch::nn::ReflectionPad1d",{{2},{3}}},
  {"torch::nn::ReflectionPad2d",{{3},{4}}},
  {"torch::nn::ReflectionPad3d",{{4},{5}}},
  {"torch::nn::ReplicationPad1d",{{2},{3}}},
  {"torch::nn::ReplicationPad2d",{{3},{4}}},
  {"torch::nn::ReplicationPad3d",{{4},{5}}},
  {"torch::nn::ZeroPad1d",{{2},{3}}},
  {"torch::nn::ZeroPad2d",{{3},{4}}},
  {"torch::nn::ZeroPad3d",{{4},{5}}},
  {"torch::nn::ConstantPad1d",{{2},{3}}},
  {"torch::nn::ConstantPad2d",{{3},{4}}},
  {"torch::nn::ConstantPad3d",{{4},{5}}},

  // Non-linear Activations (weighted sum, nonlinearity)
  {"torch::nn::ELU",{{}}},
  {"torch::nn::Hardshrink",{{}}},
  {"torch::nn::Hardtanh",{{}}},
  {"torch::nn::LeakyReLU",{{}}},
  {"torch::nn::LogSigmoid",{{}}},
  //{"torch::nn::MultiheadAttention",{{}}},
  {"torch::nn::PReLU",{{}}},
  {"torch::nn::ReLU",{{}}},
  {"torch::nn::ReLU6",{{}}},
  {"torch::nn::RReLU",{{}}},
  {"torch::nn::SELU",{{}}},
  {"torch::nn::CELU",{{}}},
  {"torch::nn::Sigmoid",{{}}},
  {"torch::nn::SiLU",{{}}},
  {"torch::nn::Mish",{{}}},
  {"torch::nn::Softplus",{{}}},
  {"torch::nn::Softshrink",{{}}},
  {"torch::nn::Softsign",{{}}},
  {"torch::nn::Tanh",{{}}},
  {"torch::nn::Tanhshrink",{{}}},
  {"torch::nn::Threshold",{{}}},
  {"torch::nn::GLU",{{}}},

  // Non-linear Activations (other)
  {"torch::nn::Softmin",{{}}},
  {"torch::nn::Softmax",{{}}},
  {"torch::nn::Softmax2d",{{3},{4}}},
  {"torch::nn::LogSoftmax",{{}}},
  {"torch::nn::AdaptiveLogSoftmaxWithLoss",{{1},{2}}},

  // Normalization Layers
  {"torch::nn::BatchNorm1d",{{2},{3}}},
  {"torch::nn::BatchNorm2d",{{4}}},
  {"torch::nn::BatchNorm3d",{{5}}},
  {"torch::nn::GroupNorm",{{2},{3},{4},{5}}},
  {"torch::nn::InstanceNorm1d",{{2},{3}}},
  {"torch::nn::InstanceNorm2d",{{3},{4}}},
  {"torch::nn::InstanceNorm3d",{{4},{5}}},
  {"torch::nn::LayerNorm",{{}}},
  {"torch::nn::LocalResponseNorm",{{2},{3},{4},{5}}},

  // Recurrent Layers
  {"torch::nn::RNN",{{2,2},{3,3}}},
  {"torch::nn::LSTM",{{2,2,2},{3,3,3}}},
  {"torch::nn::GRU",{{2,2},{3,3}}},
  {"torch::nn::RNNCell",{{1,1},{2,2}}},
  {"torch::nn::LSTMCell",{{1,1,1},{2,2,2}}},
  {"torch::nn::GRUCell",{{1,1},{2,2}}},

  // Transformer Layers
  //{"torch::nn::Transformer",{{}}},
  //{"torch::nn::TransformerEncoder",{{}}},
  //{"torch::nn::TransformerDecoder",{{}}},
  //{"torch::nn::TransformerEncoderLayer",{{}}},
  //{"torch::nn::TransformerDecoderLayer",{{}}},

  // Linear Layers
  {"torch::nn::Identity",{{}}},
  {"torch::nn::Linear",{{}}},
  {"torch::nn::Bilinear",{same_ranks}},

  // Dropout Layers
  {"torch::nn::Dropout",{same_ranks}},
  {"torch::nn::Dropout2d",{{3},{4}}},
  {"torch::nn::Dropout3d",{{4},{5}}},
  {"torch::nn::AlphaDropout",{{}}},
  {"torch::nn::FeatureAlphaDropout",{{4},{5}}},

  // Sparse Layers
  {"torch::nn::Embedding",{{}}},
  {"torch::nn::EmbeddingBag",{{}}},

  // Distance Functions
  {"torch::nn::CosineSimilarity",{same_ranks}},
  {"torch::nn::PairwiseDistance",{{1,1},{2,2}}},

  // Loss Functions
  {"torch::nn::L1Loss",{same_ranks}},
  {"torch::nn::MSELoss",{same_ranks}},
  {"torch::nn::CrossEntropyLoss",{{1,0},{2,1},{3,2},{4,3},{5,4}}},
  {"torch::nn::CTCLoss",{{3,2,1,1},{2,1,0,0}}},
  {"torch::nn::NLLLoss",{{1,0},{2,1},{3,2},{4,3},{5,4}}},
  {"torch::nn::PoissonNLLLoss",{same_ranks}},
  {"torch::nn::KLDivLoss",{same_ranks}},
  {"torch::nn::BCELoss",{same_ranks}},
  {"torch::nn::BCEWithLogitsLoss",{same_ranks}},
  {"torch::nn::MarginRankingLoss",{{1,1,1},{2,2,2},{3,3,3},{4,4,4},{5,5,5}}},
  {"torch::nn::HingeEmbeddingLoss",{same_ranks}},
  {"torch::nn::MultiLabelMarginLoss",{{1,1},{2,2}}},
  {"torch::nn::HuberLoss",{same_ranks}},
  {"torch::nn::SmoothL1Loss",{same_ranks}},
  {"torch::nn::SoftMarginLoss",{same_ranks}},
  {"torch::nn::MultiLabelSoftMarginLoss",{{2,2}}},
  {"torch::nn::CosineEmbeddingLoss",{{2,2,1},{1,1,0}}},
  {"torch::nn::MultiMarginLoss",{{2,1},{1,0}}},
  {"torch::nn::TripletMarginLoss",{{1,1,1},{2,2,2}}},
  {"torch::nn::TripletMarginWithDistanceLoss",{{1,1,1},{2,2,2},{3,3,3},{4,4,4},{5,5,5}}},
  
  // Vision Layers
  {"torch::nn::PixelShuffle",{{3},{4},{5}}},
  {"torch::nn::PixelUnshuffle",{{3},{4},{5}}},
  {"torch::nn::Upsample",{{3},{4},{5}}},
  
  // Utilities
  {"torch::nn::Flatten",{{}}},
  {"torch::nn::Unflatten",{{}}},
  
  // etc
  {"torch::nn::CrossMapLRN2d",{{}}},
};

std::string current_target;
tensor_rank current_tensor_rank;
size_t current_tensor_rank_idx;
bool option_class_done;
bool general_tensor_rank = false;

Optional<size_t> get_rank() {
  if (general_tensor_rank)
    return None;

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

std::vector<std::pair<std::string, std::string>> functional_common_files;
std::vector<std::pair<std::string, std::string>> functional_additional_files;
std::vector<std::pair<std::string, std::string>> at_files;
std::vector<std::pair<std::string, std::string>> module_files;

void map_concat(std::map<std::string, std::set<tensor_rank>>& orig,std::map<std::string, std::set<tensor_rank>>& other) {
  orig.insert(other.begin(), other.end());
}

bool in(std::string api_name, std::map<std::string, std::set<tensor_rank>> apis) {
  return apis.find(api_name) != apis.end();
}

class FindNamedClassVisitor
  : public RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  explicit FindNamedClassVisitor(ASTContext *Context)
    : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    std::map<std::string, std::set<tensor_rank>> target_api;
    map_concat(target_api, functional_common);
    map_concat(target_api, functional_additional);
    map_concat(target_api, at);

    auto api_it = target_api.find(Declaration->getQualifiedNameAsString());
    if (api_it != target_api.end()) {
      current_target = Declaration->getQualifiedNameAsString();
      std::cout << current_target << std::endl;
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

        if (params.empty())
          continue;

        bool is_module = false;
        std::string code = gen_code(current_target, params, is_module, 0);
        if (in (current_target, functional_common))
          functional_common_files.push_back(std::make_pair(current_target_unqualified, code));
        else if (in (current_target, functional_additional))
          functional_additional_files.push_back(std::make_pair(current_target_unqualified, code));
        else if (in (current_target, at))
          at_files.push_back(std::make_pair(current_target_unqualified, code));
        api_id++;
        if (general_tensor_rank)
          break;
      }
    }

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

    current_tensor_rank_idx = 0;
    option_class_done = false;
    std::vector<std::unique_ptr<Param>> params;
    for (size_t i = 0; i < num_input_tensor; i++)
      params.push_back(std::make_unique<Param>(TENSOR, get_rank()));
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

    size_t num_input_tensor;
    if (endswith(current_target, "Loss")) {
      if (current_target == "torch::nn::CTCLoss") {
        num_input_tensor = 4;
      } else if (
          current_target == "torch::nn::TripletMarginLoss" ||
          current_target == "torch::nn::TripletMarginWithDistanceLoss" ||
          current_target == "torch::nn::CosineEmbeddingLoss" ||
          current_target == "torch::nn::MarginRankingLoss") {
        num_input_tensor = 3;
      } else {
        num_input_tensor = 2;
      }
    } else {
      if (current_target == "torch::nn::LSTM" ||
          current_target == "torch::nn::LSTMCell") {
        num_input_tensor = 3;
      } else if (
          current_target == "torch::nn::MaxUnpool1d" ||
          current_target == "torch::nn::MaxUnpool2d" ||
          current_target == "torch::nn::MaxUnpool3d" ||
          current_target == "torch::nn::CosineSimilarity" ||
          current_target == "torch::nn::PairwiseDistance" ||
          current_target == "torch::nn::GRU" ||
          current_target == "torch::nn::GRUCell" ||
          current_target == "torch::nn::RNN" ||
          current_target == "torch::nn::RNNCell" ||
          current_target == "torch::nn::Bilinear") {
        num_input_tensor = 2;
      } else {
        num_input_tensor = 1;
      }
    }

    size_t target_id = 0;
    for (auto rank: api_it->second) {
      current_tensor_rank = rank;
      std::vector<std::vector<std::unique_ptr<Param>>> candidates;

      for (auto ctor: class_decl->ctors())
        if (auto parsed = parseModuleCtor(ctor, num_input_tensor))
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
              current_tensor_rank_idx = 0;
              option_class_done = false;
              std::vector<std::unique_ptr<Param>> params;
              for (size_t i = 0; i < num_input_tensor; i++)
                params.push_back(std::make_unique<Param>(TENSOR, get_rank()));
              if (auto p = parseTorchParam(targs[2].getAsType(), *Context))
                params.push_back(std::move(p));
              candidates.push_back(std::move(params));
            } else {
              std::cout << "ctsdecl:\n";
              ctsdecl->dump();
              for (auto ctor: ctsdecl->ctors())
                if (auto parsed = parseModuleCtor(ctor, num_input_tensor))
                  candidates.push_back(std::move(parsed.getValue()));
            }
          }
        }
      }

      auto params = pickBest(std::move(candidates));

      std::string filename =
        current_target_unqualified + "_" + std::to_string(target_id++) +
        ".cpp";
      bool is_module = true;
      std::string code = gen_code(current_target, params, is_module, num_input_tensor);
      module_files.push_back(std::make_pair(filename, code));
      if (general_tensor_rank)
        break;
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

bool is_file_exist(std::string filename) {
  std::ifstream infile(filename.c_str());
  return infile.good();
}

void make_dir_and_write_files(std::string dir, std::vector<std::pair<std::string, std::string>> files) {
  if (mkdir(dir.c_str(), 0775) != 0) {
    std::cout << "Failed to create directory " << dir << std::endl;
    exit(0);
  }

  std::string cmake_contents = "include(../../pathfinder.cmake)\n\n";
  for (auto p: files) {
    std::string target_name = p.first;
    std::string code = p.second;

    size_t id = 0;
    std::string filename = target_name + "_" + std::to_string(id) + ".cpp";
    while (is_file_exist(filename))
      filename = target_name + std::to_string(++id) + ".cpp";

    std::ofstream writeFile(dir + "/" + filename);
    if(writeFile.is_open()){
      writeFile << code;
      writeFile.close();
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

  general_tensor_rank = true;
  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  make_dir_and_write_files("functional_common", functional_common_files);
  make_dir_and_write_files("functional_additional", functional_additional_files);
  make_dir_and_write_files("at", at_files);
  make_dir_and_write_files("module", module_files);

  std::ofstream writeFile("CMakeLists.txt");
  if(writeFile.is_open()){
    writeFile << "add_subdirectory(functional_common)\n";
    writeFile << "add_subdirectory(functional_additional)\n";
    writeFile << "add_subdirectory(at)\n";
    writeFile << "add_subdirectory(module)\n\n";
    writeFile.close();
  } else {
    assert(false);
  }

  return 0;
}
