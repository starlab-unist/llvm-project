#ifndef PATHFINDER_GEN_EXTRACT
#define PATHFINDER_GEN_EXTRACT

#include "param.h"

extern bool option_class_done;

std::unique_ptr<TorchParam> extractTorchParam(clang::QualType t, std::string name, ASTContext &Ctx);

#endif
