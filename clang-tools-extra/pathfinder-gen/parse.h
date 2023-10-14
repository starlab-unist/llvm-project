#ifndef PATHFINDER_GEN_PARSE
#define PATHFINDER_GEN_PARSE

#include "param.h"

extern bool option_class_done;

std::unique_ptr<TorchParam> parseTorchParam(clang::QualType t, std::string name, ASTContext &Ctx);

// Base case
std::unique_ptr<TorchParam> parseBuiltin(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseDtype(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseEnum(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseVector(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseTensor(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseExpandingArray(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseExpandingArrayWithOptionalElem(clang::QualType t, std::string name, ASTContext &Ctx);

// Recursive case
std::unique_ptr<TorchParam> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx);
std::unique_ptr<TorchParam> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx);

#endif
