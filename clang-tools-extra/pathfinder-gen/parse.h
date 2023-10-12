#ifndef PATHFINDER_GEN_PARSE
#define PATHFINDER_GEN_PARSE

#include "param.h"

extern bool option_class_done;

std::unique_ptr<Param> parseTorchParam(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseBuiltin(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseDtype(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseEnum(clang::QualType t, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseVector(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseTensor(clang::QualType t, std::string name);
Optional<std::unique_ptr<Param>> parseIntArrayRef(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseExpandingArray(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseExpandingArrayWithOptionalElem(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseOptional(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseVariant(clang::QualType t, std::string name, ASTContext &Ctx);
Optional<std::unique_ptr<Param>> parseMAP(clang::QualType t, std::string name, ASTContext &Ctx);

size_t num_input_tensor(std::string& module_name);

#endif
