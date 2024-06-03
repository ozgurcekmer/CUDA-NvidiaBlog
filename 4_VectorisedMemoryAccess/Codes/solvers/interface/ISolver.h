#pragma once

#include "../../Parameters.h"

#include <vector>

template <typename T>
class ISolver
{
protected:
	const std::vector<T>& A;
	std::vector<T>& B;

public:
	ISolver(const std::vector<T>& A,
		std::vector<T>& B) : A{A}, B{B} {}
	virtual ~ISolver() {}
	virtual void solver() = 0;
};