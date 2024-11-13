// Copyright(c) 2024, Pete Brubaker <pete.brubaker@intel.com>
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ISPC: Making CPU SIMD fun while tracing rays!
// Part 1
//
// Graphics Programming Conference 2024
// https://www.graphicsprogrammingconference.nl/
//
// Simple Iteration and Reductions
//

#pragma once

#include <vector>
#include <algorithm>
#include <float.h>

using std::vector;

inline void AddArrayElements(vector<float>& output, const vector<float>& a, const vector<float>& b, const size_t count)
{
	#pragma loop(no_vector)
	for (size_t i = 0; i < count; ++i)
	{
		output[i] = a[i] + b[i];
	}
}

inline void SumArray(float& sum_output, const vector<float>& a, const size_t count)
{
	float sum = 0.f;

	#pragma loop(no_vector)
	for (size_t i = 0; i < count; ++i)
	{
		sum += a[i];
	}

	sum_output = sum;
}

inline void MinArray(float& min_output, const vector<float>& a, const size_t count)
{
	float min = FLT_MAX;

	#pragma loop(no_vector)
	for (size_t i = 0; i < count; ++i)
	{
		if (a[i] < min)
		{
			min = a[i];
		}
	}

	min_output = min;
}

inline void MaxArray(float& max_output, const vector<float>& a, const size_t count)
{
	float max = -FLT_MAX;

	#pragma loop(no_vector)
	for (size_t i = 0; i < count; ++i)
	{
		if (a[i] > max)
		{
			max = a[i];
		}
	}

	max_output = max;
}

inline void AverageArray(float& avg_output, const vector<float>& a, const size_t count)
{
	float sum = 0.f;

	#pragma loop(no_vector)
	for (size_t i = 0; i < count; ++i)
	{
		sum += a[i];
	}

	avg_output = sum / static_cast<float>(count);
}