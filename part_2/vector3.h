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

#pragma once

#include <vector>
#include <immintrin.h>

using std::vector;

namespace Types
{
	// regular unaligned Vector3
	struct Vector3
	{
		float x, y, z;
	};

	// aligned vector
	struct alignas(16) Vector3_SSE
	{
		__m128 m_vec;
	};

	// SSE functions
//	void DotProduct_HADD(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N);
//	void DotProduct_DPPS(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N);
//	void DotProduct_SHUFFLE(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N);

	#define VCI_IRR 0 // Component index to be used
	enum VectorComponentIndex
	{
		VCI_X,
		VCI_Y,
		VCI_Z,
		VCI_W
	};

	#define D_SHUF_PASS(__v__, CI_1) _mm_shuffle_ps(__v__, __v__, _MM_SHUFFLE(VCI_IRR, VCI_IRR, VCI_IRR, CI_1))

	inline __m128 sse_dp3_HADD(__m128 vecA, __m128 vecB)
	{
		__m128 aMulB = _mm_mul_ps(vecA, vecB);
		aMulB = _mm_insert_ps(aMulB, aMulB, _MM_MK_INSERTPS_NDX(0, 0, 8));
		__m128 pair = _mm_hadd_ps(aMulB, aMulB);
		return _mm_hadd_ps(pair, pair);
	}

	inline __m128 sse_dp3_DPPS(__m128 vecA, __m128 vecB)
	{
		return _mm_dp_ps(vecA, vecB, 0x71);
	}

	inline __m128 sse_dp3_SHUFFLE(__m128 vecA, __m128 vecB)
	{
		__m128 aMulB = _mm_mul_ps(vecA, vecB);
		return _mm_add_ss(aMulB, _mm_add_ss(D_SHUF_PASS(aMulB, VCI_Y), D_SHUF_PASS(aMulB, VCI_Z)));
	}

inline void DotProduct_HADD(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		Vector3_SSE tmp;
		tmp.m_vec = sse_dp3_HADD(src[i].m_vec, src[i].m_vec);
		dst[i] = _mm_cvtss_f32(tmp.m_vec);
	}
}


inline void DotProduct_DPPS(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		Vector3_SSE tmp;
		tmp.m_vec = sse_dp3_DPPS(src[i].m_vec, src[i].m_vec);
		dst[i] = _mm_cvtss_f32(tmp.m_vec);
	}
}

inline void DotProduct_SHUFFLE(vector<float>& dst, const Vector3_SSE* __restrict src, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		Vector3_SSE tmp;
		tmp.m_vec = sse_dp3_SHUFFLE(src[i].m_vec, src[i].m_vec);
		dst[i] = _mm_cvtss_f32(tmp.m_vec);
	}
}
}
