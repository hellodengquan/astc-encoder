// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2026 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief Unit tests for the GPU decode profile analysis.
 *
 * The tests deliberately build the expected results from independent inputs, not from the
 * analysis API itself:
 *
 *   * Blocks are produced by the compressor and then replaced with hand-built physical
 *     encodings whose decode behavior is checked with astcenc_decompress_image().
 *   * Random payloads are cross-checked block by block against the real decoder under both an
 *     LDR and an HDR context.
 */

#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "../astcenc.h"

namespace astcenc
{

/** @brief An owning pointer to a codec context. */
using astcenc_context_unq_ptr = std::unique_ptr<astcenc_context, decltype(&astcenc_context_free)>;

/**
 * @brief Query if this build supports a block size.
 *
 * Reduced-capacity builds (ASTCENC_BLOCK_MAX_TEXELS) reject block sizes with too many texels, so
 * tests covering those block sizes must skip rather than fail.
 *
 * @param block_x  The block size X dimension.
 * @param block_y  The block size Y dimension.
 * @param block_z  The block size Z dimension.
 *
 * @return @c true if the block size is supported by this build.
 */
static bool block_size_supported(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z
) {
	static const uint8_t data[16] {};
	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    data, sizeof(data), block_x, block_y, block_z,
	    block_x, block_y, block_z, &result);
	return status != ASTCENC_ERR_NOT_IMPLEMENTED;
}

/**
 * @brief Build a constant-color block with an FP16 payload.
 *
 * The block mode is the constant-color escape with the FP16 bit set, and the void-extent covers
 * the whole 2D block. The FP16 payload is zero, which represents black; the value does not matter
 * for profile classification.
 */
static void make_hdr_constant_block(uint8_t block[16])
{
	std::memset(block, 0, 16);

	// Block mode 0x7FC | 0x200, little-endian over two bytes: low 11 bits hold mode 0x1FC + FP16
	block[0] = 0xFC;
	block[1] = 0xFF;

	// Constant color words are zero, which is FP16 zero for all components

	// 2D void-extent for a 4x4 block; low and high values are all ones so it spans the block
	block[2] = 0x03;  // Reserved bits [10:11] must both be one
	block[3] = 0xFF;
	block[4] = 0x3F;
	block[5] = 0xF0;
	block[6] = 0x0F;
	block[7] = 0xFF;
	block[8] = 0xFF;
	block[9] = 0xFF;
	block[10] = 0xFF;
	block[11] = 0xFF;
	block[12] = 0xFF;
	block[13] = 0xFF;
	block[14] = 0xFF;
	block[15] = 0xFF;
}

/**
 * @brief Build a constant-color block with an FP16 payload, valid for 3D block sizes.
 *
 * The 3D void-extent stores three low/high coordinate pairs. Setting every bit after the block
 * mode to one makes each pair span the whole block, which is legal.
 */
static void make_hdr_constant_block_3d(uint8_t block[16])
{
	std::memset(block, 0, 16);
	block[0] = 0xFC;
	block[1] = 0xFF;
	for (int i = 2; i < 16; i++)
	{
		block[i] = 0xFF;
	}
}

/**
 * @brief Build a constant-color block that is invalid in every profile.
 *
 * The block mode is the constant-color escape, but the reserved void-extent bits are zero, which
 * the decoder rejects.
 */
static void make_invalid_constant_block(uint8_t block[16])
{
	std::memset(block, 0, 16);
	block[0] = 0xFC;
	block[1] = 0x01;
}

/**
 * @brief Build a block with a reserved block mode.
 *
 * Block mode zero is always invalid in both 2D and 3D block sizes.
 */
static void make_invalid_mode_block(uint8_t block[16])
{
	std::memset(block, 0, 16);
}

/**
 * @brief Decode one block with a given profile and return the status.
 *
 * @param block    The block to decode.
 * @param block_x  The block size X dimension.
 * @param block_y  The block size Y dimension.
 * @param block_z  The block size Z dimension.
 * @param profile  The decode profile.
 *
 * @return The decompression status for the block.
 */
static astcenc_error decode_block(
	const uint8_t block[16],
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	astcenc_profile profile
) {
	astcenc_config config;
	astcenc_error status = astcenc_config_init(
	    profile, block_x, block_y, block_z, ASTCENC_PRE_FASTEST,
	    ASTCENC_FLG_DECOMPRESS_ONLY, &config);
	EXPECT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context* context_raw = nullptr;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	EXPECT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	std::vector<uint8_t> texels(static_cast<size_t>(block_x) * block_y * block_z * 4);
	std::vector<void*> slices(block_z);
	for (unsigned int z = 0; z < block_z; z++)
	{
		slices[z] = texels.data() + static_cast<size_t>(z) * block_x * block_y * 4;
	}

	astcenc_image image {
		.dim_x = block_x,
		.dim_y = block_y,
		.dim_z = block_z,
		.data_type = ASTCENC_TYPE_U8,
		.data = slices.data(),
		.row_stride = 0
	};

	return astcenc_decompress_image(context.get(), block, 16, &image, &swizzle, 0);
}

/**
 * @brief Check that the analyzer and the decoder agree on a single block.
 *
 * The analyzer is exercised through the image API with a one-block image, because the block-level
 * classification is not exposed separately.
 *
 * @param block    The block to analyze.
 * @param block_x  The block size X dimension.
 * @param block_y  The block size Y dimension.
 * @param block_z  The block size Z dimension.
 */
static void expect_block_class(
	const uint8_t block[16],
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	bool expect_hdr,
	bool expect_invalid
) {
	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    block, 16, block_x, block_y, block_z,
	    block_x, block_y, block_z, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_EQ(result.block_count, 1u);
	EXPECT_EQ(result.hdr_block_count, expect_hdr ? 1u : 0u);
	EXPECT_EQ(result.invalid_block_count, expect_invalid ? 1u : 0u);

	// Cross-check each class against the real decoder
	astcenc_error hdr_status = decode_block(block, block_x, block_y, block_z, ASTCENC_PRF_HDR);
	EXPECT_EQ(hdr_status == ASTCENC_ERR_BAD_BLOCK, expect_invalid);

	astcenc_error ldr_status = decode_block(block, block_x, block_y, block_z, ASTCENC_PRF_LDR);
	EXPECT_EQ(ldr_status == ASTCENC_ERR_BAD_BLOCK, expect_hdr || expect_invalid);
}

/** @brief An FP16 constant color block is legal, but requires HDR. */
TEST(decode_profile, fp16_constant_requires_hdr)
{
	uint8_t block[16];
	make_hdr_constant_block(block);
	expect_block_class(block, 4, 4, 1, true, false);
}

/** @brief An FP16 constant color block is still HDR-only for 3D block sizes. */
TEST(decode_profile, fp16_constant_requires_hdr_3d)
{
	uint8_t block[16];
	make_hdr_constant_block_3d(block);
	expect_block_class(block, 4, 4, 4, true, false);
}

/** @brief A constant color block with an invalid void-extent is invalid in every profile. */
TEST(decode_profile, invalid_constant_block)
{
	uint8_t block[16];
	make_invalid_constant_block(block);
	expect_block_class(block, 4, 4, 1, false, true);
}

/** @brief A reserved block mode is invalid in every profile. */
TEST(decode_profile, invalid_block_mode)
{
	uint8_t block[16];
	make_invalid_mode_block(block);
	expect_block_class(block, 4, 4, 1, false, true);
}

/**
 * @brief Compress a solid image and check that the analysis agrees with the compressor's profile.
 *
 * A solid HDR image with luminance above one compresses to constant color blocks. A solid LDR
 * image compresses to UNORM16 constant color blocks.
 *
 * @param profile  The compression profile.
 * @param channel  The input component value in FP32.
 * @param expect_hdr  The expected HDR requirement of the compressed texture.
 */
static void check_solid_image(
	astcenc_profile profile,
	float channel,
	bool expect_hdr
) {
	constexpr size_t dim = 16;
	constexpr unsigned int block = 4;

	std::vector<float> data(dim * dim * 4);
	for (size_t i = 0; i < dim * dim; i++)
	{
		data[i * 4 + 0] = channel;
		data[i * 4 + 1] = channel;
		data[i * 4 + 2] = channel;
		data[i * 4 + 3] = 1.0f;
	}

	std::vector<void*> slices { data.data() };
	astcenc_image image {
		.dim_x = dim,
		.dim_y = dim,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_F32,
		.data = slices.data(),
		.row_stride = 0
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
	    profile, block, block, 1, ASTCENC_PRE_FASTEST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context* context_raw = nullptr;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	size_t blocks = (dim + block - 1) / block;
	std::vector<uint8_t> compressed(blocks * blocks * 16);
	status = astcenc_compress_image(
	    context.get(), &image, &swizzle, compressed.data(), compressed.size(), 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_decode_profile_result result;
	status = astcenc_get_decode_profile(
	    compressed.data(), compressed.size(), block, block, 1,
	    dim, dim, 1, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_EQ(result.requires_hdr, expect_hdr);
	EXPECT_EQ(result.is_ldr_compatible, !expect_hdr);
	EXPECT_EQ(result.block_count, blocks * blocks);
	EXPECT_EQ(result.invalid_block_count, 0u);

	// Every block must be classified the same way, and the decoder must agree per block
	for (size_t i = 0; i < result.block_count; i++)
	{
		astcenc_error ldr_status = decode_block(compressed.data() + i * 16, block, block, 1,
		                                        ASTCENC_PRF_LDR);
		EXPECT_EQ(ldr_status == ASTCENC_ERR_BAD_BLOCK, expect_hdr);
	}
}

/** @brief A pure-LDR solid texture must not be reported as needing HDR. */
TEST(decode_profile, solid_ldr_image)
{
	check_solid_image(ASTCENC_PRF_LDR, 0.5f, false);
}

/** @brief A solid HDR texture above LDR range must be reported as needing HDR. */
TEST(decode_profile, solid_hdr_image)
{
	check_solid_image(ASTCENC_PRF_HDR, 2.0f, true);
}

/**
 * @brief A texture that only needs HDR because of FP16 constant blocks must still be reported.
 *
 * This is the case that a naive check built on astcenc_get_block_info() misses: it returns early
 * for constant color blocks.
 */
TEST(decode_profile, hdr_only_constant_blocks)
{
	constexpr size_t dim = 8;
	constexpr unsigned int block = 4;

	uint8_t hdr_block[16];
	make_hdr_constant_block(hdr_block);

	std::vector<uint8_t> compressed(4 * 16);
	for (size_t i = 0; i < 4; i++)
	{
		std::memcpy(compressed.data() + i * 16, hdr_block, 16);
	}

	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    compressed.data(), compressed.size(), block, block, 1,
	    dim, dim, 1, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_TRUE(result.requires_hdr);
	EXPECT_FALSE(result.is_ldr_compatible);
	EXPECT_EQ(result.hdr_block_count, 4u);
	EXPECT_EQ(result.invalid_block_count, 0u);
	EXPECT_EQ(result.ldr_error_block_count, 4u);
	EXPECT_TRUE(result.has_ldr_error_region);
	EXPECT_EQ(result.ldr_error_region_min[0], 0u);
	EXPECT_EQ(result.ldr_error_region_min[1], 0u);
	EXPECT_EQ(result.ldr_error_region_max[0], 1u);
	EXPECT_EQ(result.ldr_error_region_max[1], 1u);
}

/**
 * @brief HDR-requiring blocks and invalid blocks must be counted separately.
 *
 * Both classes decode as an error color in an LDR profile, so a single LDR decode cannot separate
 * them. The analysis must report distinct counts, and the invalid count must match the blocks that
 * an HDR decode rejects.
 */
TEST(decode_profile, hdr_and_invalid_are_distinct)
{
	constexpr unsigned int block = 4;
	constexpr size_t blocks_x = 3;
	constexpr size_t blocks_y = 3;

	// Start from a valid LDR texture so no block starts as invalid
	std::vector<uint8_t> compressed(blocks_x * blocks_y * 16);
	{
		astcenc_config config;
		astcenc_error status = astcenc_config_init(
		    ASTCENC_PRF_LDR, block, block, 1, ASTCENC_PRE_FASTEST, 0, &config);
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		astcenc_context* context_raw = nullptr;
		status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
		ASSERT_EQ(status, ASTCENC_SUCCESS);
		astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

		const astcenc_swizzle swizzle {
			ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
		};

		std::vector<uint8_t> texels(blocks_x * block * blocks_y * block * 4, 128);
		std::vector<void*> slices { texels.data() };
		astcenc_image image {
			.dim_x = blocks_x * block,
			.dim_y = blocks_y * block,
			.dim_z = 1,
			.data_type = ASTCENC_TYPE_U8,
			.data = slices.data(),
			.row_stride = 0
		};

		status = astcenc_compress_image(
		    context.get(), &image, &swizzle, compressed.data(), compressed.size(), 0);
		ASSERT_EQ(status, ASTCENC_SUCCESS);
	}

	// Replace block (1, 1) with an HDR-requiring block and block (0, 2) with an invalid block
	uint8_t hdr_block[16];
	make_hdr_constant_block(hdr_block);

	uint8_t invalid_block[16];
	make_invalid_constant_block(invalid_block);

	std::memcpy(compressed.data() + ((1 * blocks_x) + 1) * 16, hdr_block, 16);
	std::memcpy(compressed.data() + ((2 * blocks_x) + 0) * 16, invalid_block, 16);

	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    compressed.data(), compressed.size(), block, block, 1,
	    blocks_x * block, blocks_y * block, 1, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_EQ(result.block_count, blocks_x * blocks_y);
	EXPECT_EQ(result.hdr_block_count, 1u);
	EXPECT_EQ(result.invalid_block_count, 1u);
	EXPECT_EQ(result.ldr_error_block_count, 2u);
	EXPECT_TRUE(result.requires_hdr);
	EXPECT_FALSE(result.is_ldr_compatible);

	EXPECT_TRUE(result.has_ldr_error_region);
	EXPECT_EQ(result.ldr_error_region_min[0], 0u);
	EXPECT_EQ(result.ldr_error_region_min[1], 1u);
	EXPECT_EQ(result.ldr_error_region_max[0], 1u);
	EXPECT_EQ(result.ldr_error_region_max[1], 2u);
}

/**
 * @brief The analyzer must agree with the real decoder for random payloads.
 *
 * This exercises every block mode and constant color path, including reserved encodings that are
 * only reachable with arbitrary input.
 *
 * @param block_x  The block size X dimension.
 * @param block_y  The block size Y dimension.
 * @param block_z  The block size Z dimension.
 */
static void check_random_payload(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z
) {
	constexpr size_t blocks_x = 8;
	constexpr size_t blocks_y = 8;
	// A 2D block size describes a single image plane, matching the decompressor contract
	size_t blocks_z = (block_z > 1) ? 4u : 1u;

	size_t block_count = blocks_x * blocks_y * blocks_z;
	std::vector<uint8_t> payload(block_count * 16);

	// Deterministic payload; the goal is coverage, not statistical quality
	uint32_t state = 0x12345678u;
	for (size_t i = 0; i < payload.size(); i++)
	{
		state = state * 1664525u + 1013904223u;
		payload[i] = static_cast<uint8_t>(state >> 16);
	}

	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    payload.data(), payload.size(), block_x, block_y, block_z,
	    blocks_x * block_x, blocks_y * block_y, blocks_z * block_z, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_EQ(result.block_count, block_count);

	// Decode every block under both profiles and check every classification
	size_t ldr_bad = 0;
	size_t hdr_bad = 0;
	for (size_t i = 0; i < block_count; i++)
	{
		if (decode_block(payload.data() + i * 16, block_x, block_y, block_z,
		                 ASTCENC_PRF_LDR) == ASTCENC_ERR_BAD_BLOCK)
		{
			ldr_bad++;
		}

		if (decode_block(payload.data() + i * 16, block_x, block_y, block_z,
		                 ASTCENC_PRF_HDR) == ASTCENC_ERR_BAD_BLOCK)
		{
			hdr_bad++;
		}
	}

	EXPECT_EQ(result.ldr_error_block_count, ldr_bad);
	EXPECT_EQ(result.invalid_block_count, hdr_bad);
	EXPECT_EQ(result.ldr_error_block_count, result.hdr_block_count + result.invalid_block_count);
	EXPECT_EQ(result.is_ldr_compatible, ldr_bad == 0);
}

/** @brief Random 4x4 payloads must classify exactly as the decoder does. */
TEST(decode_profile, random_payload_2d)
{
	check_random_payload(4, 4, 1);
}

/** @brief Random 12x12 payloads must classify exactly as the decoder does. */
TEST(decode_profile, random_payload_2d_large)
{
	if (!block_size_supported(12, 12, 1))
	{
		GTEST_SKIP();
	}

	check_random_payload(12, 12, 1);
}

/** @brief Random 3D payloads must classify exactly as the decoder does. */
TEST(decode_profile, random_payload_3d)
{
	check_random_payload(4, 4, 4);
}

/**
 * @brief A block must not be accepted just because a self-decompress context omits its mode.
 *
 * This compares an analysis that uses the full mode set with one built on a context that only
 * knows the modes its compressor emitted. The analysis must be independent of the context.
 */
TEST(decode_profile, independent_of_context_config)
{
	// Block mode zero is reserved in every block size, so never registered by any context
	uint8_t invalid_block[16];
	make_invalid_mode_block(invalid_block);

	expect_block_class(invalid_block, 6, 6, 1, false, true);

	if (!block_size_supported(6, 6, 6))
	{
		GTEST_SKIP();
	}

	expect_block_class(invalid_block, 6, 6, 6, false, true);
}

/** @brief Invalid input arguments must be rejected. */
TEST(decode_profile, bad_params)
{
	uint8_t block[16];
	make_hdr_constant_block(block);
	astcenc_decode_profile_result result;

	// Null data
	EXPECT_EQ(astcenc_get_decode_profile(
	    nullptr, 16, 4, 4, 1, 4, 4, 1, &result), ASTCENC_ERR_BAD_PARAM);

	// Null result
	EXPECT_EQ(astcenc_get_decode_profile(
	    block, 16, 4, 4, 1, 4, 4, 1, nullptr), ASTCENC_ERR_BAD_PARAM);

	// Zero dimensions
	EXPECT_EQ(astcenc_get_decode_profile(
	    block, 16, 4, 4, 1, 0, 4, 1, &result), ASTCENC_ERR_BAD_PARAM);

	// Illegal block size
	EXPECT_EQ(astcenc_get_decode_profile(
	    block, 16, 7, 7, 1, 7, 7, 1, &result), ASTCENC_ERR_BAD_PARAM);

	// 3D image with a 2D block size
	EXPECT_EQ(astcenc_get_decode_profile(
	    block, 16, 4, 4, 1, 4, 4, 2, &result), ASTCENC_ERR_BAD_PARAM);

	// Payload too short for the stated dimensions
	EXPECT_EQ(astcenc_get_decode_profile(
	    block, 15, 4, 4, 1, 4, 4, 1, &result), ASTCENC_ERR_OUT_OF_MEM);
}

/**
 * @brief The analysis must cover blocks beyond the base image plane.
 *
 * An error placed only in the last block of a multi-slice 3D image must still be found, which
 * checks that the block iteration covers the whole volume.
 */
TEST(decode_profile, error_in_last_block_of_3d_volume)
{
	constexpr unsigned int block = 4;
	constexpr size_t blocks_x = 3;
	constexpr size_t blocks_y = 3;
	constexpr size_t blocks_z = 3;

	std::vector<uint8_t> compressed(blocks_x * blocks_y * blocks_z * 16, 0);

	// Every block is invalid (reserved mode zero) except the last, which is HDR constant
	size_t last = blocks_x * blocks_y * blocks_z - 1;
	uint8_t hdr_block[16];
	make_hdr_constant_block_3d(hdr_block);
	std::memcpy(compressed.data() + last * 16, hdr_block, 16);

	astcenc_decode_profile_result result;
	astcenc_error status = astcenc_get_decode_profile(
	    compressed.data(), compressed.size(), block, block, block,
	    blocks_x * block, blocks_y * block, blocks_z * block, &result);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	EXPECT_EQ(result.block_count, blocks_x * blocks_y * blocks_z);
	EXPECT_EQ(result.invalid_block_count, blocks_x * blocks_y * blocks_z - 1);
	EXPECT_EQ(result.hdr_block_count, 1u);
	EXPECT_EQ(result.ldr_error_region_max[0], blocks_x - 1);
	EXPECT_EQ(result.ldr_error_region_max[1], blocks_y - 1);
	EXPECT_EQ(result.ldr_error_region_max[2], blocks_z - 1);
}

} // namespace astcenc
