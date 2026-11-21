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
 * @brief Compressed texture GPU decode profile analysis.
 *
 * The analysis answers the question a texture packing tool needs to answer: can a compressed
 * texture be sampled correctly on a device that only implements the LDR ASTC decode profiles, or
 * does it require the optional HDR profile? It also distinguishes the blocks that merely need HDR
 * from the blocks that are invalid in every profile.
 *
 * The classification is deliberately decoupled from any codec context. In particular it does not
 * use @c astcenc_get_block_info(), because that function:
 *
 *   * returns early for constant color blocks, so the FP16 constant color blocks that need HDR are
 *     never reported as HDR; and
 *   * unpacks endpoints using the context profile, so an LDR context clears the HDR flags and
 *     reports HDR endpoint blocks as ordinary LDR blocks.
 *
 * Instead, the analysis parses each block against a block size descriptor that contains every mode
 * and partitioning the ASTC format defines for the block size, regardless of any compressor
 * configuration. This means a block is never accepted just because a self-decompress-only
 * compressor context omitted the encoding it uses.
 */

#include "astcenc_internal.h"

/**
 * @brief The decode profile requirement of a single compressed block.
 */
enum class block_decode_class
{
	/** @brief The block decodes correctly in every supported profile. */
	LDR,

	/** @brief The block needs the HDR profile; it decodes as an error color on LDR-only hardware. */
	HDR_REQUIRED,

	/** @brief The block is invalid in every supported profile. */
	INVALID
};

/**
 * @brief Query if an endpoint format uses an HDR encoding.
 *
 * Any of these formats is decoded as an error color in an LDR profile, so a block that uses one in
 * any partition requires HDR decode support.
 *
 * @param format   The endpoint format.
 *
 * @return @c true if the format needs the HDR decode profile, @c false otherwise.
 */
static bool is_hdr_endpoint_format(
	unsigned int format
) {
	switch (format)
	{
	case FMT_HDR_LUMINANCE_LARGE_RANGE:
	case FMT_HDR_LUMINANCE_SMALL_RANGE:
	case FMT_HDR_RGB_SCALE:
	case FMT_HDR_RGB:
	case FMT_HDR_RGB_LDR_ALPHA:
	case FMT_HDR_RGBA:
		return true;
	default:
		return false;
	}
}

/**
 * @brief Validate a block size for profile analysis.
 *
 * @param block_x   The block size X dimension, in texels.
 * @param block_y   The block size Y dimension, in texels.
 * @param block_z   The block size Z dimension, in texels; at least one.
 *
 * @return @c ASTCENC_SUCCESS if supported, an error otherwise.
 */
static astcenc_error validate_analysis_block_size(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z
) {
	bool is_legal;
	if (block_z <= 1)
	{
		is_legal = is_legal_2d_block_size(block_x, block_y);
	}
	else
	{
		is_legal = is_legal_3d_block_size(block_x, block_y, block_z);
	}

	if (!is_legal)
	{
		return ASTCENC_ERR_BAD_PARAM;
	}

	// Test if this build has sufficient capacity for this block size
	size_t texel_count = static_cast<size_t>(block_x) * block_y * block_z;
	if (texel_count > BLOCK_MAX_TEXELS)
	{
		return ASTCENC_ERR_NOT_IMPLEMENTED;
	}

	return ASTCENC_SUCCESS;
}

/**
 * @brief Build the block size descriptor used to analyze blocks of one block size.
 *
 * The descriptor is created with mode omission disabled and a full percentile cutoff. This is
 * required for a decode capability check, as it must consider every encoding a GPU could meet, not
 * just the encodings a particular compressor context would emit.
 *
 * @param      block_x   The block size X dimension, in texels.
 * @param      block_y   The block size Y dimension, in texels.
 * @param      block_z   The block size Z dimension, in texels; at least one.
 *
 * @return The allocated descriptor, or @c nullptr on allocation failure.
 */
static block_size_descriptor* build_analysis_bsd(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z
) {
	block_size_descriptor* bsd =
		aligned_malloc<block_size_descriptor>(sizeof(block_size_descriptor), ASTCENC_VECALIGN);
	if (!bsd)
	{
		return nullptr;
	}

	init_block_size_descriptor(block_x, block_y, block_z,
	                           false,
	                           BLOCK_MAX_PARTITIONS,
	                           1.0f,
	                           *bsd);
	return bsd;
}

/**
 * @brief Classify a single compressed block.
 *
 * @param      bsd           The block size descriptor to parse against.
 * @param      data          The compressed block.
 *
 * @return The decode profile requirement of the block.
 */
static block_decode_class classify_block(
	const block_size_descriptor& bsd,
	const uint8_t data[16]
) {
	symbolic_compressed_block scb;
	physical_to_symbolic(bsd, data, scb);

	// A block that is invalid in the full mode set is invalid in every profile
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return block_decode_class::INVALID;
	}

	// FP16 constant color blocks are only decodable in an HDR profile. UNORM16 constant color
	// blocks are legal in every profile.
	if (scb.block_type == SYM_BTYPE_CONST_F16)
	{
		return block_decode_class::HDR_REQUIRED;
	}

	if (scb.block_type == SYM_BTYPE_CONST_U16)
	{
		return block_decode_class::LDR;
	}

	// Non-constant block; any HDR endpoint format requires the HDR profile
	for (unsigned int i = 0; i < scb.partition_count; i++)
	{
		if (is_hdr_endpoint_format(scb.color_formats[i]))
		{
			return block_decode_class::HDR_REQUIRED;
		}
	}

	return block_decode_class::LDR;
}

/* See header for documentation. */
astcenc_error astcenc_get_decode_profile(
	const uint8_t* data,
	size_t data_len,
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	size_t dim_x,
	size_t dim_y,
	size_t dim_z,
	astcenc_decode_profile_result* result
) {
	if ((!data) || (!result) || (dim_x == 0) || (dim_y == 0) || (dim_z == 0))
	{
		return ASTCENC_ERR_BAD_PARAM;
	}

	block_z = astc::max(block_z, 1u);

	// Match the decompressor: volumetric images require a three-dimensional block size
	if ((dim_z > 1) && (block_z == 1))
	{
		return ASTCENC_ERR_BAD_PARAM;
	}

	astcenc_error status = validate_analysis_block_size(block_x, block_y, block_z);
	if (status != ASTCENC_SUCCESS)
	{
		return status;
	}

	size_t blocks_x = astc::get_block_count_safe(dim_x, block_x);
	size_t blocks_y = astc::get_block_count_safe(dim_y, block_y);
	size_t blocks_z = astc::get_block_count_safe(dim_z, block_z);

	bool overflow { false };
	size_t block_count = astc::mul_safe(blocks_x, blocks_y, overflow);
	block_count = astc::mul_safe(block_count, blocks_z, overflow);
	size_t size_needed = astc::mul_safe(block_count, 16, overflow);

	// A zero block count means the dimension arithmetic overflowed
	if (overflow || (block_count == 0))
	{
		return ASTCENC_ERR_BAD_PARAM;
	}

	if (data_len < size_needed)
	{
		return ASTCENC_ERR_OUT_OF_MEM;
	}

	block_size_descriptor* bsd = build_analysis_bsd(block_x, block_y, block_z);
	if (!bsd)
	{
		return ASTCENC_ERR_OUT_OF_MEM;
	}

	astcenc_decode_profile_result analysis {};
	bool first_error = true;

	for (size_t z = 0; z < blocks_z; z++)
	{
		for (size_t y = 0; y < blocks_y; y++)
		{
			for (size_t x = 0; x < blocks_x; x++)
			{
				size_t offset = (((z * blocks_y + y) * blocks_x) + x) * 16;
				block_decode_class block_class = classify_block(*bsd, data + offset);

				if (block_class == block_decode_class::INVALID)
				{
					analysis.invalid_block_count++;
				}
				else if (block_class == block_decode_class::HDR_REQUIRED)
				{
					analysis.hdr_block_count++;
				}

				// HDR-required and invalid blocks both decode as an error color on LDR-only
				// hardware, so both contribute to the LDR error count and region.
				if (block_class != block_decode_class::LDR)
				{
					analysis.ldr_error_block_count++;

					if (first_error)
					{
						analysis.ldr_error_region_min[0] = x;
						analysis.ldr_error_region_min[1] = y;
						analysis.ldr_error_region_min[2] = z;
						analysis.ldr_error_region_max[0] = x;
						analysis.ldr_error_region_max[1] = y;
						analysis.ldr_error_region_max[2] = z;
						first_error = false;
					}
					else
					{
						analysis.ldr_error_region_min[0] = astc::min(analysis.ldr_error_region_min[0], x);
						analysis.ldr_error_region_min[1] = astc::min(analysis.ldr_error_region_min[1], y);
						analysis.ldr_error_region_min[2] = astc::min(analysis.ldr_error_region_min[2], z);
						analysis.ldr_error_region_max[0] = astc::max(analysis.ldr_error_region_max[0], x);
						analysis.ldr_error_region_max[1] = astc::max(analysis.ldr_error_region_max[1], y);
						analysis.ldr_error_region_max[2] = astc::max(analysis.ldr_error_region_max[2], z);
					}
				}
			}
		}
	}

	analysis.block_count = block_count;
	analysis.block_x = block_x;
	analysis.block_y = block_y;
	analysis.block_z = block_z;
	analysis.dim_x = dim_x;
	analysis.dim_y = dim_y;
	analysis.dim_z = dim_z;
	analysis.blocks_x = blocks_x;
	analysis.blocks_y = blocks_y;
	analysis.blocks_z = blocks_z;
	analysis.has_ldr_error_region = !first_error;
	analysis.requires_hdr = analysis.hdr_block_count > 0;
	analysis.is_ldr_compatible = analysis.ldr_error_block_count == 0;

	*result = analysis;

	aligned_free<const block_size_descriptor>(bsd, ASTCENC_VECALIGN);
	return ASTCENC_SUCCESS;
}
