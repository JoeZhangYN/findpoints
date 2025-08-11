extern crate libc;

use rayon::prelude::*;
use std;
use std::arch::x86_64::*;
use std::os::raw::c_double;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;

// CPU特性检测
pub struct CpuFeatures {
    pub has_avx512: bool,
    pub has_avx2: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return CpuFeatures {
                    has_avx512: true,
                    has_avx2: true,
                };
            } else if is_x86_feature_detected!("avx2") {
                return CpuFeatures {
                    has_avx512: false,
                    has_avx2: true,
                };
            }
        }
        
        CpuFeatures {
            has_avx512: false,
            has_avx2: false,
        }
    }
}

// 全局CPU特性缓存
lazy_static::lazy_static! {
    static ref CPU_FEATURES: CpuFeatures = CpuFeatures::detect();
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Tuple {
    pub x: u32,
    pub y: u32,
}

impl Tuple {
    pub const NOT_FOUND: Tuple = Tuple { x: 245760, y: 143640 };
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[repr(C)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
pub struct MultipleResults {
    pub points: *mut Tuple,
    pub count: i32,
    pub capacity: i32,
}

#[repr(C)]
pub struct FindAllConfig {
    pub max_results: i32,
    pub min_distance: i32,
    pub early_exit: bool,
}

// 匹配参数结构体
struct MatchParams {
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    first_pixel_r: u8,
    first_pixel_g: u8,
    first_pixel_b: u8,
    total_pixels: usize,
    min_match_pixels: usize,
}

// 验证输入参数
unsafe fn validate_inputs(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
) -> Result<(), &'static str> {
    // 验证指针
    if n1.is_null() || n2.is_null() {
        return Err("图像数据指针为空");
    }
    
    // 验证长度
    if len1 == 0 || len2 == 0 {
        return Err("无效的图像数据长度");
    }
    
    // 验证图像尺寸
    if tup1.x == 0 || tup1.y == 0 || tup2.x == 0 || tup2.y == 0 {
        return Err("无效的图像尺寸");
    }
    
    // 确保子图像不大于主图像
    if tup2.x > tup1.x || tup2.y > tup1.y {
        return Err("子图像尺寸大于主图像");
    }
    
    // 验证数据长度与尺寸匹配
    let expected_len1 = (tup1.x * tup1.y * 4) as usize;
    let expected_len2 = (tup2.x * tup2.y * 4) as usize;
    
    if len1 != expected_len1 || len2 != expected_len2 {
        return Err("数据长度与图像尺寸不匹配");
    }
    
    Ok(())
}

// AVX512版本的详细匹配实现
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn perform_detailed_match_avx512(
    numbers1: &[u8],
    numbers2: &[u8],
    i: u32,
    j: u32,
    par_x: u32,
    sub_x: u32,
    sub_y: u32,
    params: &MatchParams,
) -> bool {
    let start_par_index = (j * par_x * 4 + i * 4) as usize;

    // 快速检查第一个像素
    let (par_r, par_g, par_b) = (
        *numbers1.get_unchecked(start_par_index),
        *numbers1.get_unchecked(start_par_index + 1),
        *numbers1.get_unchecked(start_par_index + 2),
    );

    if par_r == params.ignore_r && par_g == params.ignore_g && par_b == params.ignore_b {
        return false;
    }

    if par_r != params.first_pixel_r
        || par_g != params.first_pixel_g
        || par_b != params.first_pixel_b
    {
        return false;
    }

    // 使用AVX512处理更多像素
    let ignore_color_r = _mm512_set1_epi8(params.ignore_r as i8);
    let ignore_color_g = _mm512_set1_epi8(params.ignore_g as i8);
    let ignore_color_b = _mm512_set1_epi8(params.ignore_b as i8);

    let mut matched_pixels = 0usize;
    let mut total_checked_pixels = 0usize;

    // AVX512可以一次处理16个像素
    for block_j in (0..sub_y).step_by(4) {
        let actual_block_height = std::cmp::min(4, sub_y - block_j);

        for block_i in (0..sub_x).step_by(16) {
            let actual_block_width = std::cmp::min(16, sub_x - block_i);

            for local_j in 0..actual_block_height {
                let row_j = block_j + local_j;
                let sub_row_start = (row_j * sub_x * 4) as usize;
                let par_row_start = ((j + row_j) * par_x * 4 + (i + block_i) * 4) as usize;

                let pixels_remaining = actual_block_width;
                let full_vectors = pixels_remaining / 16;
                let remaining_pixels = pixels_remaining % 16;

                // AVX512向量处理
                for vec_idx in 0..full_vectors {
                    let sub_offset = sub_row_start + ((block_i + vec_idx * 16) * 4) as usize;
                    let par_offset = par_row_start + (vec_idx * 16 * 4) as usize;

                    if par_offset + 64 > numbers1.len() || sub_offset + 64 > numbers2.len() {
                        break;
                    }

                    let sub_pixels = _mm512_loadu_si512(numbers2.as_ptr().add(sub_offset) as *const __m512i);
                    let par_pixels = _mm512_loadu_si512(numbers1.as_ptr().add(par_offset) as *const __m512i);

                    // 分离RGB通道
                    let shuffle_r = _mm512_set_epi8(
                        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
                        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
                        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
                        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
                    );
                    
                    let shuffle_g = _mm512_set_epi8(
                        61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1,
                        61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1,
                        61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1,
                        61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1,
                    );
                    
                    let shuffle_b = _mm512_set_epi8(
                        62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2,
                        62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2,
                        62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2,
                        62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2,
                    );

                    let sub_r = _mm512_shuffle_epi8(sub_pixels, shuffle_r);
                    let sub_g = _mm512_shuffle_epi8(sub_pixels, shuffle_g);
                    let sub_b = _mm512_shuffle_epi8(sub_pixels, shuffle_b);

                    let par_r = _mm512_shuffle_epi8(par_pixels, shuffle_r);
                    let par_g = _mm512_shuffle_epi8(par_pixels, shuffle_g);
                    let par_b = _mm512_shuffle_epi8(par_pixels, shuffle_b);

                    // 检查忽略色
                    let ignore_mask_r = _mm512_cmpeq_epi8_mask(par_r, ignore_color_r);
                    let ignore_mask_g = _mm512_cmpeq_epi8_mask(par_g, ignore_color_g);
                    let ignore_mask_b = _mm512_cmpeq_epi8_mask(par_b, ignore_color_b);
                    let ignore_mask = ignore_mask_r & ignore_mask_g & ignore_mask_b;

                    // 比较RGB通道
                    let match_r = _mm512_cmpeq_epi8_mask(sub_r, par_r);
                    let match_g = _mm512_cmpeq_epi8_mask(sub_g, par_g);
                    let match_b = _mm512_cmpeq_epi8_mask(sub_b, par_b);
                    let match_mask = match_r & match_g & match_b;

                    // 排除忽略色的匹配
                    let valid_match_mask = match_mask & !ignore_mask;
                    let valid_pixel_mask = !ignore_mask;

                    matched_pixels += valid_match_mask.count_ones() as usize / 4;
                    total_checked_pixels += valid_pixel_mask.count_ones() as usize / 4;
                }

                // 处理剩余像素
                for pixel_idx in 0..remaining_pixels {
                    let pixel_offset = block_i + full_vectors * 16 + pixel_idx;
                    let sub_offset = sub_row_start + (pixel_offset * 4) as usize;
                    let par_offset = par_row_start + ((full_vectors * 16 + pixel_idx) * 4) as usize;

                    let sub_r = *numbers2.get_unchecked(sub_offset);
                    let sub_g = *numbers2.get_unchecked(sub_offset + 1);
                    let sub_b = *numbers2.get_unchecked(sub_offset + 2);

                    let par_r = *numbers1.get_unchecked(par_offset);
                    let par_g = *numbers1.get_unchecked(par_offset + 1);
                    let par_b = *numbers1.get_unchecked(par_offset + 2);

                    if par_r == params.ignore_r
                        && par_g == params.ignore_g
                        && par_b == params.ignore_b
                    {
                        continue;
                    }

                    total_checked_pixels += 1;

                    if sub_r == par_r && sub_g == par_g && sub_b == par_b {
                        matched_pixels += 1;
                    }
                }
            }

            // 早期终止检查
            let remaining_pixels = params.total_pixels - total_checked_pixels;
            let max_possible_matches = matched_pixels + remaining_pixels;

            if max_possible_matches < params.min_match_pixels {
                return false;
            }
        }
    }

    if total_checked_pixels > 0 {
        let final_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
        return final_match_rate >= (params.min_match_pixels as f64 / params.total_pixels as f64);
    }

    false
}

// AVX2版本
#[target_feature(enable = "avx2")]
unsafe fn perform_detailed_match_avx2(
    numbers1: &[u8],
    numbers2: &[u8],
    i: u32,
    j: u32,
    par_x: u32,
    sub_x: u32,
    sub_y: u32,
    params: &MatchParams,
) -> bool {
    let start_par_index = (j * par_x * 4 + i * 4) as usize;

    let (par_r, par_g, par_b) = (
        *numbers1.get_unchecked(start_par_index),
        *numbers1.get_unchecked(start_par_index + 1),
        *numbers1.get_unchecked(start_par_index + 2),
    );

    if par_r == params.ignore_r && par_g == params.ignore_g && par_b == params.ignore_b {
        return false;
    }

    if par_r != params.first_pixel_r
        || par_g != params.first_pixel_g
        || par_b != params.first_pixel_b
    {
        return false;
    }

    let ignore_color_r = _mm256_set1_epi8(params.ignore_r as i8);
    let ignore_color_g = _mm256_set1_epi8(params.ignore_g as i8);
    let ignore_color_b = _mm256_set1_epi8(params.ignore_b as i8);

    let mut matched_pixels = 0usize;
    let mut total_checked_pixels = 0usize;

    for block_j in (0..sub_y).step_by(4) {
        let actual_block_height = std::cmp::min(4, sub_y - block_j);

        for block_i in (0..sub_x).step_by(8) {
            let actual_block_width = std::cmp::min(8, sub_x - block_i);

            for local_j in 0..actual_block_height {
                let row_j = block_j + local_j;
                let sub_row_start = (row_j * sub_x * 4) as usize;
                let par_row_start = ((j + row_j) * par_x * 4 + (i + block_i) * 4) as usize;

                let pixels_remaining = actual_block_width;
                let full_vectors = pixels_remaining / 8;
                let remaining_pixels = pixels_remaining % 8;

                for vec_idx in 0..full_vectors {
                    let sub_offset = sub_row_start + ((block_i + vec_idx * 8) * 4) as usize;
                    let par_offset = par_row_start + (vec_idx * 8 * 4) as usize;

                    if par_offset + 32 > numbers1.len() || sub_offset + 32 > numbers2.len() {
                        break;
                    }

                    let sub_pixels =
                        _mm256_loadu_si256(numbers2.as_ptr().add(sub_offset) as *const __m256i);
                    let par_pixels =
                        _mm256_loadu_si256(numbers1.as_ptr().add(par_offset) as *const __m256i);

                    let sub_r = _mm256_and_si256(sub_pixels, _mm256_set1_epi32(0x000000FF));
                    let sub_g = _mm256_and_si256(
                        _mm256_srli_epi32(sub_pixels, 8),
                        _mm256_set1_epi32(0x000000FF),
                    );
                    let sub_b = _mm256_and_si256(
                        _mm256_srli_epi32(sub_pixels, 16),
                        _mm256_set1_epi32(0x000000FF),
                    );

                    let par_r = _mm256_and_si256(par_pixels, _mm256_set1_epi32(0x000000FF));
                    let par_g = _mm256_and_si256(
                        _mm256_srli_epi32(par_pixels, 8),
                        _mm256_set1_epi32(0x000000FF),
                    );
                    let par_b = _mm256_and_si256(
                        _mm256_srli_epi32(par_pixels, 16),
                        _mm256_set1_epi32(0x000000FF),
                    );

                    let ignore_mask_r = _mm256_cmpeq_epi8(par_r, ignore_color_r);
                    let ignore_mask_g = _mm256_cmpeq_epi8(par_g, ignore_color_g);
                    let ignore_mask_b = _mm256_cmpeq_epi8(par_b, ignore_color_b);
                    let ignore_mask = _mm256_and_si256(
                        _mm256_and_si256(ignore_mask_r, ignore_mask_g),
                        ignore_mask_b,
                    );

                    let match_r = _mm256_cmpeq_epi8(sub_r, par_r);
                    let match_g = _mm256_cmpeq_epi8(sub_g, par_g);
                    let match_b = _mm256_cmpeq_epi8(sub_b, par_b);
                    let match_mask = _mm256_and_si256(_mm256_and_si256(match_r, match_g), match_b);

                    let valid_match_mask = _mm256_andnot_si256(ignore_mask, match_mask);
                    let valid_pixel_mask = _mm256_xor_si256(ignore_mask, _mm256_set1_epi8(-1));

                    let matched_bytes =
                        _mm256_movemask_epi8(valid_match_mask).count_ones() as usize;
                    let total_bytes = _mm256_movemask_epi8(valid_pixel_mask).count_ones() as usize;

                    matched_pixels += matched_bytes / 4;
                    total_checked_pixels += total_bytes / 4;
                }

                for pixel_idx in 0..remaining_pixels {
                    let pixel_offset = block_i + full_vectors * 8 + pixel_idx;
                    let sub_offset = sub_row_start + (pixel_offset * 4) as usize;
                    let par_offset = par_row_start + ((full_vectors * 8 + pixel_idx) * 4) as usize;

                    let sub_r = *numbers2.get_unchecked(sub_offset);
                    let sub_g = *numbers2.get_unchecked(sub_offset + 1);
                    let sub_b = *numbers2.get_unchecked(sub_offset + 2);

                    let par_r = *numbers1.get_unchecked(par_offset);
                    let par_g = *numbers1.get_unchecked(par_offset + 1);
                    let par_b = *numbers1.get_unchecked(par_offset + 2);

                    if par_r == params.ignore_r
                        && par_g == params.ignore_g
                        && par_b == params.ignore_b
                    {
                        continue;
                    }

                    total_checked_pixels += 1;

                    if sub_r == par_r && sub_g == par_g && sub_b == par_b {
                        matched_pixels += 1;
                    }
                }
            }

            let remaining_pixels = params.total_pixels - total_checked_pixels;
            let max_possible_matches = matched_pixels + remaining_pixels;

            if max_possible_matches < params.min_match_pixels {
                return false;
            }
        }
    }

    if total_checked_pixels > 0 {
        let final_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
        return final_match_rate >= (params.min_match_pixels as f64 / params.total_pixels as f64);
    }

    false
}

// 标准多线程版本（无SIMD）
unsafe fn perform_detailed_match_standard(
    numbers1: &[u8],
    numbers2: &[u8],
    i: u32,
    j: u32,
    par_x: u32,
    sub_x: u32,
    sub_y: u32,
    params: &MatchParams,
) -> bool {
    let start_par_index = (j * par_x * 4 + i * 4) as usize;

    let (par_r, par_g, par_b) = (
        *numbers1.get_unchecked(start_par_index),
        *numbers1.get_unchecked(start_par_index + 1),
        *numbers1.get_unchecked(start_par_index + 2),
    );

    if par_r == params.ignore_r && par_g == params.ignore_g && par_b == params.ignore_b {
        return false;
    }

    if par_r != params.first_pixel_r
        || par_g != params.first_pixel_g
        || par_b != params.first_pixel_b
    {
        return false;
    }

    let mut matched_pixels = 0usize;
    let mut total_checked_pixels = 0usize;

    for current_j in 0..sub_y {
        let sub_row_base = current_j * sub_x * 4;
        let par_row_base = (j + current_j) * par_x * 4 + i * 4;

        for current_i in 0..sub_x {
            let sub_index = sub_row_base + current_i * 4;
            let par_index = par_row_base + current_i * 4;

            let sub_r = *numbers2.get_unchecked(sub_index as usize);
            let sub_g = *numbers2.get_unchecked(sub_index as usize + 1);
            let sub_b = *numbers2.get_unchecked(sub_index as usize + 2);

            let par_r = *numbers1.get_unchecked(par_index as usize);
            let par_g = *numbers1.get_unchecked(par_index as usize + 1);
            let par_b = *numbers1.get_unchecked(par_index as usize + 2);

            if par_r == params.ignore_r && par_g == params.ignore_g && par_b == params.ignore_b {
                continue;
            }

            total_checked_pixels += 1;

            if sub_r == par_r && sub_g == par_g && sub_b == par_b {
                matched_pixels += 1;
            }
        }

        // 早期终止检查
        let remaining_pixels = params.total_pixels - total_checked_pixels;
        let max_possible_matches = matched_pixels + remaining_pixels;

        if max_possible_matches < params.min_match_pixels {
            return false;
        }
    }

    if total_checked_pixels > 0 {
        let final_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
        return final_match_rate >= (params.min_match_pixels as f64 / params.total_pixels as f64);
    }

    false
}

// 统一的查找函数实现（内部使用）
unsafe fn find_bytes_internal(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    tolerance: Option<u8>,
    region: Option<Region>,
) -> Tuple {
    // 验证输入
    if let Err(_) = validate_inputs(n1, len1, tup1, n2, len2, tup2) {
        return Tuple::NOT_FOUND;
    }
    
    let par_x = tup1.x;
    let par_y = tup1.y;
    let sub_x = tup2.x;
    let sub_y = tup2.y;

    // 确定搜索区域
    let (start_x, end_x, start_y, end_y) = if let Some(reg) = region {
        if reg.x >= par_x || reg.y >= par_y {
            return Tuple::NOT_FOUND;
        }
        let region_end_x = (reg.x + reg.width).min(par_x);
        let region_end_y = (reg.y + reg.height).min(par_y);
        if region_end_x < reg.x + sub_x || region_end_y < reg.y + sub_y {
            return Tuple::NOT_FOUND;
        }
        (reg.x, region_end_x - sub_x, reg.y, region_end_y - sub_y)
    } else {
        (0, par_x - sub_x, 0, par_y - sub_y)
    };

    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    let match_params = MatchParams {
        ignore_r,
        ignore_g,
        ignore_b,
        first_pixel_r: numbers2[0],
        first_pixel_g: numbers2[1],
        first_pixel_b: numbers2[2],
        total_pixels: (sub_x * sub_y) as usize,
        min_match_pixels: ((sub_x * sub_y) as f64 * match_rate) as usize,
    };

    // 根据是否有容差选择不同的实现
    if let Some(tol) = tolerance {
        // 容差版本的实现
        return find_bytes_tolerance_internal(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, tol,
        );
    }

    // 选择最优的实现方式
    let result = (start_x..=end_x).into_par_iter().find_map_any(|i| {
        for j in start_y..=end_y {
            let matched = if CPU_FEATURES.has_avx512 {
                perform_detailed_match_avx512(
                    numbers1,
                    numbers2,
                    i,
                    j,
                    par_x,
                    sub_x,
                    sub_y,
                    &match_params,
                )
            } else if CPU_FEATURES.has_avx2 {
                perform_detailed_match_avx2(
                    numbers1,
                    numbers2,
                    i,
                    j,
                    par_x,
                    sub_x,
                    sub_y,
                    &match_params,
                )
            } else {
                perform_detailed_match_standard(
                    numbers1,
                    numbers2,
                    i,
                    j,
                    par_x,
                    sub_x,
                    sub_y,
                    &match_params,
                )
            };

            if matched {
                return Some(Tuple { x: i, y: j });
            }
        }
        None
    });

    result.unwrap_or(Tuple::NOT_FOUND)
}

// 容差版本的内部实现
unsafe fn find_bytes_tolerance_internal(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    tolerance: u8,
) -> Tuple {
    let par_x = tup1.x;
    let par_y = tup1.y;
    let sub_x = tup2.x;
    let sub_y = tup2.y;

    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    let first_pixel_r = numbers2[0];
    let first_pixel_g = numbers2[1];
    let first_pixel_b = numbers2[2];
    let min_match_pixels = (sub_x * sub_y) as f64 * match_rate;
    let tolerance_i32 = tolerance as i32;

    #[inline(always)]
    fn color_match_with_tolerance(
        c1_r: u8,
        c1_g: u8,
        c1_b: u8,
        c2_r: u8,
        c2_g: u8,
        c2_b: u8,
        tolerance: i32,
    ) -> bool {
        (c1_r as i32 - c2_r as i32).abs() <= tolerance
            && (c1_g as i32 - c2_g as i32).abs() <= tolerance
            && (c1_b as i32 - c2_b as i32).abs() <= tolerance
    }

    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let start_par_index = (j * par_x * 4 + i * 4) as usize;

            let par_r = *numbers1.get_unchecked(start_par_index);
            let par_g = *numbers1.get_unchecked(start_par_index + 1);
            let par_b = *numbers1.get_unchecked(start_par_index + 2);

            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            if !color_match_with_tolerance(
                par_r,
                par_g,
                par_b,
                first_pixel_r,
                first_pixel_g,
                first_pixel_b,
                tolerance_i32,
            ) {
                continue;
            }

            let mut matched_pixels = 0usize;
            let mut total_pixels = 0usize;

            for current_j in 0..sub_y {
                let sub_row_base = current_j * sub_x * 4;
                let par_row_base = (j + current_j) * par_x * 4 + i * 4;

                for current_i in 0..sub_x {
                    let sub_index = sub_row_base + current_i * 4;
                    let par_index = par_row_base + current_i * 4;

                    let sub_r = *numbers2.get_unchecked(sub_index as usize);
                    let sub_g = *numbers2.get_unchecked(sub_index as usize + 1);
                    let sub_b = *numbers2.get_unchecked(sub_index as usize + 2);

                    let par_r = *numbers1.get_unchecked(par_index as usize);
                    let par_g = *numbers1.get_unchecked(par_index as usize + 1);
                    let par_b = *numbers1.get_unchecked(par_index as usize + 2);

                    if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                        continue;
                    }

                    total_pixels += 1;

                    if color_match_with_tolerance(
                        par_r, par_g, par_b, sub_r, sub_g, sub_b, tolerance_i32,
                    ) {
                        matched_pixels += 1;
                    }
                }
            }

            if total_pixels > 0 && matched_pixels as f64 >= min_match_pixels {
                return Some(Tuple { x: i, y: j });
            }
        }
        None
    });

    result.unwrap_or(Tuple::NOT_FOUND)
}

// ========== 统一的外部接口 ==========

/// 统一的查找接口
#[no_mangle]
pub extern "C" fn FindBytes(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
) -> Tuple {
    unsafe {
        find_bytes_internal(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, None, None,
        )
    }
}

/// 带容差的查找接口
#[no_mangle]
pub extern "C" fn FindBytesWithTolerance(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    tolerance: u8,
) -> Tuple {
    unsafe {
        find_bytes_internal(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, Some(tolerance), None,
        )
    }
}

/// 区域查找接口
#[no_mangle]
pub extern "C" fn FindBytesInRegion(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
) -> Tuple {
    unsafe {
        find_bytes_internal(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, None, Some(region),
        )
    }
}

/// 查找所有匹配
#[no_mangle]
pub extern "C" fn FindAllBytesInRegion(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    config: FindAllConfig,
) -> MultipleResults {
    unsafe {
        // 验证输入
        if let Err(_) = validate_inputs(n1, len1, tup1, n2, len2, tup2) {
            return MultipleResults {
                points: std::ptr::null_mut(),
                count: 0,
                capacity: 0,
            };
        }
        
        // 根据CPU特性选择实现
        if CPU_FEATURES.has_avx512 {
            find_all_bytes_avx512(
                n1, len1, tup1, n2, len2, tup2, region, match_rate, ignore_r, ignore_g, ignore_b, config,
            )
        } else if CPU_FEATURES.has_avx2 {
            find_all_bytes_avx2(
                n1, len1, tup1, n2, len2, tup2, region, match_rate, ignore_r, ignore_g, ignore_b, config,
            )
        } else {
            find_all_bytes_standard(
                n1, len1, tup1, n2, len2, tup2, region, match_rate, ignore_r, ignore_g, ignore_b, config,
            )
        }
    }
}

// AVX512版本的批量查找
unsafe fn find_all_bytes_avx512(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    config: FindAllConfig,
) -> MultipleResults {
    find_all_bytes_internal(
        n1, len1, tup1, n2, len2, tup2, region, match_rate, 
        ignore_r, ignore_g, ignore_b, config, 
        |numbers1, numbers2, i, j, par_x, sub_x, sub_y, params| {
            unsafe {
                perform_detailed_match_avx512(numbers1, numbers2, i, j, par_x, sub_x, sub_y, params)
            }
        }
    )
}

// AVX2版本的批量查找
unsafe fn find_all_bytes_avx2(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    config: FindAllConfig,
) -> MultipleResults {
    find_all_bytes_internal(
        n1, len1, tup1, n2, len2, tup2, region, match_rate, 
        ignore_r, ignore_g, ignore_b, config, 
        |numbers1, numbers2, i, j, par_x, sub_x, sub_y, params| {
            unsafe {
                perform_detailed_match_avx2(numbers1, numbers2, i, j, par_x, sub_x, sub_y, params)
            }
        }
    )
}

// 标准版本的批量查找
unsafe fn find_all_bytes_standard(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    config: FindAllConfig,
) -> MultipleResults {
    find_all_bytes_internal(
        n1, len1, tup1, n2, len2, tup2, region, match_rate, 
        ignore_r, ignore_g, ignore_b, config, 
        |numbers1, numbers2, i, j, par_x, sub_x, sub_y, params| {
            unsafe {
                perform_detailed_match_standard(numbers1, numbers2, i, j, par_x, sub_x, sub_y, params)
            }
        }
    )
}

// 通用的批量查找内部实现
unsafe fn find_all_bytes_internal<F>(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    region: Region,
    match_rate: c_double,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    config: FindAllConfig,
    match_fn: F,
) -> MultipleResults 
where
    F: Fn(&[u8], &[u8], u32, u32, u32, u32, u32, &MatchParams) -> bool + Sync,
{
    let par_x = tup1.x;
    let par_y = tup1.y;
    let sub_x = tup2.x;
    let sub_y = tup2.y;

    if region.x >= par_x || region.y >= par_y {
        return MultipleResults {
            points: std::ptr::null_mut(),
            count: 0,
            capacity: 0,
        };
    }

    let region_end_x = (region.x + region.width).min(par_x);
    let region_end_y = (region.y + region.height).min(par_y);

    if region_end_x < region.x + sub_x || region_end_y < region.y + sub_y {
        return MultipleResults {
            points: std::ptr::null_mut(),
            count: 0,
            capacity: 0,
        };
    }

    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    let match_params = MatchParams {
        ignore_r,
        ignore_g,
        ignore_b,
        first_pixel_r: numbers2[0],
        first_pixel_g: numbers2[1],
        first_pixel_b: numbers2[2],
        total_pixels: (sub_x * sub_y) as usize,
        min_match_pixels: ((sub_x * sub_y) as f64 * match_rate) as usize,
    };

    let results: Mutex<Vec<Tuple>> = Mutex::new(Vec::with_capacity(config.max_results as usize));
    let found_count = AtomicI32::new(0);

    let strip_height = (sub_y as usize).max(50);
    let num_strips = ((region_end_y - region.y) as usize + strip_height - 1) / strip_height;

    (0..num_strips).into_par_iter().for_each(|strip_idx| {
        let strip_start_y = region.y + (strip_idx * strip_height) as u32;
        let strip_end_y = (strip_start_y + strip_height as u32 + sub_y).min(region_end_y);

        if config.early_exit && found_count.load(Ordering::Relaxed) >= config.max_results {
            return;
        }

        for j in strip_start_y..=strip_end_y.saturating_sub(sub_y) {
            for i in region.x..=region_end_x - sub_x {
                if config.early_exit && found_count.load(Ordering::Relaxed) >= config.max_results {
                    return;
                }

                if match_fn(
                    numbers1,
                    numbers2,
                    i,
                    j,
                    par_x,
                    sub_x,
                    sub_y,
                    &match_params,
                ) {
                    let mut should_add = true;

                    if config.min_distance > 0 {
                        let mut results_guard = results.lock().unwrap();

                        for existing in results_guard.iter() {
                            let dx = ((existing.x as i32) - (i as i32)).abs();
                            let dy = ((existing.y as i32) - (j as i32)).abs();

                            if dx < config.min_distance && dy < config.min_distance {
                                should_add = false;
                                break;
                            }
                        }

                        if should_add {
                            results_guard.push(Tuple { x: i, y: j });
                            found_count.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        results.lock().unwrap().push(Tuple { x: i, y: j });
                        found_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });

    let mut final_results = results.into_inner().unwrap();
    let count = final_results.len() as i32;
    let capacity = final_results.capacity() as i32;
    let points = final_results.as_mut_ptr();

    std::mem::forget(final_results);

    MultipleResults {
        points,
        count,
        capacity,
    }
}

/// 释放结果内存
#[no_mangle]
pub extern "C" fn FreeMultipleResults(results: MultipleResults) {
    if !results.points.is_null() && results.capacity > 0 {
        unsafe {
            Vec::from_raw_parts(
                results.points,
                results.count as usize,
                results.capacity as usize,
            );
        }
    }
}

// ========== 颜色比较函数 ==========

/// 创建RGBA颜色
#[no_mangle]
pub extern "C" fn rgba_new(r: u8, g: u8, b: u8, a: u8) -> Rgba {
    Rgba { r, g, b, a }
}

/// 颜色比较（统一接口）
#[no_mangle]
pub extern "C" fn color_equal(
    color_a: *const Rgba,
    color_b: *const Rgba,
    error_range: u8,
) -> bool {
    unsafe {
        if color_a.is_null() || color_b.is_null() {
            return false;
        }
        
        if CPU_FEATURES.has_avx512 {
            color_equal_avx512(color_a, color_b, error_range as i32)
        } else if CPU_FEATURES.has_avx2 {
            color_equal_avx2(color_a, color_b, error_range as i32)
        } else {
            color_equal_standard(color_a, color_b, error_range as i32)
        }
    }
}

/// RGB颜色比较（统一接口）
#[no_mangle]
pub extern "C" fn color_equal_rgb(
    color_a: *const Rgba,
    color_b: *const Rgba,
    error_r: u8,
    error_g: u8,
    error_b: u8,
) -> bool {
    unsafe {
        if color_a.is_null() || color_b.is_null() {
            return false;
        }
        
        if CPU_FEATURES.has_avx512 {
            color_equal_rgb_avx512(
                color_a,
                color_b,
                error_r as i32,
                error_g as i32,
                error_b as i32,
            )
        } else if CPU_FEATURES.has_avx2 {
            color_equal_rgb_avx2(
                color_a,
                color_b,
                error_r as i32,
                error_g as i32,
                error_b as i32,
            )
        } else {
            color_equal_rgb_standard(
                color_a,
                color_b,
                error_r as i32,
                error_g as i32,
                error_b as i32,
            )
        }
    }
}

// AVX512版本的颜色比较
#[target_feature(enable = "avx512f")]
unsafe fn color_equal_avx512(a: *const Rgba, b: *const Rgba, error: i32) -> bool {
    let a_vals = _mm_set_epi32((*a).a as i32, (*a).r as i32, (*a).g as i32, (*a).b as i32);
    let b_vals = _mm_set_epi32((*b).a as i32, (*b).r as i32, (*b).g as i32, (*b).b as i32);
    let error_vals = _mm_set1_epi32(error);

    let diff = _mm_sub_epi32(a_vals, b_vals);
    let abs_diff = _mm_abs_epi32(diff);
    let cmp = _mm_cmpgt_epi32(abs_diff, error_vals);

    _mm_testz_si128(cmp, cmp) != 0
}

// AVX2版本的颜色比较
#[target_feature(enable = "avx2")]
unsafe fn color_equal_avx2(a: *const Rgba, b: *const Rgba, error: i32) -> bool {
    let a_vals = _mm_set_epi32((*a).a as i32, (*a).r as i32, (*a).g as i32, (*a).b as i32);
    let b_vals = _mm_set_epi32((*b).a as i32, (*b).r as i32, (*b).g as i32, (*b).b as i32);
    let error_vals = _mm_set1_epi32(error);

    let diff = _mm_sub_epi32(a_vals, b_vals);
    let abs_diff = _mm_abs_epi32(diff);
    let cmp = _mm_cmpgt_epi32(abs_diff, error_vals);

    _mm_testz_si128(cmp, cmp) != 0
}

// 标准版本的颜色比较
unsafe fn color_equal_standard(a: *const Rgba, b: *const Rgba, error: i32) -> bool {
    let a = &*a;
    let b = &*b;
    
    (a.r as i32 - b.r as i32).abs() <= error
        && (a.g as i32 - b.g as i32).abs() <= error
        && (a.b as i32 - b.b as i32).abs() <= error
        && (a.a as i32 - b.a as i32).abs() <= error
}

// AVX512版本的RGB颜色比较
#[target_feature(enable = "avx512f")]
unsafe fn color_equal_rgb_avx512(
    a: *const Rgba,
    b: *const Rgba,
    error_r: i32,
    error_g: i32,
    error_b: i32,
) -> bool {
    let a_vals = _mm_set_epi32(0, (*a).r as i32, (*a).g as i32, (*a).b as i32);
    let b_vals = _mm_set_epi32(0, (*b).r as i32, (*b).g as i32, (*b).b as i32);
    let error_vals = _mm_set_epi32(0, error_r, error_g, error_b);

    let diff = _mm_sub_epi32(a_vals, b_vals);
    let abs_diff = _mm_abs_epi32(diff);
    let cmp = _mm_cmpgt_epi32(abs_diff, error_vals);

    _mm_testz_si128(cmp, cmp) != 0
}

// AVX2版本的RGB颜色比较
#[target_feature(enable = "avx2")]
unsafe fn color_equal_rgb_avx2(
    a: *const Rgba,
    b: *const Rgba,
    error_r: i32,
    error_g: i32,
    error_b: i32,
) -> bool {
    let a_vals = _mm_set_epi32(0, (*a).r as i32, (*a).g as i32, (*a).b as i32);
    let b_vals = _mm_set_epi32(0, (*b).r as i32, (*b).g as i32, (*b).b as i32);
    let error_vals = _mm_set_epi32(0, error_r, error_g, error_b);

    let diff = _mm_sub_epi32(a_vals, b_vals);
    let abs_diff = _mm_abs_epi32(diff);
    let cmp = _mm_cmpgt_epi32(abs_diff, error_vals);

    _mm_testz_si128(cmp, cmp) != 0
}

// 标准版本的RGB颜色比较
unsafe fn color_equal_rgb_standard(
    a: *const Rgba,
    b: *const Rgba,
    error_r: i32,
    error_g: i32,
    error_b: i32,
) -> bool {
    let a = &*a;
    let b = &*b;
    
    (a.r as i32 - b.r as i32).abs() <= error_r
        && (a.g as i32 - b.g as i32).abs() <= error_g
        && (a.b as i32 - b.b as i32).abs() <= error_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let features = CpuFeatures::detect();
        println!("AVX512: {}, AVX2: {}", features.has_avx512, features.has_avx2);
    }

    #[test]
    fn test_color_equality() {
        let color1 = Rgba {
            r: 100,
            g: 150,
            b: 200,
            a: 255,
        };
        let color2 = Rgba {
            r: 102,
            g: 153,
            b: 198,
            a: 255,
        };

        assert!(color_equal(&color1, &color2, 5));
        assert!(color_equal_rgb(&color1, &color2, 5, 5, 5));
    }

    #[test]
    fn test_color_inequality() {
        let color1 = Rgba {
            r: 100,
            g: 150,
            b: 200,
            a: 255,
        };
        let color2 = Rgba {
            r: 110,
            g: 160,
            b: 210,
            a: 255,
        };

        assert!(!color_equal(&color1, &color2, 5));
        assert!(!color_equal_rgb(&color1, &color2, 5, 5, 5));
    }
}