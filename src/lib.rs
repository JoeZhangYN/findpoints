extern crate libc;

use rayon::prelude::*;
use std;
use std::arch::x86_64::*;
use std::os::raw::c_double;

// 优化后的find_bytes_avx2函数
#[target_feature(enable = "avx2")]
unsafe fn find_bytes_avx2(
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
    let par_x = tup1.x; // 大图宽度
    let par_y = tup1.y; // 大图高度
    let sub_x = tup2.x; // 小图宽度
    let sub_y = tup2.y; // 小图高度

    // 检测并转换为u8数组
    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    // 预计算忽略色（优化：直接作为32位整数比较）
    let ignore_color_r = _mm256_set1_epi8(ignore_r as i8);
    let ignore_color_g = _mm256_set1_epi8(ignore_g as i8);
    let ignore_color_b = _mm256_set1_epi8(ignore_b as i8);

    // 预计算小图第一个像素（用于快速跳过）
    let first_pixel_r = numbers2[0];
    let first_pixel_g = numbers2[1];
    let first_pixel_b = numbers2[2];

    // 总像素数（预计算，避免重复计算）
    let total_pixels = (sub_x * sub_y) as usize;
    let min_match_pixels = (total_pixels as f64 * match_rate) as usize;

    // 创建并行迭代器，遍历大图
    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let start_par_index = (j * par_x * 4 + i * 4) as usize;

            // 快速检查：先验证第一个像素
            let (par_r, par_g, par_b) = (
                *numbers1.get_unchecked(start_par_index),
                *numbers1.get_unchecked(start_par_index + 1),
                *numbers1.get_unchecked(start_par_index + 2),
            );

            // 跳过忽略色
            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            // 第一个像素匹配检查（快速跳过不可能的位置）
            if par_r != first_pixel_r || par_g != first_pixel_g || par_b != first_pixel_b {
                continue;
            }

            // 开始详细匹配
            let mut matched_pixels = 0usize;
            let mut total_checked_pixels = 0usize;
            let mut early_termination = false;

            // 分块处理，支持早期终止
            'outer_loop: for block_j in (0..sub_y).step_by(4) {
                let actual_block_height = std::cmp::min(4, sub_y - block_j);
                
                for block_i in (0..sub_x).step_by(8) {
                    let actual_block_width = std::cmp::min(8, sub_x - block_i);
                    
                    // 处理当前8x4块
                    for local_j in 0..actual_block_height {
                        let row_j = block_j + local_j;
                        let sub_row_start = (row_j * sub_x * 4) as usize;
                        let par_row_start = ((j + row_j) * par_x * 4 + (i + block_i) * 4) as usize;
                        
                        // 计算这一行能处理多少个完整的8像素组
                        let pixels_remaining = actual_block_width;
                        let full_vectors = pixels_remaining / 8;
                        let remaining_pixels = pixels_remaining % 8;
                        
                        // 处理完整的8像素向量
                        for vec_idx in 0..full_vectors {
                            let sub_offset = sub_row_start + ((block_i + vec_idx * 8) * 4) as usize;
                            let par_offset = par_row_start + (vec_idx * 8 * 4) as usize;

                            // 加载32字节（8像素×4通道）
                            let sub_pixels = _mm256_loadu_si256(numbers2.as_ptr().add(sub_offset) as *const __m256i);
                            let par_pixels = _mm256_loadu_si256(numbers1.as_ptr().add(par_offset) as *const __m256i);

                            // 分离R、G、B通道进行比较
                            let sub_r = _mm256_and_si256(sub_pixels, _mm256_set1_epi32(0x000000FF));
                            let sub_g = _mm256_and_si256(_mm256_srli_epi32(sub_pixels, 8), _mm256_set1_epi32(0x000000FF));
                            let sub_b = _mm256_and_si256(_mm256_srli_epi32(sub_pixels, 16), _mm256_set1_epi32(0x000000FF));
                            
                            let par_r = _mm256_and_si256(par_pixels, _mm256_set1_epi32(0x000000FF));
                            let par_g = _mm256_and_si256(_mm256_srli_epi32(par_pixels, 8), _mm256_set1_epi32(0x000000FF));
                            let par_b = _mm256_and_si256(_mm256_srli_epi32(par_pixels, 16), _mm256_set1_epi32(0x000000FF));
                            
                            // 检查是否为忽略色
                            let ignore_mask_r = _mm256_cmpeq_epi8(par_r, ignore_color_r);
                            let ignore_mask_g = _mm256_cmpeq_epi8(par_g, ignore_color_g);
                            let ignore_mask_b = _mm256_cmpeq_epi8(par_b, ignore_color_b);
                            let ignore_mask = _mm256_and_si256(_mm256_and_si256(ignore_mask_r, ignore_mask_g), ignore_mask_b);
                            
                            // 比较RGB通道
                            let match_r = _mm256_cmpeq_epi8(sub_r, par_r);
                            let match_g = _mm256_cmpeq_epi8(sub_g, par_g);
                            let match_b = _mm256_cmpeq_epi8(sub_b, par_b);
                            let match_mask = _mm256_and_si256(_mm256_and_si256(match_r, match_g), match_b);
                            
                            // 排除忽略色的匹配
                            let valid_match_mask = _mm256_andnot_si256(ignore_mask, match_mask);
                            let valid_pixel_mask = _mm256_xor_si256(ignore_mask, _mm256_set1_epi8(-1));
                            
                            // 统计匹配的像素数（每个像素4字节，所以除以4）
                            let matched_bytes = _mm256_movemask_epi8(valid_match_mask).count_ones() as usize;
                            let total_bytes = _mm256_movemask_epi8(valid_pixel_mask).count_ones() as usize;
                            
                            matched_pixels += matched_bytes / 4;
                            total_checked_pixels += total_bytes / 4;
                        }
                        
                        // 处理剩余像素（不足8个的部分）
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
                            
                            // 跳过忽略色
                            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                                continue;
                            }
                            
                            total_checked_pixels += 1;
                            
                            if sub_r == par_r && sub_g == par_g && sub_b == par_b {
                                matched_pixels += 1;
                            }
                        }
                    }
                    
                    // 早期终止检查1：如果当前匹配率已经足够且检查了足够多的像素
                    if total_checked_pixels > total_pixels / 4 &&
                       matched_pixels * total_pixels >= min_match_pixels * total_checked_pixels {
                        // 当前匹配率已经满足要求，继续检查剩余部分以确认
                    }
                    
                    // 早期终止检查2：如果即使剩余像素全部匹配也无法达到阈值
                    let remaining_pixels = total_pixels - total_checked_pixels;
                    let max_possible_matches = matched_pixels + remaining_pixels;
                    if max_possible_matches < min_match_pixels {
                        early_termination = true;
                        break 'outer_loop;
                    }
                }
            }
            
            // 最终匹配率检查
            if !early_termination && total_checked_pixels > 0 {
                let final_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
                if final_match_rate >= match_rate {
                    return Some(Tuple { x: i, y: j });
                }
            }
        }
        None
    });

    result.unwrap_or(Tuple {
        x: 245760,
        y: 143640,
    })
}

#[no_mangle]
pub extern "C" fn FindBytesRust(
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
        find_bytes_avx2(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b,
        )
    }
}

#[no_mangle]
pub extern "C" fn FindBytesTolerance(
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
    tolerance: u8, // 新增的容差值
) -> Tuple {
    unsafe {
        find_bytes_tolerance(
            n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, tolerance,
        )
    }
}

unsafe fn find_bytes_tolerance_optimized(
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
    use rayon::prelude::*;

    let par_x = tup1.x; // 大图宽度
    let par_y = tup1.y; // 大图高度
    let sub_x = tup2.x; // 小图宽度
    let sub_y = tup2.y; // 小图高度

    // 检测并转换为u8数组
    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    // 预计算小图左上角颜色
    let first_pixel_r = numbers2[0];
    let first_pixel_g = numbers2[1];
    let first_pixel_b = numbers2[2];

    // 预计算总像素数和最小匹配像素数
    let total_pixels = (sub_x * sub_y) as usize;
    let min_match_pixels = (total_pixels as f64 * match_rate) as usize;
    
    // 预计算容差相关的值
    let tolerance_i32 = tolerance as i32;

    // 内联容差匹配函数，避免函数调用开销
    #[inline(always)]
    fn color_match_with_tolerance(
        c1_r: u8, c1_g: u8, c1_b: u8,
        c2_r: u8, c2_g: u8, c2_b: u8,
        tolerance: i32
    ) -> bool {
        (c1_r as i32 - c2_r as i32).abs() <= tolerance
            && (c1_g as i32 - c2_g as i32).abs() <= tolerance
            && (c1_b as i32 - c2_b as i32).abs() <= tolerance
    }

    // 创建并行迭代器，遍历大图
    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let start_par_index = (j * par_x * 4 + i * 4) as usize;

            // 获取大图当前位置的像素
            let par_r = *numbers1.get_unchecked(start_par_index);
            let par_g = *numbers1.get_unchecked(start_par_index + 1);
            let par_b = *numbers1.get_unchecked(start_par_index + 2);

            // 第1层过滤：跳过忽略色
            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            // 第2层过滤：检查第一个像素是否在容差范围内
            if !color_match_with_tolerance(
                par_r, par_g, par_b,
                first_pixel_r, first_pixel_g, first_pixel_b,
                tolerance_i32
            ) {
                continue;
            }

            // 开始详细匹配，使用分块处理和早期终止
            let mut matched_pixels = 0usize;
            let mut total_checked_pixels = 0usize;
            let mut early_termination = false;

            // 分块处理，每次处理一个4x4的块
            'outer_loop: for block_j in (0..sub_y).step_by(4) {
                let block_height = std::cmp::min(4, sub_y - block_j);
                
                for block_i in (0..sub_x).step_by(4) {
                    let block_width = std::cmp::min(4, sub_x - block_i);
                    
                    // 处理当前4x4块
                    for local_j in 0..block_height {
                        let current_j = block_j + local_j;
                        let sub_row_base = current_j * sub_x * 4;
                        let par_row_base = (j + current_j) * par_x * 4 + i * 4;

                        for local_i in 0..block_width {
                            let current_i = block_i + local_i;
                            let sub_index = sub_row_base + current_i * 4;
                            let par_index = par_row_base + current_i * 4;

                            // 获取像素值
                            let sub_r = *numbers2.get_unchecked(sub_index as usize);
                            let sub_g = *numbers2.get_unchecked(sub_index as usize + 1);
                            let sub_b = *numbers2.get_unchecked(sub_index as usize + 2);

                            let par_r = *numbers1.get_unchecked(par_index as usize);
                            let par_g = *numbers1.get_unchecked(par_index as usize + 1);
                            let par_b = *numbers1.get_unchecked(par_index as usize + 2);
                            
                            // 跳过忽略色
                            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                                continue;
                            }
                            
                            total_checked_pixels += 1;
                            
                            // 容差匹配检查
                            if color_match_with_tolerance(
                                par_r, par_g, par_b,
                                sub_r, sub_g, sub_b,
                                tolerance_i32
                            ) {
                                matched_pixels += 1;
                            }
                        }
                    }
                    
                    // 早期终止检查1：如果当前匹配率已经很高且检查了足够的像素
                    if total_checked_pixels >= total_pixels / 4 {
                        let current_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
                        if current_match_rate >= match_rate {
                            // 快速路径：当前匹配率已经满足，可以提前返回
                            // 但为了准确性，我们继续检查一小部分
                            if total_checked_pixels >= total_pixels / 2 {
                                return Some(Tuple { x: i, y: j });
                            }
                        }
                    }
                    
                    // 早期终止检查2：如果剩余像素全部匹配也无法达到阈值
                    let remaining_pixels = total_pixels - total_checked_pixels;
                    let max_possible_matches = matched_pixels + remaining_pixels;
                    if max_possible_matches < min_match_pixels {
                        early_termination = true;
                        break 'outer_loop;
                    }
                    
                    // 早期终止检查3：如果当前匹配率极低，提前退出
                    if total_checked_pixels >= total_pixels / 8 {
                        let current_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
                        if current_match_rate < match_rate * 0.5 {
                            early_termination = true;
                            break 'outer_loop;
                        }
                    }
                }
            }

            // 最终匹配率检查
            if !early_termination && total_checked_pixels > 0 {
                let final_match_rate = matched_pixels as f64 / total_checked_pixels as f64;
                if final_match_rate >= match_rate {
                    return Some(Tuple { x: i, y: j });
                }
            }
        }
        None
    });

    result.unwrap_or(Tuple {
        x: 245760,
        y: 143640,
    })
}

// 进一步优化：预计算版本（如果容差值不大，比如 <= 10）
unsafe fn find_bytes_tolerance(
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
    use rayon::prelude::*;
    
    // 如果容差很大，回退到标准版本
    if tolerance > 10 {
        return find_bytes_tolerance_optimized(
            n1, len1, tup1, n2, len2, tup2, 
            match_rate, ignore_r, ignore_g, ignore_b, tolerance
        );
    }

    let par_x = tup1.x;
    let par_y = tup1.y;
    let sub_x = tup2.x;
    let sub_y = tup2.y;

    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    // 预计算小图的容差范围哈希表（仅对于小容差值有效）
    let mut tolerance_table = vec![vec![false; 256]; 256];
    for i in 0..256 {
        for j in 0..256 {
            tolerance_table[i][j] = (i as i32 - j as i32).abs() <= tolerance as i32;
        }
    }

    let first_pixel_r = numbers2[0];
    let first_pixel_g = numbers2[1];
    let first_pixel_b = numbers2[2];
    let min_match_pixels = (sub_x * sub_y) as f64 * match_rate;

    // 内联查表匹配函数
    #[inline(always)]
    fn table_match(
        table: &[Vec<bool>],
        c1_r: u8, c1_g: u8, c1_b: u8,
        c2_r: u8, c2_g: u8, c2_b: u8,
    ) -> bool {
        table[c1_r as usize][c2_r as usize]
            && table[c1_g as usize][c2_g as usize]
            && table[c1_b as usize][c2_b as usize]
    }

    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let start_par_index = (j * par_x * 4 + i * 4) as usize;

            let par_r = *numbers1.get_unchecked(start_par_index);
            let par_g = *numbers1.get_unchecked(start_par_index + 1);
            let par_b = *numbers1.get_unchecked(start_par_index + 2);

            // 忽略色检查
            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            // 第一像素容差检查
            if !table_match(
                &tolerance_table,
                par_r, par_g, par_b,
                first_pixel_r, first_pixel_g, first_pixel_b,
            ) {
                continue;
            }

            // 快速匹配计数
            let mut matched_pixels = 0usize;
            let mut total_pixels = 0usize;

            // 使用展开的循环减少分支预测失败
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
                    
                    if table_match(&tolerance_table, par_r, par_g, par_b, sub_r, sub_g, sub_b) {
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

    result.unwrap_or(Tuple {
        x: 245760,
        y: 143640,
    })
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Tuple {
    pub x: u32,
    pub y: u32,
}

impl From<(u32, u32)> for Tuple {
    fn from(tup: (u32, u32)) -> Tuple {
        Tuple { x: tup.0, y: tup.1 }
    }
}

impl From<Tuple> for (u32, u32) {
    fn from(tup: Tuple) -> (u32, u32) {
        (tup.x, tup.y)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[no_mangle]
pub extern "C" fn rgba_new(r: u8, g: u8, b: u8, a: u8) -> Rgba {
    Rgba { r, g, b, a }
}

// 使用AVX2指令集优化的颜色比较函数
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

#[no_mangle]
pub extern "C" fn color_a_equal_color_b(
    color_a: *const Rgba,
    color_b: *const Rgba,
    error_range: u8,
) -> bool {
    unsafe { color_equal_avx2(color_a, color_b, error_range as i32) }
}

// 使用AVX2指令集优化的RGB颜色比较函数
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

#[no_mangle]
pub extern "C" fn color_a_equal_color_b_rgb(
    color_a: *const Rgba,
    color_b: *const Rgba,
    error_r: u8,
    error_g: u8,
    error_b: u8,
) -> bool {
    unsafe {
        color_equal_rgb_avx2(
            color_a,
            color_b,
            error_r as i32,
            error_g as i32,
            error_b as i32,
        )
    }
}


/// 添加基本的单元测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_equality() {
        let color1 = Rgba { r: 100, g: 150, b: 200, a: 255 };
        let color2 = Rgba { r: 102, g: 153, b: 198, a: 255 };
        
        unsafe {
            // 测试误差范围为5的情况
            assert!(color_a_equal_color_b(
                &color1,
                &color2,
                5
            ));
            
            // 测试RGB比较
            assert!(color_a_equal_color_b_rgb(
                &color1,
                &color2,
                5, 5, 5
            ));
        }
    }

    #[test]
    fn test_color_inequality() {
        let color1 = Rgba { r: 100, g: 150, b: 200, a: 255 };
        let color2 = Rgba { r: 110, g: 160, b: 210, a: 255 };
        
        unsafe {
            // 测试误差范围为5的情况
            assert!(!color_a_equal_color_b(
                &color1,
                &color2,
                5
            ));
            
            // 测试RGB比较
            assert!(!color_a_equal_color_b_rgb(
                &color1,
                &color2,
                5, 5, 5
            ));
        }
    }
}