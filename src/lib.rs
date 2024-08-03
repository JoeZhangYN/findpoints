extern crate libc;

use std;

#[target_feature(enable = "avx2")]
unsafe fn find_bytes_avx2(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: f64,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
) -> (u32, u32) {
    use std::arch::x86_64::*;
    use rayon::prelude::*;

    let par_x = tup1.x; // 大图宽度
    let par_y = tup1.y; // 大图高度
    let sub_x = tup2.x; // 小图宽度
    let sub_y = tup2.y; // 小图高度

    // 检测并转换为u8数组
    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    // 获取小图左上角颜色
    let (sub_r, sub_g, sub_b) = (numbers2[0], numbers2[1], numbers2[2]);

    // 创建一个并行迭代器，遍历大图
    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let par_index = (j * par_x * 4 + i * 4) as usize;

            let (par_r, par_g, par_b) = (
                numbers1[par_index],
                numbers1[par_index + 1],
                numbers1[par_index + 2],
            );

            // 跳过忽略色
            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            // 检查大图当前位置与小图左上角颜色是否匹配
            if par_r == sub_r && par_g == sub_g && par_b == sub_b {
                let mut sum: f64 = 0.0;
                let mut match_num: f64 = 0.0;

                // 使用AVX进行内层循环匹配
                for i1 in (0..sub_x).step_by(8) {
                    for j1 in 0..sub_y {
                        let sub_index = (j1 * sub_x * 4 + i1 * 4) as usize;
                        let par_index = ((j + j1) * par_x * 4 + (i + i1) * 4) as usize;

                        if i1 + 8 <= sub_x {  // 确保处理8个像素块时不会越界
                            // 加载小图和大图对应部分的8个像素
                            let sub_pixels = _mm256_loadu_si256(numbers2[sub_index..].as_ptr() as *const __m256i);
                            let par_pixels = _mm256_loadu_si256(numbers1[par_index..].as_ptr() as *const __m256i);

                            // 忽略特定颜色的掩码
                            let ignore_mask = _mm256_set1_epi32(((ignore_r as i32) << 16) | ((ignore_g as i32) << 8) | (ignore_b as i32));

                            // 比较像素并创建匹配像素的掩码
                            let match_mask = _mm256_cmpeq_epi8(sub_pixels, par_pixels);
                            let ignore_masked = _mm256_cmpeq_epi8(par_pixels, ignore_mask);

                            // 创建一个全1的掩码用于取反
                            let not_mask = _mm256_set1_epi8(-1); // 等价于0xFF

                            // 统计匹配数
                            let match_count = _mm256_movemask_epi8(_mm256_and_si256(match_mask, _mm256_xor_si256(ignore_masked, not_mask)));
                            let sum_count = _mm256_movemask_epi8(_mm256_xor_si256(ignore_masked, not_mask));

                            sum += sum_count.count_ones() as f64;
                            match_num += match_count.count_ones() as f64;
                        } else {
                            // 对于不足8像素宽度的处理
                            for k in 0..(sub_x % 8) {
                                let sub_pixel = &numbers2[sub_index + k as usize * 4..sub_index + (k as usize + 1) * 4];
                                let par_pixel = &numbers1[par_index + k as usize * 4..par_index + (k as usize + 1) * 4];
                                if par_pixel[0] == ignore_r && par_pixel[1] == ignore_g && par_pixel[2] == ignore_b {
                                    continue;
                                }

                                sum += 1.0;
                                if sub_pixel == par_pixel {
                                    match_num += 1.0;
                                }
                            }
                        }
                    }
                }

                // 检查匹配率
                if sum > 0.0 && (match_num / sum) >= match_rate {
                    return Some((i, j));
                }
            }
        }
        None
    });

    result.unwrap_or((245760, 143640))
}

#[no_mangle]
pub extern "C" fn FindBytesRust(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: f64,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
) -> (u32, u32) {
    unsafe {
        find_bytes_avx2(n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b)
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
    match_rate: f64,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    tolerance: u8, // 新增的容差值
) -> (u32, u32) {
    unsafe {
        find_bytes_tolerance(n1, len1, tup1, n2, len2, tup2, match_rate, ignore_r, ignore_g, ignore_b, tolerance)
    }
}

unsafe fn find_bytes_tolerance(
    n1: *const u8,
    len1: usize,
    tup1: Tuple,
    n2: *const u8,
    len2: usize,
    tup2: Tuple,
    match_rate: f64,
    ignore_r: u8,
    ignore_g: u8,
    ignore_b: u8,
    tolerance: u8, // 新增的容差值
) -> (u32, u32) {
    use rayon::prelude::*;

    let par_x = tup1.x; // 大图宽度
    let par_y = tup1.y; // 大图高度
    let sub_x = tup2.x; // 小图宽度
    let sub_y = tup2.y; // 小图高度

    // 检测并转换为u8数组
    let numbers1 = std::slice::from_raw_parts(n1, len1);
    let numbers2 = std::slice::from_raw_parts(n2, len2);

    // 获取小图左上角颜色
    let (sub_r, sub_g, sub_b) = (numbers2[0], numbers2[1], numbers2[2]);

    // 创建一个并行迭代器，遍历大图
    let result = (0..=par_x - sub_x).into_par_iter().find_map_any(|i| {
        for j in 0..=par_y - sub_y {
            let par_index = (j * par_x * 4 + i * 4) as usize;

            let (par_r, par_g, par_b) = (
                *numbers1.get(par_index).unwrap(),
                *numbers1.get(par_index + 1).unwrap(),
                *numbers1.get(par_index + 2).unwrap(),
            );

            // 跳过忽略色
            if par_r == ignore_r && par_g == ignore_g && par_b == ignore_b {
                continue;
            }

            // 检查大图当前位置与小图左上角颜色是否匹配（带容差）
            if (par_r as i32 - sub_r as i32).abs() <= tolerance as i32 &&
               (par_g as i32 - sub_g as i32).abs() <= tolerance as i32 &&
               (par_b as i32 - sub_b as i32).abs() <= tolerance as i32 {

                let mut sum: f64 = 0.0;
                let mut match_num: f64 = 0.0;

                // 单纯循环匹配
                for i1 in 0..sub_x {
                    for j1 in 0..sub_y {
                        let sub_index = (j1 * sub_x * 4 + i1 * 4) as usize;
                        let par_index = ((j + j1) * par_x * 4 + (i + i1) * 4) as usize;

                        let sub_pixel_r = *numbers2.get(sub_index).unwrap() as i32;
                        let sub_pixel_g = *numbers2.get(sub_index + 1).unwrap() as i32;
                        let sub_pixel_b = *numbers2.get(sub_index + 2).unwrap() as i32;
                        let par_pixel_r = *numbers1.get(par_index).unwrap() as i32;
                        let par_pixel_g = *numbers1.get(par_index + 1).unwrap() as i32;
                        let par_pixel_b = *numbers1.get(par_index + 2).unwrap() as i32;

                        if par_pixel_r == ignore_r as i32 && par_pixel_g == ignore_g as i32 && par_pixel_b == ignore_b as i32 {
                            continue;
                        }

                        sum += 1.0;
                        if (par_pixel_r - sub_pixel_r).abs() <= tolerance as i32 &&
                           (par_pixel_g - sub_pixel_g).abs() <= tolerance as i32 &&
                           (par_pixel_b - sub_pixel_b).abs() <= tolerance as i32 {
                            match_num += 1.0;
                        }
                    }
                }

                // 检查匹配率
                if sum > 0.0 && (match_num / sum) >= match_rate {
                    return Some((i, j));
                }
            }
        }
        None
    });

    result.unwrap_or((245760, 143640))
}


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Tuple {
    x: u32,
    y: u32,
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

#[target_feature(enable = "avx2")]
unsafe fn color_equal_avx2(a: *const Rgba, b: *const Rgba, error: i32) -> bool {
    use std::arch::x86_64::*;
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
    unsafe {
        color_equal_avx2(color_a, color_b, error_range as i32)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn color_equal_rgb_avx2(a: *const Rgba, b: *const Rgba, error_r: i32, error_g: i32, error_b: i32) -> bool {
    use std::arch::x86_64::*;
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
        color_equal_rgb_avx2(color_a, color_b, error_r as i32, error_g as i32, error_b as i32)
    }
}