extern crate libc;

use libc::size_t;
use std::convert::From;
use std::slice;

// A Rust function that accepts a tuple
#[no_mangle]
pub extern "C" fn FindBytesRust(
    n1: *const u8,
    len1: size_t,
    tup1: Tuple,
    n2: *const u8,
    len2: size_t,
    tup2: Tuple,
    match_rate: f64,
) -> (u32, u32) {
    let par_x = tup1.x; // 大图宽度
    let par_y = tup1.y; // 大图高度
    let sub_x = tup2.x; // 小图宽度
    let sub_y = tup2.y; // 小图高度
                        // // 调试输出
                        // println!("par_x:{}, sub_x:{}", par_x, sub_x);
                        // println!("par_y:{}, sub_y:{}", par_y, sub_y);

    // 检测是否为u8数组
    let numbers1 = unsafe {
        assert!(!n1.is_null());
        slice::from_raw_parts(n1, len1 as usize)
    };
    let numbers2 = unsafe {
        assert!(!n1.is_null());
        slice::from_raw_parts(n2, len2 as usize)
    };

    // 循环从大图0到大图宽度-小图宽度，0到大图高度-小图高度
    // i 代表每个循环的X
    for i in 0..par_x - sub_x {
        // j 代表每个循环的Y
        for j in 0..par_y - sub_y {
            // 用于计算当前实际位置 当前y值乘以宽度乘以4，加上当前宽度乘以4 返回实际位置 255,0,0,0
            let par_index = (j * par_x * 4 + i * 4) as usize;
            // 当前RGB 不等于 初始RGB 则继续循环
            if numbers1[par_index + 3] != numbers2[3]
                || numbers1[par_index + 2] != numbers2[2]
                || numbers1[par_index + 1] != numbers2[1]
            {
                continue;
            }
            // 执行到此块，说明当前第一个点匹配
            // 用于计数对比点的总数
            let mut sum: f64 = 0.0;
            // 对于计数匹配的点
            let mut match_num: f64 = 0.0;

            // i1 代表每个循环的X
            for i1 in 0..sub_x {
                // j1 代表每个循环的Y
                for j1 in 0..sub_y {
                    // 用于计算小图当前实际位置 当前y值乘以宽度乘以4，加上当前宽度乘以4 返回实际位置 255,0,0,0
                    let sub_index = (j1 * sub_x * 4 + i1 * 4) as usize;
                    // 用于计算大图匹配小图当前实际位置 当前y值乘以宽度乘以4，加上当前宽度乘以4 返回实际位置 255,0,0,0
                    let par_index1 = ((j + j1) * par_x * 4 + (i + i1) * 4) as usize;
                    // 计数+1
                    sum += 1.0;
                    // 对比
                    if numbers1[par_index1 + 3] == numbers2[sub_index + 3]
                        && numbers1[par_index1 + 2] == numbers2[sub_index + 2]
                        && numbers1[par_index1 + 1] == numbers2[sub_index + 1]
                    {
                        // 如果匹配匹配数+1
                        match_num += 1.0;
                    }
                }
            }
            // 匹配率到达后返回对应坐标
            if match_num / sum >= match_rate {
                println!("i:{}, j:{}", i, j);
                drop(numbers1);
                drop(numbers2);
                drop(n1);
                drop(n2);
                return (i, j);
            }
        }
    }
    drop(numbers1);
    drop(numbers2);
    drop(n1);
    drop(n2);
    (0, 0)
}

// A struct that can be passed between C and Rust
#[repr(C)]
pub struct Tuple {
    x: u32,
    y: u32,
}

// Conversion functions
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
