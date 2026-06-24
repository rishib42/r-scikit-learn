use crate::error::CoreError;
use matrixmultiply::dgemm;
use rayon::prelude::*;
#[cfg(target_os = "macos")]
use std::os::raw::c_int;
use wide::f64x4;

const KNEIGHBORS_QUERY_BLOCK_ROWS: usize = 64;
const PREDICT_QUERY_BLOCK_ROWS: usize = 128;
const KNEIGHBORS_TRAIN_BLOCK_ROWS: usize = 1024;
const PREDICT_TRAIN_BLOCK_ROWS: usize = 1024;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_dgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
}

#[derive(Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
}

#[derive(Clone, Copy)]
pub enum WeightMode {
    Uniform,
    Distance,
}

#[inline(always)]
fn euclidean_squared_pruned(
    query: &[f64],
    train: &[f64],
    query_offset: usize,
    train_offset: usize,
    columns: usize,
    threshold: f64,
) -> f64 {
    let mut total = 0.0;
    for column in 0..columns {
        let diff = query[query_offset + column] - train[train_offset + column];
        total += diff * diff;
        if total > threshold {
            break;
        }
    }
    total
}

#[inline(always)]
fn euclidean_squared_simd(
    query: &[f64],
    train: &[f64],
    query_offset: usize,
    train_offset: usize,
    columns: usize,
) -> f64 {
    let mut vector_total = f64x4::ZERO;
    let vector_columns = columns / 4 * 4;
    let mut column = 0;
    while column < vector_columns {
        let query_values = f64x4::new([
            query[query_offset + column],
            query[query_offset + column + 1],
            query[query_offset + column + 2],
            query[query_offset + column + 3],
        ]);
        let train_values = f64x4::new([
            train[train_offset + column],
            train[train_offset + column + 1],
            train[train_offset + column + 2],
            train[train_offset + column + 3],
        ]);
        let diff = query_values - train_values;
        vector_total += diff * diff;
        column += 4;
    }
    let mut total = vector_total.reduce_add();
    while column < columns {
        let diff = query[query_offset + column] - train[train_offset + column];
        total += diff * diff;
        column += 1;
    }
    total
}

#[inline(always)]
fn manhattan_distance(
    query: &[f64],
    train: &[f64],
    query_offset: usize,
    train_offset: usize,
    columns: usize,
) -> f64 {
    let mut total = 0.0;
    for column in 0..columns {
        total += (query[query_offset + column] - train[train_offset + column]).abs();
    }
    total
}

fn should_insert(
    distance: f64,
    index: usize,
    best_distances: &[f64],
    best_indices: &[usize],
) -> bool {
    let last = best_distances.len() - 1;
    distance < best_distances[last]
        || (distance == best_distances[last] && index < best_indices[last])
}

fn insert_neighbor(
    distance: f64,
    index: usize,
    best_distances: &mut [f64],
    best_indices: &mut [usize],
) {
    let mut position = best_distances.len() - 1;
    while position > 0
        && (distance < best_distances[position - 1]
            || (distance == best_distances[position - 1] && index < best_indices[position - 1]))
    {
        best_distances[position] = best_distances[position - 1];
        best_indices[position] = best_indices[position - 1];
        position -= 1;
    }
    best_distances[position] = distance;
    best_indices[position] = index;
}

fn should_replace_worst(
    distance: f64,
    index: usize,
    worst_position: usize,
    best_distances: &[f64],
    best_indices: &[usize],
) -> bool {
    distance < best_distances[worst_position]
        || (distance == best_distances[worst_position] && index < best_indices[worst_position])
}

fn worst_neighbor_position(best_distances: &[f64], best_indices: &[usize]) -> usize {
    let mut worst_position = 0;
    for position in 1..best_distances.len() {
        if best_distances[position] > best_distances[worst_position]
            || (best_distances[position] == best_distances[worst_position]
                && best_indices[position] > best_indices[worst_position])
        {
            worst_position = position;
        }
    }
    worst_position
}

fn replace_worst_neighbor(
    distance: f64,
    index: usize,
    worst_position: &mut usize,
    best_distances: &mut [f64],
    best_indices: &mut [usize],
) {
    best_distances[*worst_position] = distance;
    best_indices[*worst_position] = index;
    *worst_position = worst_neighbor_position(best_distances, best_indices);
}

pub fn row_norms(input: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let offset = row * columns;
            let mut total = 0.0;
            for column in 0..columns {
                let value = input[offset + column];
                total += value * value;
            }
            total
        })
        .collect()
}

fn row_norms_serial(input: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut output = vec![0.0; rows];
    for (row, value) in output.iter_mut().enumerate() {
        let offset = row * columns;
        let mut total = 0.0;
        for column in 0..columns {
            let value = input[offset + column];
            total += value * value;
        }
        *value = total;
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn fill_pairwise_dot_block_into(
    query: &[f64],
    train_transposed: &[f64],
    block_rows: usize,
    train_rows: usize,
    train_start: usize,
    train_block_rows: usize,
    columns: usize,
    dots: &mut [f64],
) {
    // SAFETY: query, train, and dots are valid contiguous f64 buffers sized for
    // the provided matrix dimensions. The output buffer is distinct from both
    // inputs, and the strides describe non-overlapping row-major matrices:
    // query is (block_rows x columns), train_transposed is a contiguous
    // (columns x train_rows) matrix, and dots is row-major
    // (block_rows x train_block_rows).
    debug_assert!(dots.len() >= block_rows * train_block_rows);
    debug_assert!(train_start + train_block_rows <= train_rows);
    #[cfg(target_os = "macos")]
    if let (Ok(block_rows), Ok(train_rows), Ok(train_block_rows), Ok(columns)) = (
        c_int::try_from(block_rows),
        c_int::try_from(train_rows),
        c_int::try_from(train_block_rows),
        c_int::try_from(columns),
    ) {
        unsafe {
            const CBLAS_ROW_MAJOR: c_int = 101;
            const CBLAS_NO_TRANS: c_int = 111;
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                block_rows,
                train_block_rows,
                columns,
                1.0,
                query.as_ptr(),
                columns,
                train_transposed.as_ptr().add(train_start),
                train_rows,
                0.0,
                dots.as_mut_ptr(),
                train_block_rows,
            );
        }
        return;
    }
    unsafe {
        dgemm(
            block_rows,
            columns,
            train_block_rows,
            1.0,
            query.as_ptr(),
            columns as isize,
            1,
            train_transposed.as_ptr().add(train_start),
            train_rows as isize,
            1,
            0.0,
            dots.as_mut_ptr(),
            train_block_rows as isize,
            1,
        );
    }
}

#[inline(always)]
fn euclidean_squared_from_dot(query_norm: f64, train_norm: f64, dot: f64) -> f64 {
    (-2.0_f64).mul_add(dot, query_norm + train_norm).max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn euclidean_kneighbors_blocked(
    query: &[f64],
    _train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    exclude_self: bool,
) -> Result<(Vec<f64>, Vec<i64>), CoreError> {
    if train_transposed.len() != train_rows * columns || train_norms.len() != train_rows {
        return Err(CoreError::ShapeMismatch);
    }
    let mut distances = vec![0.0; query_rows * k];
    let mut indices = vec![0_i64; query_rows * k];
    distances
        .par_chunks_mut(KNEIGHBORS_QUERY_BLOCK_ROWS * k)
        .zip(indices.par_chunks_mut(KNEIGHBORS_QUERY_BLOCK_ROWS * k))
        .enumerate()
        .for_each(|(block, (distances_chunk, indices_chunk))| {
            let query_start = block * KNEIGHBORS_QUERY_BLOCK_ROWS;
            let block_rows = (query_rows - query_start).min(KNEIGHBORS_QUERY_BLOCK_ROWS);
            let query_slice = &query[query_start * columns..(query_start + block_rows) * columns];
            let query_norms = row_norms_serial(query_slice, block_rows, columns);
            let mut best_distances = vec![f64::INFINITY; block_rows * k];
            let mut best_indices = vec![usize::MAX; block_rows * k];
            let mut dots = vec![0.0; block_rows * KNEIGHBORS_TRAIN_BLOCK_ROWS];
            for train_start in (0..train_rows).step_by(KNEIGHBORS_TRAIN_BLOCK_ROWS) {
                let train_block_rows = (train_rows - train_start).min(KNEIGHBORS_TRAIN_BLOCK_ROWS);
                let dots = &mut dots[..block_rows * train_block_rows];
                fill_pairwise_dot_block_into(
                    query_slice,
                    train_transposed,
                    block_rows,
                    train_rows,
                    train_start,
                    train_block_rows,
                    columns,
                    dots,
                );
                for query_row in 0..block_rows {
                    let global_query_row = query_start + query_row;
                    let row_offset = query_row * k;
                    let row_distances = &mut best_distances[row_offset..row_offset + k];
                    let row_indices = &mut best_indices[row_offset..row_offset + k];
                    for train_row in 0..train_block_rows {
                        let global_train_row = train_start + train_row;
                        if exclude_self && global_query_row == global_train_row {
                            continue;
                        }
                        let distance = euclidean_squared_from_dot(
                            query_norms[query_row],
                            train_norms[global_train_row],
                            dots[query_row * train_block_rows + train_row],
                        );
                        if should_insert(distance, global_train_row, row_distances, row_indices) {
                            insert_neighbor(distance, global_train_row, row_distances, row_indices);
                        }
                    }
                }
            }
            for query_row in 0..block_rows {
                let row_offset = query_row * k;
                let row_distances = &mut best_distances[row_offset..row_offset + k];
                let row_indices = &mut best_indices[row_offset..row_offset + k];
                for neighbor in 0..k {
                    distances_chunk[row_offset + neighbor] = row_distances[neighbor].sqrt();
                    indices_chunk[row_offset + neighbor] = row_indices[neighbor] as i64;
                }
            }
        });
    Ok((distances, indices))
}

#[allow(clippy::too_many_arguments)]
fn euclidean_uniform_votes_blocked(
    query: &[f64],
    _train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    labels: &[i64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    classes: usize,
) -> Result<Vec<f64>, CoreError> {
    if train_transposed.len() != train_rows * columns || train_norms.len() != train_rows {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = vec![0.0; query_rows * classes];
    output
        .par_chunks_mut(PREDICT_QUERY_BLOCK_ROWS * classes)
        .enumerate()
        .for_each(|(block, output_chunk)| {
            let query_start = block * PREDICT_QUERY_BLOCK_ROWS;
            let block_rows = (query_rows - query_start).min(PREDICT_QUERY_BLOCK_ROWS);
            let query_slice = &query[query_start * columns..(query_start + block_rows) * columns];
            let query_norms = row_norms_serial(query_slice, block_rows, columns);
            let mut best_distances = vec![f64::INFINITY; block_rows * k];
            let mut best_indices = vec![usize::MAX; block_rows * k];
            let mut dots = vec![0.0; block_rows * PREDICT_TRAIN_BLOCK_ROWS];
            for train_start in (0..train_rows).step_by(PREDICT_TRAIN_BLOCK_ROWS) {
                let train_block_rows = (train_rows - train_start).min(PREDICT_TRAIN_BLOCK_ROWS);
                let dots = &mut dots[..block_rows * train_block_rows];
                fill_pairwise_dot_block_into(
                    query_slice,
                    train_transposed,
                    block_rows,
                    train_rows,
                    train_start,
                    train_block_rows,
                    columns,
                    dots,
                );
                for query_row in 0..block_rows {
                    let row_offset = query_row * k;
                    let row_distances = &mut best_distances[row_offset..row_offset + k];
                    let row_indices = &mut best_indices[row_offset..row_offset + k];
                    for train_row in 0..train_block_rows {
                        let global_train_row = train_start + train_row;
                        let distance = euclidean_squared_from_dot(
                            query_norms[query_row],
                            train_norms[global_train_row],
                            dots[query_row * train_block_rows + train_row],
                        );
                        if should_insert(distance, global_train_row, row_distances, row_indices) {
                            insert_neighbor(distance, global_train_row, row_distances, row_indices);
                        }
                    }
                }
            }
            let inverse_k = 1.0 / k as f64;
            for query_row in 0..block_rows {
                let neighbor_offset = query_row * k;
                let probability_offset = query_row * classes;
                for &neighbor in &best_indices[neighbor_offset..neighbor_offset + k] {
                    let label = labels[neighbor];
                    if label >= 0 && (label as usize) < classes {
                        output_chunk[probability_offset + label as usize] += inverse_k;
                    }
                }
            }
        });
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn euclidean_uniform_regression_blocked(
    query: &[f64],
    train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    targets: &[f64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    outputs: usize,
) -> Result<Vec<f64>, CoreError> {
    if train_transposed.len() != train_rows * columns
        || train_norms.len() != train_rows
        || targets.len() != train_rows * outputs
    {
        return Err(CoreError::ShapeMismatch);
    }
    if outputs == 1 {
        return euclidean_uniform_regression_single_output_blocked(
            query,
            train,
            train_transposed,
            train_norms,
            targets,
            query_rows,
            train_rows,
            columns,
            k,
        );
    }
    let mut output = vec![0.0; query_rows * outputs];
    output
        .par_chunks_mut(PREDICT_QUERY_BLOCK_ROWS * outputs)
        .enumerate()
        .for_each(|(block, output_chunk)| {
            let query_start = block * PREDICT_QUERY_BLOCK_ROWS;
            let block_rows = (query_rows - query_start).min(PREDICT_QUERY_BLOCK_ROWS);
            let query_slice = &query[query_start * columns..(query_start + block_rows) * columns];
            let query_norms = row_norms_serial(query_slice, block_rows, columns);
            let mut best_distances = vec![f64::INFINITY; block_rows * k];
            let mut best_indices = vec![usize::MAX; block_rows * k];
            let mut dots = vec![0.0; block_rows * PREDICT_TRAIN_BLOCK_ROWS];
            for train_start in (0..train_rows).step_by(PREDICT_TRAIN_BLOCK_ROWS) {
                let train_block_rows = (train_rows - train_start).min(PREDICT_TRAIN_BLOCK_ROWS);
                let dots = &mut dots[..block_rows * train_block_rows];
                fill_pairwise_dot_block_into(
                    query_slice,
                    train_transposed,
                    block_rows,
                    train_rows,
                    train_start,
                    train_block_rows,
                    columns,
                    dots,
                );
                for query_row in 0..block_rows {
                    let row_offset = query_row * k;
                    let row_distances = &mut best_distances[row_offset..row_offset + k];
                    let row_indices = &mut best_indices[row_offset..row_offset + k];
                    for train_row in 0..train_block_rows {
                        let global_train_row = train_start + train_row;
                        let distance = euclidean_squared_from_dot(
                            query_norms[query_row],
                            train_norms[global_train_row],
                            dots[query_row * train_block_rows + train_row],
                        );
                        if should_insert(distance, global_train_row, row_distances, row_indices) {
                            insert_neighbor(distance, global_train_row, row_distances, row_indices);
                        }
                    }
                }
            }
            let inverse_k = 1.0 / k as f64;
            for query_row in 0..block_rows {
                let neighbor_offset = query_row * k;
                let output_offset = query_row * outputs;
                for &neighbor in &best_indices[neighbor_offset..neighbor_offset + k] {
                    let target_offset = neighbor * outputs;
                    for target in 0..outputs {
                        output_chunk[output_offset + target] +=
                            targets[target_offset + target] * inverse_k;
                    }
                }
            }
        });
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn euclidean_uniform_regression_single_output_blocked(
    query: &[f64],
    _train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    targets: &[f64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
) -> Result<Vec<f64>, CoreError> {
    if train_transposed.len() != train_rows * columns
        || train_norms.len() != train_rows
        || targets.len() != train_rows
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = vec![0.0; query_rows];
    output
        .par_chunks_mut(PREDICT_QUERY_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block, output_chunk)| {
            let query_start = block * PREDICT_QUERY_BLOCK_ROWS;
            let block_rows = (query_rows - query_start).min(PREDICT_QUERY_BLOCK_ROWS);
            let query_slice = &query[query_start * columns..(query_start + block_rows) * columns];
            let query_norms = row_norms_serial(query_slice, block_rows, columns);
            let mut best_distances = vec![f64::INFINITY; block_rows * k];
            let mut best_indices = vec![usize::MAX; block_rows * k];
            let mut dots = vec![0.0; block_rows * PREDICT_TRAIN_BLOCK_ROWS];
            for train_start in (0..train_rows).step_by(PREDICT_TRAIN_BLOCK_ROWS) {
                let train_block_rows = (train_rows - train_start).min(PREDICT_TRAIN_BLOCK_ROWS);
                let dots = &mut dots[..block_rows * train_block_rows];
                fill_pairwise_dot_block_into(
                    query_slice,
                    train_transposed,
                    block_rows,
                    train_rows,
                    train_start,
                    train_block_rows,
                    columns,
                    dots,
                );
                for query_row in 0..block_rows {
                    let row_offset = query_row * k;
                    let row_distances = &mut best_distances[row_offset..row_offset + k];
                    let row_indices = &mut best_indices[row_offset..row_offset + k];
                    for train_row in 0..train_block_rows {
                        let global_train_row = train_start + train_row;
                        let distance = euclidean_squared_from_dot(
                            query_norms[query_row],
                            train_norms[global_train_row],
                            dots[query_row * train_block_rows + train_row],
                        );
                        if should_insert(distance, global_train_row, row_distances, row_indices) {
                            insert_neighbor(distance, global_train_row, row_distances, row_indices);
                        }
                    }
                }
            }
            let inverse_k = 1.0 / k as f64;
            for (query_row, prediction_output) in output_chunk.iter_mut().enumerate() {
                let neighbor_offset = query_row * k;
                let mut prediction = 0.0;
                for &neighbor in &best_indices[neighbor_offset..neighbor_offset + k] {
                    prediction += targets[neighbor] * inverse_k;
                }
                *prediction_output = prediction;
            }
        });
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn row_neighbor_indices_euclidean(
    query: &[f64],
    train: &[f64],
    row: usize,
    train_rows: usize,
    columns: usize,
    exclude_self: bool,
    best_distances: &mut [f64],
    best_indices: &mut [usize],
) {
    let query_offset = row * columns;
    for train_row in 0..train_rows {
        if exclude_self && train_row == row {
            continue;
        }
        let train_offset = train_row * columns;
        let threshold = best_distances[best_distances.len() - 1];
        let candidate =
            euclidean_squared_pruned(query, train, query_offset, train_offset, columns, threshold);
        if should_insert(candidate, train_row, best_distances, best_indices) {
            insert_neighbor(candidate, train_row, best_distances, best_indices);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn row_neighbor_indices_for_vote(
    query: &[f64],
    train: &[f64],
    row: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    metric: DistanceMetric,
) -> (Vec<f64>, Vec<usize>) {
    let query_offset = row * columns;
    let mut best_distances = vec![f64::INFINITY; k];
    let mut best_indices = vec![usize::MAX; k];
    let mut worst_position = 0;
    match metric {
        DistanceMetric::Euclidean => {
            for train_row in 0..train_rows {
                let train_offset = train_row * columns;
                let distance =
                    euclidean_squared_simd(query, train, query_offset, train_offset, columns);
                if should_replace_worst(
                    distance,
                    train_row,
                    worst_position,
                    &best_distances,
                    &best_indices,
                ) {
                    replace_worst_neighbor(
                        distance,
                        train_row,
                        &mut worst_position,
                        &mut best_distances,
                        &mut best_indices,
                    );
                }
            }
        }
        DistanceMetric::Manhattan => {
            for train_row in 0..train_rows {
                let train_offset = train_row * columns;
                let distance =
                    manhattan_distance(query, train, query_offset, train_offset, columns);
                if should_replace_worst(
                    distance,
                    train_row,
                    worst_position,
                    &best_distances,
                    &best_indices,
                ) {
                    replace_worst_neighbor(
                        distance,
                        train_row,
                        &mut worst_position,
                        &mut best_distances,
                        &mut best_indices,
                    );
                }
            }
        }
    }
    (best_distances, best_indices)
}

#[allow(clippy::too_many_arguments)]
fn row_neighbor_indices(
    query: &[f64],
    train: &[f64],
    row: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    metric: DistanceMetric,
    exclude_self: bool,
) -> (Vec<f64>, Vec<usize>) {
    let query_offset = row * columns;
    let mut best_distances = vec![f64::INFINITY; k];
    let mut best_indices = vec![usize::MAX; k];
    match metric {
        DistanceMetric::Euclidean => {
            row_neighbor_indices_euclidean(
                query,
                train,
                row,
                train_rows,
                columns,
                exclude_self,
                &mut best_distances,
                &mut best_indices,
            );
        }
        DistanceMetric::Manhattan => {
            for train_row in 0..train_rows {
                if exclude_self && train_row == row {
                    continue;
                }
                let train_offset = train_row * columns;
                let candidate =
                    manhattan_distance(query, train, query_offset, train_offset, columns);
                if should_insert(candidate, train_row, &best_distances, &best_indices) {
                    insert_neighbor(candidate, train_row, &mut best_distances, &mut best_indices);
                }
            }
        }
    }
    (best_distances, best_indices)
}

fn output_distance(distance: f64, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => distance.sqrt(),
        DistanceMetric::Manhattan => distance,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kneighbors(
    query: &[f64],
    train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    metric: DistanceMetric,
    exclude_self: bool,
) -> Result<(Vec<f64>, Vec<i64>), CoreError> {
    if query_rows == 0 || train_rows == 0 || columns == 0 {
        return Err(CoreError::EmptyInput);
    }
    if query.len() != query_rows * columns || train.len() != train_rows * columns {
        return Err(CoreError::ShapeMismatch);
    }
    if k == 0 || k > train_rows.saturating_sub(usize::from(exclude_self)) {
        return Err(CoreError::ShapeMismatch);
    }
    if matches!(metric, DistanceMetric::Euclidean) && !exclude_self {
        return euclidean_kneighbors_blocked(
            query,
            train,
            train_transposed,
            train_norms,
            query_rows,
            train_rows,
            columns,
            k,
            exclude_self,
        );
    }
    let mut distances = vec![0.0; query_rows * k];
    let mut indices = vec![0_i64; query_rows * k];
    distances
        .par_chunks_mut(k)
        .zip(indices.par_chunks_mut(k))
        .enumerate()
        .for_each(|(row, (row_distances, row_indices))| {
            let (best_distances, best_indices) = row_neighbor_indices(
                query,
                train,
                row,
                train_rows,
                columns,
                k,
                metric,
                exclude_self,
            );
            for neighbor in 0..k {
                row_distances[neighbor] = output_distance(best_distances[neighbor], metric);
                row_indices[neighbor] = best_indices[neighbor] as i64;
            }
        });
    Ok((distances, indices))
}

#[allow(clippy::too_many_arguments)]
fn row_probability(
    query: &[f64],
    train: &[f64],
    labels: &[i64],
    row: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    classes: usize,
    metric: DistanceMetric,
    weight_mode: WeightMode,
) -> Result<Vec<f64>, CoreError> {
    let (distances, indices) =
        row_neighbor_indices_for_vote(query, train, row, train_rows, columns, k, metric);
    let output_distances: Vec<f64> = distances
        .into_iter()
        .map(|value| output_distance(value, metric))
        .collect();
    let output_indices: Vec<i64> = indices.into_iter().map(|index| index as i64).collect();
    let mut output = vec![0.0; classes];
    vote_neighbors(
        labels,
        &output_distances,
        &output_indices,
        classes,
        weight_mode,
        &mut output,
    )?;
    Ok(output)
}

fn vote_uniform(
    labels: &[i64],
    indices: &[usize],
    classes: usize,
    output: &mut [f64],
) -> Result<(), CoreError> {
    output.fill(0.0);
    for &index in indices {
        let label = labels[index];
        if label < 0 || label as usize >= classes {
            return Err(CoreError::InvalidCode(label));
        }
        output[label as usize] += 1.0;
    }
    Ok(())
}

fn best_voted_class(votes: &[f64]) -> i64 {
    let mut best_class = 0_i64;
    let mut best_vote = votes[0];
    for (class, &vote) in votes.iter().enumerate().skip(1) {
        if vote > best_vote {
            best_vote = vote;
            best_class = class as i64;
        }
    }
    best_class
}

fn vote_neighbors(
    labels: &[i64],
    distances: &[f64],
    indices: &[i64],
    classes: usize,
    weight_mode: WeightMode,
    output: &mut [f64],
) -> Result<(), CoreError> {
    output.fill(0.0);
    match weight_mode {
        WeightMode::Uniform => {
            for &index in indices {
                let label = labels[index as usize];
                if label < 0 || label as usize >= classes {
                    return Err(CoreError::InvalidCode(label));
                }
                output[label as usize] += 1.0;
            }
        }
        WeightMode::Distance => {
            let has_exact_match = distances.contains(&0.0);
            for (&distance, &index) in distances.iter().zip(indices.iter()) {
                let label = labels[index as usize];
                if label < 0 || label as usize >= classes {
                    return Err(CoreError::InvalidCode(label));
                }
                let weight = if has_exact_match {
                    if distance == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0 / distance
                };
                output[label as usize] += weight;
            }
        }
    }
    let total: f64 = output.iter().sum();
    if total > 0.0 {
        for value in output.iter_mut() {
            *value /= total;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn predict(
    query: &[f64],
    train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    labels: &[i64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    classes: usize,
    metric: DistanceMetric,
    weight_mode: WeightMode,
) -> Result<Vec<i64>, CoreError> {
    if labels.len() != train_rows {
        return Err(CoreError::ShapeMismatch);
    }
    if classes == 0 {
        return Err(CoreError::InsufficientClasses);
    }
    if matches!(metric, DistanceMetric::Euclidean) && matches!(weight_mode, WeightMode::Uniform) {
        let probabilities = euclidean_uniform_votes_blocked(
            query,
            train,
            train_transposed,
            train_norms,
            labels,
            query_rows,
            train_rows,
            columns,
            k,
            classes,
        )?;
        return Ok((0..query_rows)
            .map(|row| {
                let start = row * classes;
                best_voted_class(&probabilities[start..start + classes])
            })
            .collect());
    }
    let mut output = vec![0_i64; query_rows];
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, prediction)| {
            let (distances, indices) =
                row_neighbor_indices_for_vote(query, train, row, train_rows, columns, k, metric);
            let mut votes = vec![0.0; classes];
            if matches!(weight_mode, WeightMode::Uniform) {
                vote_uniform(labels, &indices, classes, &mut votes)?;
            } else {
                let output_distances: Vec<f64> = distances
                    .into_iter()
                    .map(|value| output_distance(value, metric))
                    .collect();
                let output_indices: Vec<i64> =
                    indices.into_iter().map(|index| index as i64).collect();
                vote_neighbors(
                    labels,
                    &output_distances,
                    &output_indices,
                    classes,
                    weight_mode,
                    &mut votes,
                )?;
            }
            *prediction = best_voted_class(&votes);
            Ok(())
        })
        .map(|_| output)
}

#[allow(clippy::too_many_arguments)]
pub fn predict_proba(
    query: &[f64],
    train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    labels: &[i64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    classes: usize,
    metric: DistanceMetric,
    weight_mode: WeightMode,
) -> Result<Vec<f64>, CoreError> {
    if labels.len() != train_rows {
        return Err(CoreError::ShapeMismatch);
    }
    if classes == 0 {
        return Err(CoreError::InsufficientClasses);
    }
    if matches!(metric, DistanceMetric::Euclidean) && matches!(weight_mode, WeightMode::Uniform) {
        return euclidean_uniform_votes_blocked(
            query,
            train,
            train_transposed,
            train_norms,
            labels,
            query_rows,
            train_rows,
            columns,
            k,
            classes,
        );
    }
    let mut output = vec![0.0; query_rows * classes];
    output
        .par_chunks_mut(classes)
        .enumerate()
        .try_for_each(|(row, probabilities)| {
            if matches!(weight_mode, WeightMode::Uniform) {
                let (_, indices) = row_neighbor_indices_for_vote(
                    query, train, row, train_rows, columns, k, metric,
                );
                vote_uniform(labels, &indices, classes, probabilities)?;
                for value in probabilities.iter_mut() {
                    *value /= k as f64;
                }
                Ok(())
            } else {
                let row_probabilities = row_probability(
                    query,
                    train,
                    labels,
                    row,
                    train_rows,
                    columns,
                    k,
                    classes,
                    metric,
                    weight_mode,
                )?;
                probabilities.copy_from_slice(&row_probabilities);
                Ok(())
            }
        })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn predict_regression(
    query: &[f64],
    train: &[f64],
    train_transposed: &[f64],
    train_norms: &[f64],
    targets: &[f64],
    query_rows: usize,
    train_rows: usize,
    columns: usize,
    k: usize,
    outputs: usize,
    metric: DistanceMetric,
    weight_mode: WeightMode,
) -> Result<Vec<f64>, CoreError> {
    if query_rows == 0 || train_rows == 0 || columns == 0 || outputs == 0 {
        return Err(CoreError::EmptyInput);
    }
    if query.len() != query_rows * columns
        || train.len() != train_rows * columns
        || targets.len() != train_rows * outputs
    {
        return Err(CoreError::ShapeMismatch);
    }
    if k == 0 || k > train_rows {
        return Err(CoreError::ShapeMismatch);
    }
    if matches!(metric, DistanceMetric::Euclidean) && matches!(weight_mode, WeightMode::Uniform) {
        return euclidean_uniform_regression_blocked(
            query,
            train,
            train_transposed,
            train_norms,
            targets,
            query_rows,
            train_rows,
            columns,
            k,
            outputs,
        );
    }
    let mut output = vec![0.0; query_rows * outputs];
    output
        .par_chunks_mut(outputs)
        .enumerate()
        .for_each(|(row, row_output)| {
            let (distances, indices) =
                row_neighbor_indices_for_vote(query, train, row, train_rows, columns, k, metric);
            match weight_mode {
                WeightMode::Uniform => {
                    let inverse_k = 1.0 / k as f64;
                    for &neighbor in &indices {
                        let target_offset = neighbor * outputs;
                        for target in 0..outputs {
                            row_output[target] += targets[target_offset + target] * inverse_k;
                        }
                    }
                }
                WeightMode::Distance => {
                    let has_exact_match = distances.contains(&0.0);
                    let mut total_weight = 0.0;
                    for (&distance, &neighbor) in distances.iter().zip(indices.iter()) {
                        let distance = output_distance(distance, metric);
                        let weight = if has_exact_match {
                            if distance == 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            1.0 / distance
                        };
                        total_weight += weight;
                        let target_offset = neighbor * outputs;
                        for target in 0..outputs {
                            row_output[target] += targets[target_offset + target] * weight;
                        }
                    }
                    if total_weight > 0.0 {
                        for value in row_output.iter_mut() {
                            *value /= total_weight;
                        }
                    }
                }
            }
        });
    Ok(output)
}
