# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0

"""Used declarations from nvmath-python 1.0.0; callable signatures from the adjacent NVIDIA .pxd."""

from enum import IntEnum

class SpMVAlg(IntEnum):
    DEFAULT = ...
    CSR_ALG1 = ...
    CSR_ALG2 = ...
    COO_ALG1 = ...
    COO_ALG2 = ...
    SELL_ALG1 = ...
    BSR_ALG1 = ...

class SpMMAlg(IntEnum):
    DEFAULT = ...
    CSR_ALG1 = ...
    CSR_ALG2 = ...
    CSR_ALG3 = ...
    COO_ALG1 = ...
    COO_ALG2 = ...
    COO_ALG3 = ...
    COO_ALG4 = ...
    BSR_ALG1 = ...

class IndexType(IntEnum):
    INDEX_32I = ...

class IndexBase(IntEnum):
    ZERO = ...

class Order(IntEnum):
    ROW = ...

class Operation(IntEnum):
    NON_TRANSPOSE = ...

class PointerMode(IntEnum):
    HOST = ...

class Status(IntEnum):
    NOT_SUPPORTED = ...
    INVALID_VALUE = ...
    ARCH_MISMATCH = ...
    INSUFFICIENT_RESOURCES = ...

def create() -> int: ...
def destroy(handle: int) -> None: ...
def get_version(handle: int) -> int: ...
def get_property(type: int) -> int: ...
def set_stream(handle: int, stream_id: int) -> None: ...
def set_pointer_mode(handle: int, mode: int) -> None: ...
def create_dn_vec(size: int, values: int, value_type: int) -> int: ...
def destroy_dn_vec(dn_vec_descr: int) -> None: ...
def dn_vec_set_values(dn_vec_descr: int, values: int) -> None: ...
def destroy_sp_mat(sp_mat_descr: int) -> None: ...
def sp_mat_set_values(sp_mat_descr: int, values: int) -> None: ...
def create_csr(
    rows: int,
    cols: int,
    nnz: int,
    csr_row_offsets: int,
    csr_col_ind: int,
    csr_values: int,
    csr_row_offsets_type: int,
    csr_col_ind_type: int,
    idx_base: int,
    value_type: int,
) -> int: ...
def create_coo(
    rows: int,
    cols: int,
    nnz: int,
    coo_row_ind: int,
    coo_col_ind: int,
    coo_values: int,
    coo_idx_type: int,
    idx_base: int,
    value_type: int,
) -> int: ...
def create_dn_mat(rows: int, cols: int, ld: int, values: int, value_type: int, order: int) -> int: ...
def destroy_dn_mat(dn_mat_descr: int) -> None: ...
def dn_mat_set_values(dn_mat_descr: int, values: int) -> None: ...
def sp_mv(
    handle: int,
    op_a: int,
    alpha: int,
    mat_a: int,
    vec_x: int,
    beta: int,
    vec_y: int,
    compute_type: int,
    alg: int,
    external_buffer: int,
) -> None: ...
def sp_mv_buffer_size(
    handle: int, op_a: int, alpha: int, mat_a: int, vec_x: int, beta: int, vec_y: int, compute_type: int, alg: int
) -> int: ...
def sp_mm_buffer_size(
    handle: int,
    op_a: int,
    op_b: int,
    alpha: int,
    mat_a: int,
    mat_b: int,
    beta: int,
    mat_c: int,
    compute_type: int,
    alg: int,
) -> int: ...
def sp_mm_preprocess(
    handle: int,
    op_a: int,
    op_b: int,
    alpha: int,
    mat_a: int,
    mat_b: int,
    beta: int,
    mat_c: int,
    compute_type: int,
    alg: int,
    external_buffer: int,
) -> None: ...
def sp_mm(
    handle: int,
    op_a: int,
    op_b: int,
    alpha: int,
    mat_a: int,
    mat_b: int,
    beta: int,
    mat_c: int,
    compute_type: int,
    alg: int,
    external_buffer: int,
) -> None: ...
def create_bsr(
    brows: int,
    bcols: int,
    bnnz: int,
    row_block_size: int,
    col_block_size: int,
    bsr_row_offsets: int,
    bsr_col_ind: int,
    bsr_values: int,
    bsr_row_offsets_type: int,
    bsr_col_ind_type: int,
    idx_base: int,
    value_type: int,
    order: int,
) -> int: ...
def sp_mv_preprocess(
    handle: int,
    op_a: int,
    alpha: int,
    mat_a: int,
    vec_x: int,
    beta: int,
    vec_y: int,
    compute_type: int,
    alg: int,
    external_buffer: int,
) -> None: ...

class cuSPARSEError(Exception):
    status: int
    def __init__(self, status: int) -> None: ...
