
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Dummy16, Dummy64, Gs, Ns};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const DS_F32: f32 = Ds::SIZE as f32;

pub(crate) fn normalize_query<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 2], m![Ns % 2, Gs]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_final()
        .commit_trim::<m![Ns % 4, Gs]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn load_norm_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = rms_weight.to_dm(&mut ctx.tdma);

    ctx.sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

fn root_mean_square<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 4], m![Ns % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![1], m![Ns]>()
        .vector_final()
        .commit_trim::<m![Ns]>()
        .commit();

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .to_vrf()
}

fn scale_by_rms_and_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn scale_by_rms<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_key(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_rms_and_weight(ctx, x, &rms_vrf, &weight_vrf)
}

pub(crate) fn normalize_value_c<C: M, S: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, S, m![Ns, Ds]>,
) -> DmTensor<bf16, Chip, C, S, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);
    scale_by_rms(ctx, x, &rms_vrf)
}

pub(crate) fn normalize_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    normalize_value_c(ctx, x)
}


pub(crate) fn normalize_value_dist4(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, Ds / 4], m![Ds % 4]>,
) -> DmTensor<bf16, Chip, m![Ns / 4 % 2], Slice, m![Ns % 4, Ds]> {
    type VCluster = m![Ns / 4 % 2];
    type VSlices = m![Ns % 4, Ds / 4];
    type VReduceSlices = m![Ns % 4, Dummy64];

    let mean_square: DmTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1 # 2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let reduced: DmTensor<f32, Chip, VCluster, VReduceSlices, m![1 # 8]> = ctx.main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<VReduceSlices, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, VCluster, VReduceSlices, m![1 # 8]> = ctx.main
        .begin(reduced.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, VCluster, VSlices, m![Ds % 4]> = ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<Slice, m![Ns % 4, Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ns % 4, Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}


pub(crate) fn normalize_value_dist16(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]>,
) -> DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]> {
    type VCluster = m![Ns / 4 % 2];
    type VSlices = m![Ns % 4, 1 #{!} 4, Ds / 16];
    type VReduceSlices = m![Ns % 4, 1 #{!} 4, Dummy16];

    let mean_square: DmTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4 % 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();

    let reduced: DmTensor<f32, Chip, VCluster, VReduceSlices, m![1 # 8]> = ctx.main
        .begin(mean_square.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<VReduceSlices, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final().commit_trim::<m![1 # 8]>().commit();

    let rms: DmTensor<f32, Chip, VCluster, VReduceSlices, m![1 # 8]> = ctx.main
        .begin(reduced.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let rms: DmTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, VCluster, VSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    ctx.main.begin(x.view())
        .fetch::<m![1], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4 % 4], m![Ds % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_final().cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>().commit()
}


pub(crate) fn normalize_value_dist4_via_pack(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, Ds / 4], m![Ds % 4]>,
) -> DmTensor<bf16, Chip, m![Ns / 4 % 2], Slice, m![Ns % 4, Ds]> {
    let packed: DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]> =
        x.to_dm(&mut ctx.tdma);
    let normalized = normalize_value_dist16(ctx, &packed);
    normalized.to_dm(&mut ctx.tdma)
}


pub(crate) fn normalize_key_dist16(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]> {
    type KCluster = m![Ns / 4 % 2];
    type KSlices = m![Ns % 4, 1 #{!} 4, Ds / 16];
    type KReduceSlices = m![Ns % 4, 1 #{!} 4, Dummy16];

    let mean_square: DmTensor<f32, Chip, KCluster, KSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4 % 4], m![Ds % 4]>()
        .vector_stash().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32).vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();

    let reduced: DmTensor<f32, Chip, KCluster, KReduceSlices, m![1 # 8]> = ctx.main
        .begin(mean_square.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<KReduceSlices, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final().commit_trim::<m![1 # 8]>().commit();

    let rms: DmTensor<f32, Chip, KCluster, KReduceSlices, m![1 # 8]> = ctx.main
        .begin(reduced.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    let rms: DmTensor<f32, Chip, KCluster, KSlices, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, KCluster, KSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let weight_dm: DmTensor<bf16, Chip, KCluster, KSlices, m![Ds % 16]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, KCluster, KSlices, m![Ds % 16]> = ctx.sub
        .begin(weight_dm.view()).fetch::<m![1], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8 % 2], m![Ds % 8]>().to_vrf();

    ctx.main.begin(x.view())
        .fetch::<m![1], m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4 % 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8 % 2], m![Ds % 8]>()
        .vector_final().cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>().commit()
}

pub(crate) fn normalize_key_dist4_via_pack(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, Ds / 4], m![Ds % 4]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, m![Ns / 4 % 2], Slice, m![Ns % 4, Ds]> {
    let packed: DmTensor<bf16, Chip, m![Ns / 4 % 2], m![Ns % 4, 1 #{!} 4, Ds / 16], m![Ds % 16]> =
        x.to_dm(&mut ctx.tdma);
    let normalized = normalize_key_dist16(ctx, &packed, rms_weight);
    normalized.to_dm(&mut ctx.tdma)
}
