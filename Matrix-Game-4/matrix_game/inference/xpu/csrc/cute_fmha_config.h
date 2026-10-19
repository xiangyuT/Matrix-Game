// Private compile-time CUTE FMHA sweep configuration.
#pragma once

#ifndef CUTE_TUNE_Q_TILE
#define CUTE_TUNE_Q_TILE 256
#endif
#ifndef CUTE_TUNE_KV_TILE
#define CUTE_TUNE_KV_TILE 32
#endif
#ifndef CUTE_TUNE_V_TILE
#define CUTE_TUNE_V_TILE 32
#endif
#ifndef CUTE_TUNE_MMA_K
#define CUTE_TUNE_MMA_K 32
#endif
#ifndef CUTE_TUNE_SUBGROUP_LAYOUT_Q
#define CUTE_TUNE_SUBGROUP_LAYOUT_Q 16
#endif
#ifndef CUTE_TUNE_PIPELINE_STAGES
#define CUTE_TUNE_PIPELINE_STAGES 2
#endif
#ifndef CUTE_TUNE_GRF_SIZE
#define CUTE_TUNE_GRF_SIZE 256
#endif

namespace cute_fmha_config {

struct BMGTag {};
struct PTLHTag {};

template <
    typename ArchTag,
    int QTile,
    int KvTile,
    int VTile,
    int MmaK,
    int HeadDim,
    int SubgroupLayoutQ,
    int PipelineStages,
    int GrfSize>
struct Config {
  static constexpr int Q_TILE = QTile;
  static constexpr int KV_TILE = KvTile;
  static constexpr int V_TILE = VTile;
  static constexpr int MMA_K = MmaK;
  static constexpr int HEAD_DIM = HeadDim;
  static constexpr int SUBGROUP_LAYOUT_Q = SubgroupLayoutQ;
  static constexpr int PIPELINE_STAGES = PipelineStages;
  static constexpr int GRF_SIZE = GrfSize;
};

using ConfigBMG = Config<
    BMGTag,
    CUTE_TUNE_Q_TILE,
    CUTE_TUNE_KV_TILE,
    CUTE_TUNE_V_TILE,
    CUTE_TUNE_MMA_K,
    128,
    CUTE_TUNE_SUBGROUP_LAYOUT_Q,
    CUTE_TUNE_PIPELINE_STAGES,
    CUTE_TUNE_GRF_SIZE>;
using ConfigPTLH = Config<
    PTLHTag,
    CUTE_TUNE_Q_TILE,
    CUTE_TUNE_KV_TILE,
    CUTE_TUNE_V_TILE,
    CUTE_TUNE_MMA_K,
    128,
    CUTE_TUNE_SUBGROUP_LAYOUT_Q,
    CUTE_TUNE_PIPELINE_STAGES,
    CUTE_TUNE_GRF_SIZE>;

#if defined(OMNI_XPU_ARCH_BMG) && defined(OMNI_XPU_ARCH_PTL_H)
#error "Select only one omni_xpu_kernel GPU architecture"
#elif defined(OMNI_XPU_ARCH_BMG)
using ActiveConfig = ConfigBMG;
#elif defined(OMNI_XPU_ARCH_PTL_H)
using ActiveConfig = ConfigPTLH;
#else
#error "Define OMNI_XPU_ARCH_BMG or OMNI_XPU_ARCH_PTL_H"
#endif

}  // namespace cute_fmha_config
