# Grid Search Results for Level: MEM_SHARE_T8_memory_comm

Generated on: 2026-06-14 21:02:25

Runs are sorted by **Map Coverage** (descending). OOMs at the bottom are sorted by allocation size (ascending).

| Rank | Run Name | Comm Merge | Attention Mode | Gradient Mode | Status | Steps | SPS | Map Coverage | Target Found | Success Rate | Eval Return | Details |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `grid_MEM_SHARE_T8_memory_comm_cat_memq_dial` | concat | attend_mem_query | dial | **SUCCESS** | 50,000,000 | 28,591 | 14.6% | 50.0% | 0.0% | ---- | Completed budget or early exit |
| 2 | `grid_MEM_SHARE_T8_memory_comm_res_memq_rial` | residual | attend_mem_query | rial | **SUCCESS** | 40,000,000 | 28,821 | 14.4% | 83.1% | 0.0% | ---- | Completed budget or early exit |
| 3 | `grid_MEM_SHARE_T8_memory_comm_res_obsq_rial` | residual | attend_cur_obs_query | rial | **SUCCESS** | 50,000,000 | 29,290 | 14.4% | 36.3% | 0.0% | ---- | Completed budget or early exit |
| 4 | `grid_MEM_SHARE_T8_memory_comm_res_obsmemq_dial` | residual | attend_cur_obs_and_mem_query | dial | **SUCCESS** | 32,000,000 | 28,129 | 14.2% | 58.7% | 0.0% | ---- | Completed budget or early exit |
| 5 | `grid_MEM_SHARE_T8_memory_comm_cat_gq_dial` | concat | attend_global_learned_query | dial | **SUCCESS** | 34,000,000 | 28,410 | 14.2% | 48.4% | 0.0% | ---- | Completed budget or early exit |
| 6 | `grid_MEM_SHARE_T8_memory_comm_res_gq_rial` | residual | attend_global_learned_query | rial | **SUCCESS** | 36,000,000 | 29,024 | 14.2% | 49.3% | 0.0% | ---- | Completed budget or early exit |
| 7 | `grid_MEM_SHARE_T8_memory_comm_cat_obsq_rial` | concat | attend_cur_obs_query | rial | **SUCCESS** | 28,000,000 | 26,363 | 14.1% | 69.5% | 0.0% | ---- | Completed budget or early exit |
| 8 | `grid_MEM_SHARE_T8_memory_comm_res_gq_dial` | residual | attend_global_learned_query | dial | **SUCCESS** | 32,000,000 | 27,076 | 14.0% | 71.8% | 0.0% | ---- | Completed budget or early exit |
| 9 | `grid_MEM_SHARE_T8_memory_comm_res_obsmemq_rial` | residual | attend_cur_obs_and_mem_query | rial | **SUCCESS** | 28,000,000 | 27,394 | 14.0% | 55.6% | 0.0% | ---- | Completed budget or early exit |
| 10 | `grid_MEM_SHARE_T8_memory_comm_res_memq_dial` | residual | attend_mem_query | dial | **SUCCESS** | 32,000,000 | 27,770 | 14.0% | 49.5% | 0.0% | ---- | Completed budget or early exit |
| 11 | `grid_MEM_SHARE_T8_memory_comm_cat_obsq_dial` | concat | attend_cur_obs_query | dial | **SUCCESS** | 24,000,000 | 27,274 | 13.9% | 62.9% | 0.0% | ---- | Completed budget or early exit |
| 12 | `grid_MEM_SHARE_T8_memory_comm_res_obsq_dial` | residual | attend_cur_obs_query | dial | **SUCCESS** | 24,000,000 | 28,011 | 13.9% | 49.2% | 0.0% | ---- | Completed budget or early exit |
| 13 | `grid_MEM_SHARE_T8_memory_comm_cat_obsmemq_dial` | concat | attend_cur_obs_and_mem_query | dial | **SUCCESS** | 20,000,000 | 26,849 | 13.8% | 54.4% | 0.0% | ---- | Completed budget or early exit |
| 14 | `grid_MEM_SHARE_T8_memory_comm_cat_obsmemq_rial` | concat | attend_cur_obs_and_mem_query | rial | **SUCCESS** | 30,000,000 | 26,923 | 13.7% | 47.3% | 0.0% | ---- | Completed budget or early exit |
| 15 | `grid_MEM_SHARE_T8_memory_comm_cat_memq_rial` | concat | attend_mem_query | rial | **SUCCESS** | 24,000,000 | 26,766 | 13.7% | 48.2% | 0.0% | ---- | Completed budget or early exit |
| 16 | `grid_MEM_SHARE_T8_memory_comm_cat_gq_rial` | concat | attend_global_learned_query | rial | **SUCCESS** | 40,000,000 | 28,739 | 13.5% | 51.2% | 0.0% | ---- | Completed budget or early exit |
