// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include "zlib.h"

#include "shader.h"
#include "shader_trace.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "visualizer.h"
#include "stats.h"
#include "../cuda-sim/cuda_device_runtime.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

bool g_interactive_debugger_enabled = false;

unsigned long long gpu_sim_cycle = 0;
unsigned long long gpu_tot_sim_cycle = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0;
unsigned int gpu_stall_icnt2sh = 0;
unsigned long long partiton_reqs_in_parallel = 0;
unsigned long long partiton_reqs_in_parallel_total = 0;
unsigned long long partiton_reqs_in_parallel_util = 0;
unsigned long long partiton_reqs_in_parallel_util_total = 0;
unsigned long long gpu_sim_cycle_parition_util = 0;
unsigned long long gpu_tot_sim_cycle_parition_util = 0;
unsigned long long partiton_replys_in_parallel = 0;
unsigned long long partiton_replys_in_parallel_total = 0;
std::vector<unsigned long long> *partition_replys_total_per_stream;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {

    option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
                           &g_power_config_name, "GPUWattch XML file",
                           "gpuwattch.xml");

    option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                           &g_power_simulation_enabled,
                           "Turn on power simulator (1=On, 0=Off)", "0");

    option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                           &g_power_per_cycle_dump,
                           "Dump detailed power output each cycle", "0");

    // Output Data Formats
    option_parser_register(
        opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
        "produce a file for the power trace (1=On, 0=Off)", "0");

    option_parser_register(opp, "-power_trace_zlevel", OPT_INT32,
                           &g_power_trace_zlevel,
                           "Compression level of the power trace output log "
                           "(0=no comp, 9=highest)",
                           "6");

    option_parser_register(
        opp, "-steady_power_levels_enabled", OPT_BOOL,
        &g_steady_power_levels_enabled,
        "produce a file for the steady power levels (1=On, 0=Off)", "0");

    option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                           &gpu_steady_state_definition,
                           "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
    option_parser_register(opp, "-perf_sim_memcpy", OPT_BOOL,
                           &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                           "1");
    option_parser_register(
        opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type,
        "0 = fifo, 1 = FR-FCFS (default), 2 = FR-Priority", "1");
    option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                           &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i",
                           "8:8:8:8");

    option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                           "Use a ideal L2 cache that always hit", "0");
    option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                           &m_L2_config.m_config_string,
                           "unified banked L2 data cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq>}",
                           "64:128:8,L:B:m:N,A:16:4,4");
    option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                           &m_L2_texure_only, "L2 cache used for texture only",
                           "1");
    option_parser_register(
        opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
        "number of memory modules (e.g. memory controllers) in gpu", "8");
    option_parser_register(
        opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
        &m_n_sub_partition_per_memory_channel,
        "number of memory subpartition in each memory module", "1");
    option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                           &gpu_n_mem_per_ctrlr,
                           "number of memory chips per memory controller", "1");
    option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                           &gpgpu_memlatency_stat,
                           "track and display latency statistics 0x2 enables "
                           "MC, 0x4 enables queue logs",
                           "0");
    option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size",
                           OPT_INT32, &gpgpu_frfcfs_dram_sched_queue_size,
                           "0 = unlimited (default); # entries per chip", "0");
    option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                           &gpgpu_dram_return_queue_size,
                           "0 = unlimited (default); # entries per chip", "0");
    option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                           "default = 4 bytes (8 bytes per cycle at DDR)", "4");
    option_parser_register(
        opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
        "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
    option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                           &data_command_freq_ratio,
                           "Frequency ratio between DRAM data bus and command "
                           "bus (default = 2 times, i.e. DDR)",
                           "2");
    option_parser_register(
        opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
        "DRAM timing parameters = "
        "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
        "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
    option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                           "ROP queue latency (default 85)", "85");
    option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                           "DRAM latency (default 30)", "30");
    option_parser_register(opp, "-dual_bus_interface", OPT_UINT32,
                           &dual_bus_interface,
                           "dual_bus_interface (default = 0) ", "0");
    option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                           &dram_bnk_indexing_policy,
                           "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                           "Xoring with the higher bits) (Default = 0)",
                           "0");
    option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                           &dram_bnkgrp_indexing_policy,
                           "dram_bnkgrp_indexing_policy (0 = take higher bits, "
                           "1 = take lower bits) (Default = 0)",
                           "0");
    option_parser_register(opp, "-Seperate_Write_Queue_Enable", OPT_BOOL,
                           &seperate_write_queue_enabled,
                           "Seperate_Write_Queue_Enable", "0");
    option_parser_register(opp, "-Write_Queue_Size", OPT_CSTR,
                           &write_queue_size_opt, "Write_Queue_Size",
                           "32:28:16");
    option_parser_register(
        opp, "-Elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
        "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
    option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                           "icnt_flit_size", "32");

    option_parser_register(opp, "-l2_partition_enabled", OPT_BOOL,
                           &(m_L2_config.m_partition_enabled),
                           "Enable L2 partitioning among streams", "0");

    option_parser_register(
        opp, "-l2_partition", OPT_CSTR, &(m_L2_config.cache_partition_str),
        "<l2_allocation_in_stream_default>:<in_stream_1>:<in_stream_2>",
        "0:0.5:0.5");

    option_parser_register(
        opp, "-l2d_enabled", OPT_CSTR, &(m_L2_config.l2d_enabled_str),
        "<l2d_enabled_in_stream_default>:<in_stream_1>:<in_stream_2>", "1:1:1");

    m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                           "1 = post-dominator", "1");
    option_parser_register(
        opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
        &gpgpu_shader_core_pipeline_opt,
        "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                           &m_L1T_config.m_config_string,
                           "per-shader L1 texture cache  (READ-ONLY) config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                           "8:128:5,L:R:m:N,F:128:4,128:2");
    option_parser_register(
        opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
        "per-shader L1 constant memory cache  (READ-ONLY) config "
        " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
        "merge>,<mq>} ",
        "64:64:2,L:R:f:N,A:2:32,4");
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                           &m_L1I_config.m_config_string,
                           "shader L1 instruction cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                           "4:256:4,L:R:f:N,A:2:32,4");
    option_parser_register(opp, "-gpgpu_cache:il0", OPT_CSTR,
                           &m_L0I_config.m_config_string,
                           "shader L0 instruction cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                           "4:256:4,L:R:f:N,A:2:32,4");
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                           &m_L1D_config.m_config_string,
                           "per-shader L1 data cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                           "none");
    option_parser_register(opp, "-l1_banks", OPT_UINT32, &m_L1D_config.l1_banks,
                           "The number of L1 cache banks", "1");
    option_parser_register(opp, "-l1_latency", OPT_UINT32,
                           &m_L1D_config.l1_latency, "L1 Hit Latency", "0");
    option_parser_register(opp, "-smem_latency", OPT_UINT32, &smem_latency,
                           "smem Latency", "3");
    option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                           &m_L1D_config.m_config_stringPrefL1,
                           "per-shader L1 data cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                           "none");
    option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                           &m_L1D_config.m_config_stringPrefShared,
                           "per-shader L1 data cache config "
                           " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                           "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                           "none");
    option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                           "global memory access skip L1D cache (implements "
                           "-Xptxas -dlcm=cg, default=no skip)",
                           "0");

    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                           &gpgpu_perfect_mem,
                           "enable perfect memory mode (no cache miss)", "0");
    option_parser_register(
        opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
        "group of lanes that should be read/written together)", "4");
    option_parser_register(opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL,
                           &gpgpu_clock_gated_reg_file,
                           "enable clock gated reg file for power calculations",
                           "0");
    option_parser_register(
        opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
        "enable clock gated lanes for power calculations", "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                           &gpgpu_shader_registers,
                           "Number of registers per shader core. Limits number "
                           "of concurrent CTAs. (default 8192)",
                           "8192");
    option_parser_register(
        opp, "-gpgpu_registers_per_block", OPT_UINT32,
        &gpgpu_registers_per_block,
        "Maximum number of registers per CTA. (default 8192)", "8192");
    option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                           &gpgpu_ignore_resources_limitation,
                           "gpgpu_ignore_resources_limitation (default 0)",
                           "0");
    option_parser_register(
        opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
        "Maximum number of concurrent CTAs in shader (default 8)", "8");
    option_parser_register(
        opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
        "Maximum number of named barriers per CTA (default 16)", "16");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32,
                           &n_simt_clusters, "number of processing clusters",
                           "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                           &n_simt_cores_per_cluster,
                           "number of simd cores per cluster", "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                           OPT_UINT32, &n_simt_ejection_buffer_size,
                           "number of packets in ejection buffer", "8");
    option_parser_register(
        opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
        &ldst_unit_response_queue_size,
        "number of response packets in ld/st unit ejection buffer", "2");
    option_parser_register(
        opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
        "Size of shared memory per thread block or CTA (default 48kB)",
        "49152");
    option_parser_register(
        opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
        "Size of shared memory per shader core (default 16kB)", "16384");
    option_parser_register(opp, "-adaptive_cache_config", OPT_BOOL,
                           &adaptive_volta_cache_config,
                           "adaptive_volta_cache_config", "0");
    option_parser_register(
        opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
        "Size of shared memory per shader core (default 16kB)", "16384");
    option_parser_register(
        opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
        "Size of shared memory per shader core (default 16kB)", "16384");
    option_parser_register(
        opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
        &gpgpu_shmem_sizePrefShared,
        "Size of shared memory per shader core (default 16kB)", "16384");
    option_parser_register(
        opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
        "Number of banks in the shared memory in each shader core (default 16)",
        "16");
    option_parser_register(
        opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL,
        &shmem_limited_broadcast,
        "Limit shared memory to do one broadcast per cycle (default on)", "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                           &mem_warp_parts,
                           "Number of portions a warp is divided into for "
                           "shared memory bank conflict check ",
                           "2");
    option_parser_register(
        opp, "-mem_unit_ports", OPT_INT32, &mem_unit_ports,
        "The number of memory transactions allowed per core cycle", "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                           &mem_warp_parts,
                           "Number of portions a warp is divided into for "
                           "shared memory bank conflict check ",
                           "2");
    option_parser_register(
        opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
        "Specify which shader core to collect the warp size distribution from",
        "-1");
    option_parser_register(
        opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
        "Specify which shader core to collect the warp issue distribution from",
        "0");
    option_parser_register(
        opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map,
        "Mapping from local memory space address to simulated GPU physical "
        "address space (default = enabled)",
        "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                           &gpgpu_num_reg_banks,
                           "Number of register banks (default = 8)", "8");
    option_parser_register(
        opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL,
        &gpgpu_reg_bank_use_warp_id,
        "Use warp ID in mapping registers to banks (default = off)", "0");
    option_parser_register(opp, "-sub_core_model", OPT_BOOL, &sub_core_model,
                           "Sub Core Volta/Pascal model (default = off)", "0");
    option_parser_register(opp, "-enable_specialized_operand_collector",
                           OPT_BOOL, &enable_specialized_operand_collector,
                           "enable_specialized_operand_collector", "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                           OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                           "number of collector units (default = 4)", "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                           OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                           "number of collector units (default = 0)", "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                           OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                           "number of collector units (default = 4)", "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                           OPT_INT32, &gpgpu_operand_collector_num_units_int,
                           "number of collector units (default = 0)", "0");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_units_tensor_core", OPT_INT32,
        &gpgpu_operand_collector_num_units_tensor_core,
        "number of collector units (default = 4)", "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                           OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                           "number of collector units (default = 2)", "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                           OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                           "number of collector units (default = 0)", "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
        &gpgpu_operand_collector_num_in_ports_tensor_core,
        "number of collector unit in ports (default = 1)", "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                           OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                           OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                           OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                           "number of collector unit in ports (default = 0)",
                           "0");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32,
        &gpgpu_operand_collector_num_out_ports_sfu,
        "number of collector unit in ports (default = 1)", "1");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_out_ports_int", OPT_INT32,
        &gpgpu_operand_collector_num_out_ports_int,
        "number of collector unit in ports (default = 0)", "0");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
        &gpgpu_operand_collector_num_out_ports_tensor_core,
        "number of collector unit in ports (default = 1)", "1");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32,
        &gpgpu_operand_collector_num_out_ports_mem,
        "number of collector unit in ports (default = 1)", "1");
    option_parser_register(
        opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32,
        &gpgpu_operand_collector_num_out_ports_gen,
        "number of collector unit in ports (default = 0)", "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                           &gpgpu_coalesce_arch,
                           "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                           &gpgpu_num_sched_per_core,
                           "Number of warp schedulers per core", "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                           &gpgpu_max_insn_issue_per_warp,
                           "Max number of instructions that can be issued per "
                           "warp in one cycle by scheduler (either 1 or 2)",
                           "2");
    option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                           &gpgpu_dual_issue_diff_exec_units,
                           "should dual issue use two different execution unit "
                           "resources (Default = 1)",
                           "1");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                           &simt_core_sim_order,
                           "Select the simulation order of cores in a cluster "
                           "(0=Fix, 1=Round-Robin)",
                           "1");
    option_parser_register(
        opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
        "Pipeline widths "
        "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_"
        "EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
        "1,1,1,1,1,1,1,1,1,1,1,1,1");
    option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_INT32,
                           &gpgpu_tensor_core_avail,
                           "Tensor Core Available (default=0)", "0");
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32,
                           &gpgpu_num_sp_units,
                           "Number of SP units (default=1)", "1");
    option_parser_register(opp, "-gpgpu_num_dp_units", OPT_INT32,
                           &gpgpu_num_dp_units,
                           "Number of DP units (default=0)", "0");
    option_parser_register(opp, "-gpgpu_num_int_units", OPT_INT32,
                           &gpgpu_num_int_units,
                           "Number of INT units (default=0)", "0");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32,
                           &gpgpu_num_sfu_units,
                           "Number of SF units (default=1)", "1");
    option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_INT32,
                           &gpgpu_num_tensor_core_units,
                           "Number of tensor_core units (default=1)", "1");
    option_parser_register(
        opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
        "Number if ldst units (default=1) WARNING: not hooked up to anything",
        "1");
    option_parser_register(opp, "-gpgpu_num_control_units", OPT_INT32,
                           &gpgpu_num_control_units,
                           "Number of control units (default=1)", "1");
    option_parser_register(
        opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
        "Scheduler configuration: < lrr | gto | two_level_active > "
        "If "
        "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
        "prioritization>"
        "For complete list of prioritization values see shader.h enum "
        "scheduler_prioritization_type"
        "Default: gto",
        "gto");

    option_parser_register(
        opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL,
        &gpgpu_concurrent_kernel_sm,
        "Support concurrent kernels on a SM (default = disabled)", "0");

    option_parser_register(
        opp, "-gpgpu_sharing_intra_sm", OPT_UINT32, &gpgpu_sharing_intra_sm,
        "Use intra SM sharing for concurrent kernel implementation", "1");
    option_parser_register(
        opp, "-max_sm_in_stream", OPT_CSTR, &max_sm_in_stream,
        "<max_sm_in_stream_default>:<in_stream_1>:<in_stream_2>", "0:40:40");

    option_parser_register(opp, "-gpgpu_warp_sample_cta", OPT_UINT32,
                           &warp_state_sample_cta,
                           "The first warp of any CTA ID that is a multiple of "
                           "this value will be tracked for warp states.",
                           "10");
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);
    power_config::reg_options(opp);
    option_parser_register(
        opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt,
        "terminates gpu simulation early (0 = no limit)", "0");
    option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt,
                           "terminates gpu simulation early (0 = no limit)",
                           "0");
    option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                           "terminates gpu simulation early (0 = no limit)",
                           "0");
    option_parser_register(
        opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
        "display runtime statistics such as dram utilization {<freq>:<flag>}",
        "10000:0");
    option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                           &liveness_message_freq,
                           "Minimum number of seconds between simulation "
                           "liveness messages (0 = always print)",
                           "1");
    option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                           &gpgpu_compute_capability_major,
                           "Major compute capability version number", "7");
    option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                           &gpgpu_compute_capability_minor,
                           "Minor compute capability version number", "0");
    option_parser_register(
        opp, "-gpgpu_flush_l1_cache", OPT_BOOL, &gpgpu_flush_l1_cache,
        "Flush L1 cache at the end of each kernel call", "0");
    option_parser_register(
        opp, "-gpgpu_flush_l2_cache", OPT_BOOL, &gpgpu_flush_l2_cache,
        "Flush L2 cache at the end of each kernel call", "0");
    option_parser_register(
        opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
        "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
    option_parser_register(opp, "-gpgpu_ptx_instruction_classification",
                           OPT_INT32, &gpgpu_ptx_instruction_classification,
                           "if enabled will classify ptx instruction types per "
                           "kernel (Max 255 kernels now)",
                           "0");
    option_parser_register(
        opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode,
        "Select between Performance (default) or Functional simulation (1)",
        "0");
    option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                           &gpgpu_clock_domains,
                           "Clock Domain Frequencies in MhZ {<Core "
                           "Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                           "500.0:2000.0:2000.0:2000.0");
    option_parser_register(
        opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
        "maximum kernels that can run concurrently on GPU", "8");
    option_parser_register(
        opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
        "Interval between each snapshot in control flow logger", "0");
    option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                           &g_visualizer_enabled,
                           "Turn on visualizer output (1=On, 0=Off)", "1");
    option_parser_register(
        opp, "-visualizer_outputfile", OPT_CSTR, &g_visualizer_filename,
        "Specifies the output log file for visualizer", NULL);
    option_parser_register(
        opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
        "Compression level of the visualizer output log (0=no comp, 9=highest)",
        "6");
    option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                           &stack_size_limit, "GPU thread stack size", "1024");
    option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                           &heap_size_limit, "GPU malloc heap size ",
                           "8388608");
    option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                           &runtime_sync_depth_limit,
                           "GPU device runtime synchronize depth", "2");
    option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                           OPT_INT32, &runtime_pending_launch_count_limit,
                           "GPU device runtime pending launch count", "2048");
    option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                           "Turn on traces", "0");
    option_parser_register(opp, "-trace_components", OPT_CSTR,
                           &Trace::config_str,
                           "comma seperated list of traces to enable. "
                           "Complete list found in trace_streams.tup. "
                           "Default none",
                           "none");
    option_parser_register(
        opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
        "The core which is printed using CORE_DPRINTF. Default 0", "0");
    option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                           &Trace::sampling_memory_partition,
                           "The memory partition which is printed using "
                           "MEMPART_DPRINTF. Default -1 (i.e. all)",
                           "-1");
    ptx_file_line_stats_options(opp);

    // Jin: kernel launch latency
    option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_CSTR,
                           &kernel_launch_latency_str,
                           "Kernel launch latency in cycles. Default: 0:0:0",
                           "0:0:0");
    extern bool g_cdp_enabled;
    option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL, &g_cdp_enabled,
                           "Turn on CDP", "0");
    option_parser_register(opp, "-delayed_cycle_btw_kernels", OPT_INT32,
                           &delayed_cycle_btw_kernels,
                           "Number of cycles to delay the second kernel", "0");

    // intra-SM settings
    option_parser_register(opp, "-intra_sm_option", OPT_INT32, &intra_sm_option,
                           "0: SMK, 1: passed execution context proportion, 2: "
                           "passed max cta per stream, "
                           "3: passed cta max per kernel per stream",
                           "0");

    // Option 1: fixed execution context proportion
    option_parser_register(opp, "-ctx_ratio_in_stream", OPT_CSTR,
                           &ctx_ratio_str,
                           "<Execution context percentage in default "
                           "stream>:<in_stream_1>:<in_stream_2>",
                           "0:0.5:0.5");

    // Option 2: customize number of ctas for each stream
    option_parser_register(
        opp, "-max_cta_in_stream", OPT_CSTR, &max_cta_str,
        "<max_cta_in_stream_default>:<in_stream_1>:<in_stream_2>", "0:0:0");

    // Option 3: cta config look-up table for each kernel pair
    option_parser_register(opp, "-cta_lut", OPT_CSTR, &cta_lut_str,
                           "kidx=>cta, ...", "0:1:1=0:1:1");

    option_parser_register(
        opp, "-icnt_priority", OPT_CSTR, &icnt_priority_str,
        "<priority in stream 0>:<in_stream_1>:<in_stream_2>, higher number "
        "indicates higher priority",
        "1:1:1");

    option_parser_register(opp, "-print_at_device_sync", OPT_BOOL,
                           &g_print_at_device_sync,
                           "Only print stats when device sync is called", "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
    i.x++;
    if (i.x >= bound.x) {
        i.x = 0;
        i.y++;
        if (i.y >= bound.y) {
            i.y = 0;
            if (i.z < bound.z)
                i.z++;
        }
    }
}

void gpgpu_sim::calculate_smk_quota(
    std::map<unsigned int, kernel_usage_info> &usage_map) {
    unsigned num_kernel = usage_map.size();

    float tot_thread = 0;
    float tot_smem = 0;
    float tot_reg = 0;
    float tot_cta = 0;

    while (num_kernel > 0) {
        // find the lowest usage_map
        float min_usage = 1;
        unsigned min_k;

        for (auto k_usage : usage_map) {
            if (k_usage.second.being_considered) {
                float current_usage =
                    k_usage.second.cta_quota * k_usage.second.max_usage;

                if (current_usage < min_usage) {
                    min_usage = current_usage;
                    min_k = k_usage.first;
                }
            }
        }

        // check if we can add one cta of the min_k without exceeding resource
        // limits
        if ((usage_map[min_k].usage.thread_usage + tot_thread) <= 1.0 &&
            (usage_map[min_k].usage.smem_usage + tot_smem) <= 1.0 &&
            (usage_map[min_k].usage.reg_usage + tot_reg) <= 1.0 &&
            (usage_map[min_k].usage.cta_usage + tot_cta) <= 1.0 &&
            usage_map[min_k].cta_quota < usage_map[min_k].grid_over_sm) {

            tot_thread += usage_map[min_k].usage.thread_usage;
            tot_smem += usage_map[min_k].usage.smem_usage;
            tot_reg += usage_map[min_k].usage.reg_usage;
            tot_cta += usage_map[min_k].usage.cta_usage;

            // let's add one cta of this kernel
            usage_map[min_k].cta_quota++;
        } else {
            // mark min_k as donezo
            usage_map[min_k].being_considered = false;
            num_kernel--;
        }
    }
}

void gpgpu_sim::set_resource_config() {
    printf(">>>>>>>>> set_resource_config:\n");

    // Grab resource usage information from currently running kernels
    std::map<unsigned, kernel_usage_info> usage_map;

    for (unsigned idx = 0; idx < m_running_kernels.size(); idx++) {
        kernel_info_t *kernel = m_running_kernels[idx];

        if (kernel && !kernel->done()) {
            if (!kernel->has_set_usage()) {
                // new incoming kernel
                // iterate the following resources that could limit cta quota

                unsigned threads_per_cta = kernel->threads_per_cta();
                unsigned int padded_cta_size = threads_per_cta;
                unsigned int warp_size = getShaderCoreConfig()->warp_size;

                if (padded_cta_size % warp_size)
                    padded_cta_size =
                        ((padded_cta_size / warp_size) + 1) * (warp_size);

                // 1. thread slots
                float thread_usage = (float)padded_cta_size /
                                     getShaderCoreConfig()->n_thread_per_shader;

                const class function_info *func_info = kernel->entry();
                const struct gpgpu_ptx_sim_info *kernel_info =
                    ptx_sim_kernel_info(func_info);

                // 2. shared mem
                float smem_usage = (float)kernel_info->smem /
                                   getShaderCoreConfig()->gpgpu_shmem_size;

                // 3. register
                float reg_usage =
                    padded_cta_size * ((kernel_info->regs + 3) & ~3) /
                    ((float)getShaderCoreConfig()->gpgpu_shader_registers);

                // 4. cta slots
                float cta_usage =
                    1.0f / getShaderCoreConfig()->max_cta_per_core;

                kernel->set_usage(thread_usage, smem_usage, reg_usage,
                                  cta_usage);
            }

            // calculate usage_map for the kernel => max resource usage
            kernel_usage_info usage_info;

            Usage k_usage = kernel->get_usage();
            usage_info.usage = k_usage;

            usage_info.max_usage = std::max(
                k_usage.thread_usage,
                std::max(k_usage.smem_usage,
                         std::max(k_usage.reg_usage, k_usage.cta_usage)));

            usage_info.cta_quota = 0;

            usage_info.grid_over_sm = ceil(((float)kernel->num_blocks()) /
                                           getShaderCoreConfig()->num_shader());

            usage_info.being_considered = true;

            usage_map[idx] = usage_info;
        }
    }

    // Set CTA quota based on sharing mechanisms
    switch (m_config.intra_sm_option) {
    case intra_sm_option_t ::MAX_CTA: {
        for (auto k_usage : usage_map) {
            const unsigned stream_id =
                m_running_kernels[k_usage.first]->get_stream_id();
            const unsigned cta_quota =
                m_config.get_max_cta_by_stream(stream_id);
            assert(cta_quota > 0);

            m_running_kernels[k_usage.first]->set_cta_quota(cta_quota);
        }
        break;
    }

    case intra_sm_option_t ::CTX_RATIO: {
        for (auto k_usage : usage_map) {
            const unsigned stream_id =
                m_running_kernels[k_usage.first]->get_stream_id();
            float config_ctx_ratio =
                m_config.get_ctx_ratio_by_stream(stream_id);

            unsigned cta_quota =
                std::floor(config_ctx_ratio / k_usage.second.max_usage);

            assert(cta_quota > 0);
            m_running_kernels[k_usage.first]->set_cta_quota(cta_quota);

            printf("ctx ratio: %f, max_usage: %f\n", config_ctx_ratio,
                   k_usage.second.max_usage);
        }

        break;
    }

    case intra_sm_option_t ::CTA_LUT: {
        // Case I: there are two concurrent kernels, query LUT
        if (usage_map.size() > 1) {
            std::vector<unsigned> kidx;
            // +1 for default stream
            kidx.resize(usage_map.size() + 1, 0);

            for (auto k_usage : usage_map) {
                const unsigned stream_id =
                    m_running_kernels[k_usage.first]->get_stream_id();
                assert(stream_id < kidx.size());

                unsigned adj_kidx =
                    m_running_kernels[k_usage.first]->get_uid_in_stream() %
                    num_kernel_stream[stream_id];
                if (adj_kidx == 0) {
                    adj_kidx = num_kernel_stream[stream_id];
                }

                kidx[stream_id] = adj_kidx;
            }

            const std::vector<unsigned> cta_quota =
                m_config.get_cta_from_lut(kidx);

            for (auto k_usage : usage_map) {
                const unsigned stream_id =
                    m_running_kernels[k_usage.first]->get_stream_id();
                const unsigned quota = cta_quota[stream_id];
                assert(quota > 0);
                m_running_kernels[k_usage.first]->set_cta_quota(quota);
            }

        } else if (usage_map.size() == 1) {
            // Case II: there's only one kernel in the system,
            // use existing quota if exists,
            // else use the max possible config
            auto begin = usage_map.begin();
            unsigned quota = m_running_kernels[begin->first]->get_cta_quota();

            if (quota == 0) {
                quota = std::floor(1.0 / begin->second.max_usage);
            }

            assert(quota > 0);
            m_running_kernels[begin->first]->set_cta_quota(quota);
        }

        break;
    }

    case intra_sm_option_t ::SMK:
    default: {
        calculate_smk_quota(usage_map);

        for (auto k_usage : usage_map) {
            const unsigned cta_quota = k_usage.second.cta_quota;

            assert(cta_quota > 0);
            m_running_kernels[k_usage.first]->set_cta_quota(cta_quota);
        }
        break;
    }
    }

    // Print cta quota settings
    float tot_smem = 0;
    for (auto k_usage : usage_map) {
        // print the resource partition results
        const kernel_info_t *p_kernel = m_running_kernels[k_usage.first];
        printf("Stream %d/%d (%s): %d ctas/SM\n", p_kernel->get_stream_id(),
               usage_map.size(), p_kernel->name().c_str(),
               p_kernel->get_cta_quota());

        tot_smem +=
            p_kernel->get_cta_quota() * p_kernel->get_usage().smem_usage;
    }

    // reset volta cache / shared mem config
    reset_volta_l1_cache(tot_smem);
}

void gpgpu_sim::reset_volta_l1_cache(float tot_smem) {
    const struct shader_core_config *shader_config = getShaderCoreConfig();
    if (shader_config->adaptive_volta_cache_config) {
        // For Volta, we assign the remaining shared memory to L1 cache
        // For more info, see
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
        unsigned total_shmed_bytes =
            ceil(tot_smem * shader_config->gpgpu_shmem_size);
        assert(total_shmed_bytes >= 0 &&
               total_shmed_bytes <= shader_config->gpgpu_shmem_size);

        if (total_shmed_bytes == 0)
            shader_config->m_L1D_config.set_assoc(256); // L1 is 128KB ans shd=0
        else if (total_shmed_bytes > 0 && total_shmed_bytes <= 8192)
            shader_config->m_L1D_config.set_assoc(
                240); // L1 is 120KB ans shd=8KB
        else if (total_shmed_bytes > 8192 && total_shmed_bytes <= 16384)
            shader_config->m_L1D_config.set_assoc(
                224); // L1 is 112KB ans shd=16KB
        else if (total_shmed_bytes > 16384 && total_shmed_bytes <= 32768)
            shader_config->m_L1D_config.set_assoc(
                192); // L1 is 96KB ans shd=32KB
        else if (total_shmed_bytes > 32768 && total_shmed_bytes <= 65536)
            shader_config->m_L1D_config.set_assoc(
                128); // L1 is 64KB ans shd=64KB
        else if (total_shmed_bytes > 65536 &&
                 total_shmed_bytes <= shader_config->gpgpu_shmem_size)
            shader_config->m_L1D_config.set_assoc(64); // L1 is 32KB and
                                                       // shd=96KB
        else
            assert(0);

        printf("GPGPU-Sim: Reconfigure L1 cache in Volta Archi to %uKB\n",
               shader_config->m_L1D_config.get_total_size_inKB());
    }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
    unsigned cta_size = kinfo->threads_per_cta();
    if (cta_size > m_shader_config->n_thread_per_shader) {
        printf("Execution error: Shader kernel CTA (block) size is too large "
               "for microarch config.\n");
        printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
               cta_size, m_shader_config->n_thread_per_shader);
        printf("                 => either change -gpgpu_shader argument in "
               "gpgpusim.config file or\n");
        printf("                 modify the CUDA source to decrease the kernel "
               "block size.\n");
        abort();
    }
    unsigned n = 0;
    for (n = 0; n < m_running_kernels.size(); n++) {
        if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
            m_running_kernels[n] = kinfo;
            kinfo->launch_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
            printf(
                "\n\n\nLaunching kernel %s @ sim_cycle: %d + total: %d!!\n\n\n",
                kinfo->name().c_str(), gpu_sim_cycle, gpu_tot_sim_cycle);

            // block the next kernel launch for default_launch_wait_cycle
            m_blocked_launch_cycle = m_config.delayed_cycle_btw_kernels;

            if (getShaderCoreConfig()->gpgpu_concurrent_kernel_sm &&
                (getShaderCoreConfig()->gpgpu_sharing_intra_sm
                 == sharing_option_t::INTRA)) {
                // call resource partitioning algorithm to update cta quota for
                // each kernel
                set_resource_config();
            }

            // resize the warp state stats if we still need to record
            // performance
            const unsigned stream_id = kinfo->get_stream_id();
            if (g_stream_manager->should_record_stat(stream_id)) {
                unsigned samples = kinfo->num_blocks() /
                                   m_shader_config->warp_state_sample_cta;
                if (kinfo->num_blocks() %
                        m_shader_config->warp_state_sample_cta !=
                    0) {
                    samples += 1;
                }

                // clear active warp sampler stats
                m_shader_stats->resize_warp_stats(stream_id, samples);

                // insert element for stats
                gpu_tot_sim_cycle_stream[stream_id].push_back(0);
                gpu_tot_sim_insn_stream[stream_id].push_back(0);
                partition_replys_total_per_stream[stream_id].push_back(0);
                m_memory_stats->mem_stats_stream[stream_id].push_back(
                    memory_stats_t::mem_stats_kidx_t());
            }

            break;
        }
    }
    assert(n < m_running_kernels.size());
}

unsigned gpgpu_sim::num_running_kernel() const {
    unsigned result = 0;
    for (unsigned n = 0; n < m_running_kernels.size(); n++) {
        if ((NULL != m_running_kernels[n]) && !(m_running_kernels[n]->done())) {
            result++;
        }
    }

    return result;
}

bool gpgpu_sim::can_start_kernel() {
    if (m_blocked_launch_cycle != 0) {
        return false;
    }

    for (unsigned n = 0; n < m_running_kernels.size(); n++) {
        if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
            return true;
    }
    return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
    if (m_config.gpu_max_cta_opt != 0) {
        if ((gpu_tot_issued_cta + m_total_cta_launched) >=
            m_config.gpu_max_cta_opt)
            return true;
    }
    return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
    if (hit_max_cta_count())
        return false;

    if (kernel && !kernel->no_more_ctas_to_run())
        return true;

    return false;
}

bool gpgpu_sim::get_more_cta_left() const {
    if (hit_max_cta_count())
        return false;

    for (unsigned n = 0; n < m_running_kernels.size(); n++) {
        if (m_running_kernels[n] &&
            !m_running_kernels[n]->no_more_ctas_to_run())
            return true;
    }
    return false;
}

kernel_info_t *gpgpu_sim::select_kernel() {
    // pick the current running kernel
    if (m_running_kernels[m_last_issued_kernel] &&
        !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run()) {
        unsigned launch_uid =
            m_running_kernels[m_last_issued_kernel]->get_uid();
        if (std::find(m_executed_kernel_uids.begin(),
                      m_executed_kernel_uids.end(),
                      launch_uid) == m_executed_kernel_uids.end()) {
            m_running_kernels[m_last_issued_kernel]->start_cycle =
                gpu_sim_cycle + gpu_tot_sim_cycle;
            m_executed_kernel_uids.push_back(launch_uid);
            m_executed_kernel_names.push_back(
                m_running_kernels[m_last_issued_kernel]->name());
        }
        return m_running_kernels[m_last_issued_kernel];
    }

    // pick a new kernel
    for (unsigned n = 0; n < m_running_kernels.size(); n++) {
        unsigned idx =
            (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
        if (kernel_more_cta_left(m_running_kernels[idx])) {
            m_last_issued_kernel = idx;
            m_running_kernels[idx]->start_cycle =
                gpu_sim_cycle + gpu_tot_sim_cycle;
            // record this kernel for stat print if it is the first time this
            // kernel is selected for execution
            unsigned launch_uid = m_running_kernels[idx]->get_uid();
            assert(std::find(m_executed_kernel_uids.begin(),
                             m_executed_kernel_uids.end(),
                             launch_uid) == m_executed_kernel_uids.end());
            m_executed_kernel_uids.push_back(launch_uid);
            m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

            return m_running_kernels[idx];
        }
    }
    return NULL;
}

void gpgpu_sim::update_executed_kernel(kernel_info_t *kernel) {
    unsigned launch_uid = kernel->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
        kernel->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
        m_executed_kernel_uids.push_back(launch_uid);
        m_executed_kernel_names.push_back(kernel->name());
    }
}

bool gpgpu_sim::print_at_device_sync() {
    return m_config.g_print_at_device_sync;
}

unsigned gpgpu_sim::finished_kernel() {
    if (m_finished_kernel.empty())
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
    return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel, bool has_completed) {
    unsigned uid = kernel->get_uid();
    m_finished_kernel.push_back(uid);
    std::vector<kernel_info_t *>::iterator k;
    for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
        if (*k == kernel) {
            kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
            *k = NULL;
            break;
        }
    }

    // call resource partition algorithm to update cta quota for the remaining
    // kernels
    set_resource_config();

    assert(k != m_running_kernels.end());

    const unsigned stream_id = kernel->get_stream_id();
    assert(stream_id < m_config.get_config_num_streams());

    if (has_completed && kernel->should_record_stat()) {
        gpu_tot_sim_cycle_stream[stream_id].back() =
            kernel->end_cycle - kernel->start_cycle;
    }

    m_shader_stats->collect_warp_state_stats(stream_id);

    printf(">>>>>>>> Stream %u kernel %s launched @ %llu, "
           "started @ %llu, ended @ %llu. \n",
           kernel->get_stream_id(),
           kernel->name().c_str(), kernel->launch_cycle, kernel->start_cycle,
           kernel->end_cycle);
}

void gpgpu_sim::stop_all_running_kernels() {
    std::vector<kernel_info_t *>::iterator k;
    for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
        if (*k != NULL) {               // If a kernel is active
            set_kernel_done(*k, false); // Stop the kernel
            assert(*k == NULL);
        }
    }
}

void set_ptx_warp_size(const struct core_config *warp_size);

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config)
    : gpgpu_t(config), m_config(config) {
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
    m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
        config.g_power_simulation_enabled, config.g_power_config_name);
#endif

    m_shader_stats = new shader_core_stats(m_shader_config,
                                           m_config.get_config_num_streams());
    m_memory_stats =
        new memory_stats_t(m_config.num_shader(), m_shader_config,
                           m_memory_config, m_config.get_config_num_streams());
    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms = (float *)malloc(sizeof(float));
    m_power_stats = new power_stat_t(
        m_shader_config, average_pipeline_duty_cycle, active_sms,
        m_shader_stats, m_memory_config, m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn_stream =
        new std::vector<unsigned long long>[m_config.get_config_num_streams()];
    gpu_tot_sim_cycle_stream =
        new std::vector<unsigned long long>[m_config.get_config_num_streams()];

    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    m_total_cta_launched = 0;
    gpu_deadlock = false;

    m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
        m_cluster[i] =
            new simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                  m_shader_stats, m_memory_stats);

    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        m_memory_partition_unit[i] =
            new memory_partition_unit(i, m_memory_config, m_memory_stats);
        for (unsigned p = 0;
             p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
            unsigned submpid =
                i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
            m_memory_sub_partition[submpid] =
                m_memory_partition_unit[i]->get_sub_partition(p);
        }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition,
                config.icnt_priority_per_stream);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout,
            "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize(config.max_concurrent_kernel, NULL);
    m_last_issued_kernel = 0;
    m_last_cluster_issue = m_shader_config->n_simt_clusters -
                           1; // this causes first launch to use simt cluster 0
    *average_pipeline_duty_cycle = 0;
    *active_sms = 0;

    last_liveness_message_time = 0;

    // Jin: functional simulation for CDP
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;

    m_blocked_launch_cycle = 0;
}

gpgpu_sim::~gpgpu_sim() { delete partition_replys_total_per_stream; }

int gpgpu_sim::shared_mem_size() const {
    return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
    return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
    return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
    return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
    return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
    return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
    return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
    return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
    sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
           &l2_freq, &dram_freq);
    core_freq = core_freq MhZ;
    icnt_freq = icnt_freq MhZ;
    l2_freq = l2_freq MhZ;
    dram_freq = dram_freq MhZ;
    core_period = 1 / core_freq;
    icnt_period = 1 / icnt_freq;
    dram_period = 1 / dram_freq;
    l2_period = 1 / l2_freq;
    printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
           icnt_freq, l2_freq, dram_freq);
    printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
           core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
    core_time = 0;
    dram_time = 0;
    icnt_time = 0;
    l2_time = 0;
}

bool gpgpu_sim::active() {
    if (m_config.gpu_max_cycle_opt &&
        (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
        return false;
    if (m_config.gpu_max_insn_opt &&
        (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
        return false;
    if (m_config.gpu_max_cta_opt &&
        (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
        return false;
    if (m_config.gpu_deadlock_detect && gpu_deadlock)
        return false;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
        if (m_cluster[i]->get_not_completed() > 0)
            return true;
    ;
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
        if (m_memory_partition_unit[i]->busy() > 0)
            return true;
    ;
    if (icnt_busy())
        return true;
    if (get_more_cta_left())
        return true;
    return false;
}

void gpgpu_sim::init() {
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    for (int i = 0; i < m_config.get_config_num_streams(); i++) {
        gpu_tot_sim_insn_stream[i].reserve(5);
        gpu_tot_sim_cycle_stream[i].reserve(5);
    }
    last_gpu_sim_insn = 0;
    m_total_cta_launched = 0;
    partiton_reqs_in_parallel = 0;
    partiton_replys_in_parallel = 0;
    partiton_reqs_in_parallel_util = 0;
    gpu_sim_cycle_parition_util = 0;

    partition_replys_total_per_stream =
        new std::vector<unsigned long long>[m_config.get_config_num_streams()];
    for (unsigned stream_id = 0; stream_id < m_config.get_config_num_streams();
         stream_id++) {
        partition_replys_total_per_stream[stream_id].reserve(5);
    }

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
        m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger(m_config.num_shader(),
                               m_shader_config->n_thread_per_shader, 0,
                               m_config.gpgpu_cflog_interval);
    }
    shader_CTA_count_create(m_config.num_shader(),
                            m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
        insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
        shader_warp_occ_create(m_config.num_shader(),
                               m_shader_config->warp_size,
                               m_config.gpgpu_cflog_interval);
        shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem,
                              4, m_config.gpgpu_cflog_interval);
        shader_mem_lat_create(m_config.num_shader(),
                              m_config.gpgpu_cflog_interval);
        shader_cache_access_create(m_config.num_shader(), 3,
                                   m_config.gpgpu_cflog_interval);
        set_spill_interval(m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode)
        icnt_init();

    if (getShaderCoreConfig()->gpgpu_concurrent_kernel_sm) {
        // set cache config now to default
        change_cache_config(FuncCachePreferNone);
    }

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
        init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
                   gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif
}

void gpgpu_sim::update_stats() {
    m_memory_stats->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
    gpu_tot_issued_cta += m_total_cta_launched;
    partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
    partiton_replys_in_parallel_total += partiton_replys_in_parallel;
    partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
    gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
    gpu_tot_occupancy += gpu_occupancy;

    gpu_sim_cycle = 0;
    partiton_reqs_in_parallel = 0;
    partiton_replys_in_parallel = 0;
    partiton_reqs_in_parallel_util = 0;
    gpu_sim_cycle_parition_util = 0;
    gpu_sim_insn = 0;
    m_total_cta_launched = 0;
    gpu_occupancy = occupancy_stats();
}

void gpgpu_sim::print_stats() {
    ptx_file_line_stats_write_file();
    gpu_print_stat();

    if (g_network_mode) {
        printf("----------------------------Interconnect-DETAILS---------------"
               "-----------------\n");
        icnt_display_stats();
        //        icnt_display_overall_stats();
        printf("----------------------------END-of-Interconnect-DETAILS--------"
               "-----------------\n");
    }
}

void gpgpu_sim::deadlock_check() {
    if (m_config.gpu_deadlock_detect && gpu_deadlock) {
        fflush(stdout);
        printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last "
               "writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) "
               "(%u cycles ago)\n",
               gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
               (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
               (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
        unsigned num_cores = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            unsigned not_completed = m_cluster[i]->get_not_completed();
            if (not_completed) {
                if (!num_cores) {
                    printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer "
                           "committing instructions [core(# threads)]:\n");
                    printf("GPGPU-Sim uArch: DEADLOCK  ");
                    m_cluster[i]->print_not_completed(stdout);
                } else if (num_cores < 8) {
                    m_cluster[i]->print_not_completed(stdout);
                } else if (num_cores >= 8) {
                    printf(" + others ... ");
                }
                num_cores += m_shader_config->n_simt_cores_per_cluster;
            }
        }
        printf("\n");
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            bool busy = m_memory_partition_unit[i]->busy();
            if (busy)
                printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n",
                       i);
        }
        if (icnt_busy()) {
            printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
            icnt_display_state(stdout);
        }
        printf("\nRe-run the simulator in gdb and use debug routines in "
               ".gdbinit to debug this\n");
        fflush(stdout);
        abort();
    }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
    std::stringstream statout;

    statout << "kernel_name = ";
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
        statout << m_executed_kernel_names[k] << " ";
    }
    statout << std::endl;
    statout << "kernel_launch_uid = ";
    for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
        statout << m_executed_kernel_uids[k] << " ";
    }
    statout << std::endl;

    return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
    m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
    for (std::map<std::string, FuncCache>::iterator iter =
             m_special_cache_config.begin();
         iter != m_special_cache_config.end(); iter++) {
        std::string kernel = iter->first;
        if (kernel_name.compare(kernel) == 0) {
            return iter->second;
        }
    }
    return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
    for (std::map<std::string, FuncCache>::iterator iter =
             m_special_cache_config.begin();
         iter != m_special_cache_config.end(); iter++) {
        std::string kernel = iter->first;
        if (kernel_name.compare(kernel) == 0) {
            return true;
        }
    }
    return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
    if (has_special_cache_config(kernel_name)) {
        change_cache_config(get_cache_config(kernel_name));
    } else {
        change_cache_config(FuncCachePreferNone);
    }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
    if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
        printf("FLUSH L1 Cache at configuration change between kernels\n");
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            m_cluster[i]->cache_flush();
        }
    }

    switch (cache_config) {
    case FuncCachePreferNone:
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
        break;
    case FuncCachePreferL1:
        if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
            (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
            printf("WARNING: missing Preferred L1 configuration\n");
            m_shader_config->m_L1D_config.init(
                m_shader_config->m_L1D_config.m_config_string,
                FuncCachePreferNone);
            m_shader_config->gpgpu_shmem_size =
                m_shader_config->gpgpu_shmem_sizeDefault;

        } else {
            m_shader_config->m_L1D_config.init(
                m_shader_config->m_L1D_config.m_config_stringPrefL1,
                FuncCachePreferL1);
            m_shader_config->gpgpu_shmem_size =
                m_shader_config->gpgpu_shmem_sizePrefL1;
        }
        break;
    case FuncCachePreferShared:
        if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
            (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
            printf("WARNING: missing Preferred L1 configuration\n");
            m_shader_config->m_L1D_config.init(
                m_shader_config->m_L1D_config.m_config_string,
                FuncCachePreferNone);
            m_shader_config->gpgpu_shmem_size =
                m_shader_config->gpgpu_shmem_sizeDefault;
        } else {
            m_shader_config->m_L1D_config.init(
                m_shader_config->m_L1D_config.m_config_stringPrefShared,
                FuncCachePreferShared);
            m_shader_config->gpgpu_shmem_size =
                m_shader_config->gpgpu_shmem_sizePrefShared;
        }
        break;
    default:
        break;
    }
}

void gpgpu_sim::clear_executed_kernel_info() {
    m_executed_kernel_names.clear();
    m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
    FILE *statfout = stdout;

    std::string kernel_info_str = executed_kernel_info_string();
    fprintf(statfout, "%s", kernel_info_str.c_str());

    printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
    printf("gpu_sim_insn = %lld\n", gpu_sim_insn);

    printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
    printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
    for (unsigned i = 0; i < m_config.get_config_num_streams(); i++) {
        for (unsigned kidx = 0; kidx < gpu_tot_sim_cycle_stream[i].size();
             kidx++) {
            printf("gpu_tot_sim_cycle[%u][%u]: %lld\n", i, kidx,
                   gpu_tot_sim_cycle_stream[i][kidx]);
        }
    }
    printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
    for (unsigned i = 0; i < m_config.get_config_num_streams(); i++) {
        for (unsigned kidx = 0; kidx < gpu_tot_sim_insn_stream[i].size();
             kidx++) {
            printf("gpu_tot_sim_insn[%u][%u]: %lld\n", i, kidx,
                   gpu_tot_sim_insn_stream[i][kidx]);
        }
    }
    printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                         (gpu_tot_sim_cycle + gpu_sim_cycle));
    printf("gpu_tot_issued_cta = %lld\n",
           gpu_tot_issued_cta + m_total_cta_launched);
    printf("gpu_occupancy = %.4f\% \n", gpu_occupancy.get_occ_fraction() * 100);
    printf("gpu_tot_occupancy = %.4f\% \n",
           (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

    extern unsigned long long g_max_total_param_size;
    fprintf(statfout, "max_total_param_size = %llu\n", g_max_total_param_size);

    // performance counter for stalls due to congestion.
    printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
    printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

    // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
    // printf("partiton_reqs_in_parallel_total    = %lld\n",
    // partiton_reqs_in_parallel_total );
    printf("partiton_level_parallism = %12.4f\n",
           (float)partiton_reqs_in_parallel / gpu_sim_cycle);
    printf(
        "partiton_level_parallism_total  = %12.4f\n",
        (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
            (gpu_tot_sim_cycle + gpu_sim_cycle));
    // printf("partiton_reqs_in_parallel_util = %lld\n",
    // partiton_reqs_in_parallel_util);
    // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
    // partiton_reqs_in_parallel_util_total );
    // printf("gpu_sim_cycle_parition_util = %lld\n",
    // gpu_sim_cycle_parition_util);
    // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
    // gpu_tot_sim_cycle_parition_util );
    printf("partiton_level_parallism_util = %12.4f\n",
           (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
    printf("partiton_level_parallism_util_total  = %12.4f\n",
           (float)(partiton_reqs_in_parallel_util +
                   partiton_reqs_in_parallel_util_total) /
               (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
    // printf("partiton_replys_in_parallel = %lld\n",
    // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
    // %lld\n", partiton_replys_in_parallel_total );
    printf("L2_BW  = %12.4f GB/Sec\n",
           ((float)(partiton_replys_in_parallel * 32) /
            (gpu_sim_cycle * m_config.icnt_period)) /
               1000000000);

    double seconds = (gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period;
    printf("L2_BW_total  = %12.4f GB/Sec\n",
           ((float)((partiton_replys_in_parallel +
                     partiton_replys_in_parallel_total) *
                    32) /
            seconds) /
               1000000000);

    // L2_BW per stream
    for (unsigned stream_id = 0; stream_id < m_config.get_config_num_streams();
         stream_id++) {
        for (unsigned kidx = 0;
             kidx < partition_replys_total_per_stream[stream_id].size();
             kidx++) {
            const double k_seconds = gpu_tot_sim_cycle_stream[stream_id][kidx] *
                                     m_config.icnt_period;
            const float bw =
                ((partition_replys_total_per_stream[stream_id][kidx] * 32) /
                 k_seconds) /
                1000000000;
            printf("L2_BW_total[%u][%u] = %12.4f GB/Sec\n", stream_id, kidx,
                   bw);
        }
    }

    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - g_simulation_starttime, 1);
    printf("gpu_total_sim_rate=%u\n",
           (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

    // shader_print_l1_miss_stat( stdout );
    shader_print_cache_stats(stdout);

    cache_stats core_cache_stats;
    core_cache_stats.clear();
    for (unsigned i = 0; i < m_config.num_cluster(); i++) {
        m_cluster[i]->get_cache_stats(core_cache_stats);
    }
    printf("\nTotal_core_cache_stats:\n");
    core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
    printf("\nTotal_core_cache_fail_stats:\n");
    core_cache_stats.print_fail_stats(stdout,
                                      "Total_core_cache_fail_stats_breakdown");
    shader_print_scheduler_stat(stdout, false);

    m_shader_stats->print(stdout);

#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
        m_gpgpusim_wrapper->print_power_kernel_stats(
            gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
            kernel_info_str, true);
        mcpat_reset_perf_count(m_gpgpusim_wrapper);
    }
#endif

    // performance counter that are not local to one shader
    m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                     m_memory_config->nbk);
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
        m_memory_partition_unit[i]->print(stdout);

    // L2 cache stats
    if (!m_memory_config->m_L2_config.disabled()) {
        cache_stats l2_stats;
        struct cache_sub_stats l2_css;
        struct cache_sub_stats total_l2_css;
        l2_stats.clear();
        l2_css.clear();
        total_l2_css.clear();

        printf("\n========= L2 cache stats =========\n");
        for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
            m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

            fprintf(stdout,
                    "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate "
                    "= %.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
                    i, l2_css.accesses, l2_css.misses,
                    (double)l2_css.misses / (double)l2_css.accesses,
                    l2_css.pending_hits, l2_css.res_fails);

            total_l2_css += l2_css;
        }
        if (!m_memory_config->m_L2_config.disabled() &&
            m_memory_config->m_L2_config.get_num_lines()) {
            // L2c_print_cache_stat();
            printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
            printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
            if (total_l2_css.accesses > 0)
                printf("L2_total_cache_miss_rate = %.4lf\n",
                       (double)total_l2_css.misses /
                           (double)total_l2_css.accesses);
            printf("L2_total_cache_pending_hits = %llu\n",
                   total_l2_css.pending_hits);
            printf("L2_total_cache_reservation_fails = %llu\n",
                   total_l2_css.res_fails);
            printf("L2_total_cache_breakdown:\n");
            l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
            printf("L2_total_cache_reservation_fail_breakdown:\n");
            l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
            total_l2_css.print_port_stats(stdout, "L2_cache");
        }
    }

    if (m_config.gpgpu_cflog_interval != 0) {
        spill_log_to_file(stdout, 1, gpu_sim_cycle);
        insn_warp_occ_print(stdout);
    }
    if (gpgpu_ptx_instruction_classification) {
        StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
        StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
    }

#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
        m_gpgpusim_wrapper->detect_print_steady_state(1, gpu_tot_sim_insn +
                                                             gpu_sim_insn);
    }
#endif

    // Interconnect power stat print
    long total_simt_to_mem = 0;
    long total_mem_to_simt = 0;
    long temp_stm = 0;
    long temp_mts = 0;
    for (unsigned i = 0; i < m_config.num_cluster(); i++) {
        m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
        total_simt_to_mem += temp_stm;
        total_mem_to_simt += temp_mts;
    }
    printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
    printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

    time_vector_print();
    fflush(stdout);

    clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
    return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
    unsigned active_count = inst.active_count();
    // this breaks some encapsulation: the is_[space] functions, if you change
    // those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count;
        break;
    case sstarr_space:
        m_stats->gpgpu_n_sstarr_insn += active_count;
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if (inst.is_store())
            m_stats->gpgpu_n_store_insn += active_count;
        else
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {

    // Jin: concurrent kernels on one SM
    if (m_config->gpgpu_concurrent_kernel_sm) {
        if (m_config->max_cta(kernel) < 1)
            return false;

        return occupy_shader_resource_1block(kernel, false);
    } else {
        return (get_n_active_cta() < m_config->max_cta(kernel));
    }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy,
                                          bool from_top) {

    int step;
    bool found = false;

    if (from_top) {
        for (step = 0; (step + cta_size) < m_config->n_thread_per_shader;
             step += cta_size) {

            unsigned int hw_tid;
            for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
                if (m_occupied_hwtid.test(hw_tid))
                    break;
            }
            if (hw_tid == step + cta_size) {
                // consecutive non-active
                found = true;
                break;
            }
        }
    } else {
        for (step = m_config->n_thread_per_shader - cta_size; step >= 0;
             step -= cta_size) {

            unsigned int hw_tid;
            for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
                if (m_occupied_hwtid.test(hw_tid))
                    break;
            }
            if (hw_tid == step + cta_size) {
                // consecutive non-active
                found = true;
                break;
            }
        }
    }

    if (!found) {
        // didn't find
        return -1;
    } else {
        if (occupy) {
            for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
                m_occupied_hwtid.set(hw_tid);
            }
        }
        return step;
    }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
        padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
        return false;

    if (find_available_hwtid(padded_cta_size, false, k.allocate_from_top()) ==
        -1)
        return false;

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
        return false;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
        return false;

    if (m_occupied_ctas + 1 > m_config->max_cta_per_core)
        return false;

    if (occupy) {
        m_occupied_n_threads += padded_cta_size;
        m_occupied_shmem += kernel_info->smem;
        m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
        m_occupied_ctas++;

        SHADER_DPRINTF(LIVENESS,
                       "GPGPU-Sim uArch: Occupied %d threads, %d shared mem, "
                       "%d registers, %d ctas\n",
                       m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                       m_occupied_ctas);
    }

    return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {

    if (m_config->gpgpu_concurrent_kernel_sm) {
        unsigned threads_per_cta = k.threads_per_cta();
        const class function_info *kernel = k.entry();
        unsigned int padded_cta_size = threads_per_cta;
        unsigned int warp_size = m_config->warp_size;
        if (padded_cta_size % warp_size)
            padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

        assert(m_occupied_n_threads >= padded_cta_size);
        m_occupied_n_threads -= padded_cta_size;

        int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

        for (unsigned hwtid = start_thread;
             hwtid < start_thread + padded_cta_size; hwtid++)
            m_occupied_hwtid.reset(hwtid);
        m_occupied_cta_to_hwtid.erase(hw_ctaid);

        const struct gpgpu_ptx_sim_info *kernel_info =
            ptx_sim_kernel_info(kernel);

        assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
        m_occupied_shmem -= kernel_info->smem;

        unsigned int used_regs =
            padded_cta_size * ((kernel_info->regs + 3) & ~3);
        assert(m_occupied_regs >= used_regs);
        m_occupied_regs -= used_regs;

        assert(m_occupied_ctas >= 1);
        m_occupied_ctas--;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {

    if (!m_config->gpgpu_concurrent_kernel_sm)
        set_max_cta(kernel);
    else
        assert(occupy_shader_resource_1block(kernel, true));

    kernel.inc_running();

    // find a free CTA context
    unsigned free_cta_hw_id = (unsigned)-1;

    unsigned max_cta_per_core;
    if (!m_config->gpgpu_concurrent_kernel_sm) {
        max_cta_per_core = kernel_max_cta_per_shader;

        for (unsigned i = 0; i < max_cta_per_core; i++) {
            if (m_cta_status[i] == 0) {
                free_cta_hw_id = i;
                break;
            }
        }
    } else {
        // max number of ctas is limited by hardware slots when concurrent
        // kernels
        max_cta_per_core = m_config->max_cta_per_core;

        // cta slot 2-way allocation
        if (kernel.allocate_from_top()) {
            for (unsigned i = 0; i < max_cta_per_core; i++) {
                if (m_cta_status[i] == 0) {
                    free_cta_hw_id = i;
                    break;
                }
            }
        } else {
            for (unsigned i = max_cta_per_core - 1; i >= 0; i--) {
                if (m_cta_status[i] == 0) {
                    free_cta_hw_id = i;
                    break;
                }
            }
        }
    }
    assert(free_cta_hw_id != (unsigned)-1);

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size;
    if (cta_size % m_config->warp_size)
        padded_cta_size =
            ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

    unsigned int start_thread, end_thread;

    if (!m_config->gpgpu_concurrent_kernel_sm) {
        start_thread = free_cta_hw_id * padded_cta_size;
        end_thread = start_thread + cta_size;
    } else {
        start_thread = find_available_hwtid(padded_cta_size, true,
                                            kernel.allocate_from_top());
        assert((int)start_thread != -1);
        end_thread = start_thread + cta_size;
        assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
               m_occupied_cta_to_hwtid.end());
        m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
    }

    // reset the microarchitecture state of the selected hardware thread and
    // warp contexts
    reinit(start_thread, end_thread, false);

    // initalize scalar threads and determine which hardware warps they are
    // allocated to bind functional simulation state of threads to hardware
    // resources (simulation)
    warp_set_t warps;
    unsigned nthreads_in_block = 0;
    function_info *kernel_func_info = kernel.entry();
    symbol_table *symtab = kernel_func_info->get_symtab();
    unsigned ctaid = kernel.get_next_cta_id_single();
    dim3 cta_id3d = kernel.get_next_cta_id();
    checkpoint *g_checkpoint = new checkpoint();
    for (unsigned i = start_thread; i < end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i / m_config->warp_size;

        nthreads_in_block += ptx_sim_init_thread(
            kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
            m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
            m_cluster->get_gpu());

        m_threadState[i].m_active = true;

        // load thread local memory and register file
        // checkpoint?
        if (m_gpu->resume_option == 1 &&
            kernel.get_uid() == m_gpu->resume_kernel &&
            ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
            char fname[2048];
            snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
                     i % cta_size, ctaid);
            m_thread[i]->resume_reg_thread(fname, symtab);
            char f1name[2048];
            snprintf(f1name, 2048,
                     "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
                     i % cta_size, ctaid);
            g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
        }

        // preempted?
        if (kernel.has_preempted_cta()) {
            preempted_cta_context context = kernel.m_preempted_queue.front();
            unsigned tid_in_cta = i - start_thread;

            if (context.retired_threads[tid_in_cta]) {
                m_thread[i]->set_done();
                m_thread[i]->exitCore();
                m_thread[i]->registerExit();

                --nthreads_in_block;
                m_threadState[i].m_active = false;
            } else {
                m_thread[i]->set_local_mem_stack_pointer(
                    context.local_mem_stack_pointer[tid_in_cta]);
                m_thread[i]->resume_reg_thread_strbuf(context.regs[tid_in_cta],
                                                      symtab);
                m_thread[i]->m_local_mem->load(context.local_mem[tid_in_cta]);
                m_thread[i]->set_npc(context.pcs[tid_in_cta]);
                m_thread[i]->update_pc();
            }
        }

        warps.set(warp_id);
    }

    // should be at least one, but less than max
    assert(nthreads_in_block > 0 &&
           nthreads_in_block <= m_config->n_thread_per_shader);
    m_cta_status[free_cta_hw_id] = nthreads_in_block;

    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char f1name[2048];
        snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

        g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem,
                                      f1name);
    }

    if (kernel.has_preempted_cta()) {
        preempted_cta_context context = kernel.m_preempted_queue.front();
        m_thread[start_thread]->m_shared_mem->load(context.shared_mem);

#ifdef TIMELINE_ON
        printf("TIMELINE: Restore preempted kernel %d cta %d,%d,%d on shader "
               "%d @ cycle %d\n",
               kernel.get_uid(), cta_id3d.x, cta_id3d.y, cta_id3d.z, get_sid(),
               gpu_sim_cycle + gpu_tot_sim_cycle);
    } else {

        printf(
            "TIMELINE: Launch kernel %d cta %d,%d,%d on shader %d @ cycle %d\n",
            kernel.get_uid(), cta_id3d.x, cta_id3d.y, cta_id3d.z, get_sid(),
            gpu_sim_cycle + gpu_tot_sim_cycle);
#endif
    }

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id, warps);

    if (kernel.has_preempted_cta()) {
        preempted_cta_context context = kernel.m_preempted_queue.front();
        m_barriers.restore_preempted_context(free_cta_hw_id, context);
    }

    // initialize the SIMT stacks and fetch hardware
    init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size,
               &kernel);
    m_n_active_cta++;

    // store kernel to cta info
    unsigned kernel_id = kernel.get_uid();
    if (m_kernel2ctas.find(kernel_id) == m_kernel2ctas.end()) {
        // this core is running this kernel for the first time
        m_kernel2ctas[kernel_id] = std::vector<unsigned>();
        m_kernel2ctas[kernel_id].reserve(m_config->max_cta_per_core);
        m_kernel2ctas[kernel_id].push_back(free_cta_hw_id);
    } else {
        m_kernel2ctas[kernel_id].push_back(free_cta_hw_id);
    }

    shader_CTA_count_log(m_sid, 1);
    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                   "initialized @(%lld,%lld)\n",
                   free_cta_hw_id, start_thread, end_thread, gpu_sim_cycle,
                   gpu_tot_sim_cycle);

    kernel.increment_cta_id();
    kernel.dec_pending();

    // update shader resource usage
    shader_usage.cta_usage += kernel.get_usage().cta_usage;
    shader_usage.reg_usage += kernel.get_usage().reg_usage;
    shader_usage.smem_usage += kernel.get_usage().smem_usage;
    shader_usage.thread_usage += kernel.get_usage().thread_usage;
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
    if (task == SAMPLELOG) {
        StatAddSample(mrqq_Dist, que_length());
    } else if (task == DUMPLOG) {
        printf("Queue Length DRAM[%d] ", id);
        StatDisp(mrqq_Dist);
    }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
    double smallest = min3(core_time, icnt_time, dram_time);
    int mask = 0x00;
    if (l2_time <= smallest) {
        smallest = l2_time;
        mask |= L2;
        l2_time += m_config.l2_period;
    }
    if (icnt_time <= smallest) {
        mask |= ICNT;
        icnt_time += m_config.icnt_period;
    }
    if (dram_time <= smallest) {
        mask |= DRAM;
        dram_time += m_config.dram_period;
    }
    if (core_time <= smallest) {
        mask |= CORE;
        core_time += m_config.core_period;
    }
    return mask;
}

void gpgpu_sim::issue_block2core() {
    unsigned last_issued = m_last_cluster_issue;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
        unsigned num = m_cluster[idx]->issue_block2core();
        if (num) {
            m_last_cluster_issue = idx;
            m_total_cta_launched += num;
        }
    }

    // SMK block issue algorithm: sort simt clusters according to their resource
    // usage pick the lowest usage first
    //	std::vector<simt_core_cluster*> clusters(m_cluster,
    //m_cluster+m_shader_config->n_simt_clusters);
    //
    //	std::sort(clusters.begin(), clusters.end(),
    //	    [](const simt_core_cluster* a, const simt_core_cluster* b) -> bool
    //	{
    //	    return a->get_core_min_usage() < b->get_core_min_usage();
    //	});
    //
    //	for (auto c : clusters) {
    //		unsigned num = c->issue_block2core();
    //		if (num) {
    //			m_total_cta_launched += num;
    //		}
    //	}
}

unsigned long long g_single_step =
    0; // set this in gdb to single step the pipeline

void gpgpu_sim::cycle() {
    int clock_mask = next_clock_domain();

    if (clock_mask & CORE) {
        // shader core loading (pop from ICNT into core) follows CORE clock
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
            m_cluster[i]->icnt_cycle();
    }
    unsigned partiton_replys_in_parallel_per_cycle = 0;

    unsigned partiton_replys_in_parallel_per_cycle_stream
        [m_config.get_config_num_streams()];
    memset(partiton_replys_in_parallel_per_cycle_stream, 0,
           sizeof(unsigned) * m_config.get_config_num_streams());

    if (clock_mask & ICNT) {
        // pop from memory controller to interconnect
        for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            mem_fetch *mf = m_memory_sub_partition[i]->top();
            if (mf) {
                unsigned response_size =
                    mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
                if (::icnt_has_buffer(m_shader_config->mem2device(i),
                                      response_size)) {
                    mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                    mf->set_status(IN_ICNT_TO_SHADER,
                                   gpu_sim_cycle + gpu_tot_sim_cycle);
                    ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(),
                                mf, response_size);
                    m_memory_sub_partition[i]->pop();
                    partiton_replys_in_parallel_per_cycle++;

                    unsigned stream_id = mf->get_stream_id();
                    assert(stream_id < m_config.get_config_num_streams());
                    partiton_replys_in_parallel_per_cycle_stream[stream_id]++;
                } else {
                    gpu_stall_icnt2sh++;
                }
            } else {
                m_memory_sub_partition[i]->pop();
            }
        }
    }

    partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

    for (unsigned stream_id = 0; stream_id < m_config.get_config_num_streams();
         stream_id++) {
        partition_replys_total_per_stream[stream_id].back() +=
            partiton_replys_in_parallel_per_cycle_stream[stream_id];
    }

    if (clock_mask & DRAM) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            m_memory_partition_unit[i]
                ->dram_cycle(); // Issue the dram command (scheduler + delay
                                // model)
            // Update performance counters for DRAM
            m_memory_partition_unit[i]->set_dram_power_stats(
                m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
        }
    }

    // L2 operations follow L2 clock domain
    unsigned partiton_reqs_in_parallel_per_cycle = 0;
    if (clock_mask & L2) {
        m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
        for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            // move memory request from interconnect into memory partition (if
            // not backed up) Note:This needs to be called in DRAM clock domain
            // if there is no L2 cache in the system In the worst case, we may
            // need to push SECTOR_CHUNCK_SIZE requests, so ensure you have
            // enough buffer for them
            if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
                gpu_stall_dramfull++;
            } else {
                // LSRR scheduling
                for (unsigned stream_id = 1;
                     stream_id <= m_config.get_config_num_streams();
                     stream_id++) {
                    unsigned turn =
                        (m_memory_sub_partition[i]->get_icnt_L2_turn_stream() +
                         stream_id) %
                        m_config.get_config_num_streams();

                    mem_fetch *mf = (mem_fetch *)icnt_pop(
                        m_shader_config->mem2device(i), turn);
                    m_memory_sub_partition[i]->push(mf, gpu_sim_cycle +
                                                            gpu_tot_sim_cycle);
                    if (mf) {
                        partiton_reqs_in_parallel_per_cycle++;
                        m_memory_sub_partition[i]->set_icnt_L2_turn_stream(
                            turn);

                        break; // break out of the loose round robin loop
                    }
                }
            }

            m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle +
                                                   gpu_tot_sim_cycle);
            m_memory_sub_partition[i]->accumulate_L2cache_stats(
                m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
        }
    }
    partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
    if (partiton_reqs_in_parallel_per_cycle > 0) {
        partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        gpu_sim_cycle_parition_util++;
    }

    if (clock_mask & ICNT) {
        icnt_transfer();
    }

    if (clock_mask & CORE) {
        // L1 cache + shader core pipeline stages
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
                m_cluster[i]->core_cycle();
                *active_sms += m_cluster[i]->get_n_active_sms();
            }
            // Update core icnt/cache stats for GPUWattch
            m_cluster[i]->get_icnt_stats(
                m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
                m_power_stats->pwr_mem_stat
                    ->n_mem_to_simt[CURRENT_STAT_IDX][i]);
            m_cluster[i]->get_cache_stats(
                m_power_stats->pwr_mem_stat
                    ->core_cache_stats[CURRENT_STAT_IDX]);
            m_cluster[i]->get_current_occupancy(
                gpu_occupancy.aggregate_warp_slot_filled,
                gpu_occupancy.aggregate_theoretical_warp_slots);
        }
        float temp = 0;
        for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
            temp += m_shader_stats->m_pipeline_duty_cycle[i];
        }
        temp = temp / m_shader_config->num_shader();
        *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
        // cout<<"Average pipeline duty cycle:
        // "<<*average_pipeline_duty_cycle<<endl;

        if (g_single_step &&
            ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
            raise(SIGTRAP); // Debug breakpoint
        }
        gpu_sim_cycle++;
        dec_wait_cycle();

        if (g_interactive_debugger_enabled)
            gpgpu_debug();

            // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
        if (m_config.g_power_simulation_enabled) {
            mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                        m_power_stats, m_config.gpu_stat_sample_freq,
                        gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                        gpu_sim_insn);
        }
#endif

        issue_block2core();

        // Depending on configuration, invalidate the caches once all of threads
        // are completed.
        int all_threads_complete = 1;
        if (m_config.gpgpu_flush_l1_cache) {
            for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
                if (m_cluster[i]->get_not_completed() == 0)
                    m_cluster[i]->cache_invalidate();
                else
                    all_threads_complete = 0;
            }
        }

        if (m_config.gpgpu_flush_l2_cache) {
            if (!m_config.gpgpu_flush_l1_cache) {
                for (unsigned i = 0; i < m_shader_config->n_simt_clusters;
                     i++) {
                    if (m_cluster[i]->get_not_completed() != 0) {
                        all_threads_complete = 0;
                        break;
                    }
                }
            }

            if (all_threads_complete &&
                !m_memory_config->m_L2_config.disabled()) {
                printf("Flushed L2 caches...\n");
                if (m_memory_config->m_L2_config.get_num_lines()) {
                    int dlc = 0;
                    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
                        dlc = m_memory_sub_partition[i]->flushL2();
                        assert(dlc == 0); // TODO: need to model actual writes
                                          // to DRAM here
                        printf("Dirty lines flushed from L2 %d is %d\n", i,
                               dlc);
                    }
                }
            }
        }

        if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
            time_t days, hrs, minutes, sec;
            time_t curr_time;
            time(&curr_time);
            unsigned long long elapsed_time =
                MAX(curr_time - g_simulation_starttime, 1);
            if ((elapsed_time - last_liveness_message_time) >=
                    m_config.liveness_message_freq &&
                DTRACE(LIVENESS)) {
                days = elapsed_time / (3600 * 24);
                hrs = elapsed_time / 3600 - 24 * days;
                minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
                sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

                unsigned long long active = 0, total = 0;
                for (unsigned i = 0; i < m_shader_config->n_simt_clusters;
                     i++) {
                    m_cluster[i]->get_current_occupancy(active, total);
                }
                DPRINTF(
                    LIVENESS,
                    "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f\% [%llu / %llu]) "
                    "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                    gpu_tot_sim_insn + gpu_sim_insn,
                    (double)gpu_sim_insn / (double)gpu_sim_cycle,
                    float(active) / float(total) * 100, active, total,
                    (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) /
                               elapsed_time),
                    (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                    (unsigned)sec, ctime(&curr_time));
                fflush(stdout);
                last_liveness_message_time = elapsed_time;
            }
            visualizer_printstat();
            m_memory_stats->memlatstat_lat_pw();
            if (m_config.gpgpu_runtime_stat &&
                (m_config.gpu_runtime_stat_flag != 0)) {
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
                    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
                        m_memory_partition_unit[i]->print_stat(stdout);
                    printf("maxmrqlatency = %d \n",
                           m_memory_stats->max_mrq_latency);
                    printf("maxmflatency = %d \n",
                           m_memory_stats->max_mf_latency);
                }
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
                    shader_print_runtime_stat(stdout);
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
                    shader_print_l1_miss_stat(stdout);
                if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
                    shader_print_scheduler_stat(stdout, false);
            }
        }

        if (!(gpu_sim_cycle % 50000)) {
            // deadlock detection
            if (m_config.gpu_deadlock_detect &&
                gpu_sim_insn == last_gpu_sim_insn) {
                gpu_deadlock = true;
            } else {
                last_gpu_sim_insn = gpu_sim_insn;
            }
        }
        try_snap_shot(gpu_sim_cycle);
        spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
        // launch device kernel
        launch_one_device_kernel();
#endif
    }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
    fprintf(fout, "\n");
    fprintf(fout, "per warp functional simulation status:\n");
    for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
        m_warp[w].print(fout);
}

bool shader_core_ctx::should_preempt_kernel(kernel_info_t *&victim,
                                            kernel_info_t *&candidate) {
    bool found_candidate = false;
    bool found_victim = false;

    const std::vector<kernel_info_t *> &running_kernels =
        m_gpu->get_running_kernels();

    // Candidate kernel is the first kernel in the list that 1) has more ctas
    // 2) running ctas is less than its quota
    // Victim kernel is the first kernel in the list that has more running ctas
    // than quota
    for (unsigned idx = 0; idx < running_kernels.size(); idx++) {
        kernel_info_t *current_kernel = running_kernels[idx];

        if (!current_kernel || current_kernel->done()) {
            continue;
        }

        // calculate normalized cta usage
        const unsigned running_cta =
            m_kernel2ctas.count(current_kernel->get_uid()) == 0
                ? 0
                : m_kernel2ctas[current_kernel->get_uid()].size();

        // always pick the kernel with the least number of running ctas
        if (!found_candidate && m_gpu->kernel_more_cta_left(current_kernel) &&
            running_cta < current_kernel->get_cta_quota()) {
            candidate = current_kernel;
            found_candidate = true;
        }

        if (!found_victim && running_cta > current_kernel->get_cta_quota()) {
            victim = current_kernel;
            found_victim = true;
        }

        if (found_candidate && found_victim) {
            break;
        }
    }

    // check if we need to preempt any one to fit one more cta of the candidate
    // kernel
    if (found_victim && found_candidate) {
        // this funky more_cta_including_pending checks whether we have in fact
        // have enough cta preemption in progress to accommodate all the ctas
        // candidate has
        if (candidate->more_cta_including_pending() &&
            !can_issue_1block(*candidate)) {
            // candidate kernel is running under its quota and not enough
            // resources are available to launch more cta
            candidate->inc_pending();
        }
    }

    // we should always preempt the victim if one exists
    return found_victim;
}

// attempts to preempt enough ctas of the victim kernel in return for one cta of
// the candidate kernel
bool shader_core_ctx::preempt_ctas(kernel_info_t *victim,
                                   kernel_info_t *candidate) {
    assert(victim != NULL);
    assert(candidate != NULL);

    if (is_preemption_wip()) {
        if ((gpu_sim_cycle + gpu_tot_sim_cycle) % 1000 == 0)
            printf(">>>>>>>>>>>>>>>>>>>>> WIP: victim %s for candidate %s on "
                   "shader %d\n",
                   victim->name().c_str(), candidate->name().c_str(),
                   this->m_sid);

        return false;
    }

    if (m_kernel2ctas.count(victim->get_uid()) == 0)
        return false;

    // first, calculate how many ctas of victim needs to be swapped out to
    // allocate one block of candidate

    unsigned int vic_padded_cta_size = victim->threads_per_cta();
    unsigned int can_padded_cta_size = candidate->threads_per_cta();

    const unsigned int warp_size = m_config->warp_size;

    if (vic_padded_cta_size % warp_size)
        vic_padded_cta_size =
            ((vic_padded_cta_size / warp_size) + 1) * (warp_size);

    if (can_padded_cta_size % warp_size)
        can_padded_cta_size =
            ((can_padded_cta_size / warp_size) + 1) * (warp_size);

    const struct gpgpu_ptx_sim_info *vic_info =
        ptx_sim_kernel_info(victim->entry());
    const struct gpgpu_ptx_sim_info *can_info =
        ptx_sim_kernel_info(candidate->entry());

    const unsigned vic_smem = vic_info->smem;
    const unsigned can_smem = can_info->smem;
    const unsigned vic_reg = vic_padded_cta_size * ((vic_info->regs + 3) & ~3);
    const unsigned can_reg = can_padded_cta_size * ((can_info->regs + 3) & ~3);

    unsigned num_ctas = 1;

    // iteratively determine how many ctas of victim are needed
    while (num_ctas <= m_kernel2ctas[victim->get_uid()].size()) {

        // check shared memory usage
        if (m_occupied_shmem - num_ctas * vic_smem + can_smem <
            m_config->gpgpu_shmem_size)
            // check register usage
            if (m_occupied_regs - num_ctas * vic_reg + can_reg <
                m_config->gpgpu_shader_registers)
                // check hw thread slots
                if (m_occupied_n_threads - num_ctas * vic_padded_cta_size +
                        can_padded_cta_size <
                    m_config->n_thread_per_shader)
                    break;

        ++num_ctas;
    }

    // give up now if we didn't find the number of victim ctas enough for one
    // candidate cta
    if (num_ctas > m_kernel2ctas[victim->get_uid()].size())
        return false;

    // mark preempted ctas
    unsigned start_tid, end_tid;
    std::vector<unsigned>::iterator start_cta_it, end_cta_it;

    // sort victim ctas list (ascending order)
    std::sort(m_kernel2ctas[victim->get_uid()].begin(),
              m_kernel2ctas[victim->get_uid()].end());

    if (victim->allocate_from_top()) {
        start_cta_it = m_kernel2ctas[victim->get_uid()].end() - num_ctas;
        start_tid = m_occupied_cta_to_hwtid[*start_cta_it];

        end_cta_it = m_kernel2ctas[victim->get_uid()].end() - 1;
        end_tid = m_occupied_cta_to_hwtid[*end_cta_it] + vic_padded_cta_size;

    } else {
        start_cta_it = m_kernel2ctas[victim->get_uid()].begin();
        start_tid = m_occupied_cta_to_hwtid[*start_cta_it];

        end_cta_it = m_kernel2ctas[victim->get_uid()].begin() + num_ctas - 1;
        end_tid = m_occupied_cta_to_hwtid[*end_cta_it] + vic_padded_cta_size;
    }

    // FIXME: can increase the number of preempted cta to find contiguous tids
    // check if the range of tids belongs to victim kernel or simply idle
    for (unsigned tid = start_tid; tid < end_tid; ++tid) {
        assert(tid >= 0 && tid < m_warp_count * m_warp_size);
        if (m_occupied_hwtid.test(tid)) {
            // if the bit is set, the thread slot might be
            // 1. null if the slot was taken due to padded cta size
            // 2. not null in which case we need check whether the kernel taking
            // it is the victim kernel if not, we can't preempt this slot
            if (m_thread[tid] &&
                m_thread[tid]->get_kernel().get_uid() != victim->get_uid())
                return false;
        }
    }

    m_preempted_ctas = std::vector<unsigned>(start_cta_it, end_cta_it + 1);

    return true;
}

bool shader_core_ctx::preempt_ctas(kernel_info_t *victim) {
    assert(victim != NULL);

    if (is_preemption_wip()) {
        if ((gpu_sim_cycle + gpu_tot_sim_cycle) % 1000 == 0)
            printf(">>>>>>>>>>>>>>>>>>>>> WIP: victim %s on shader %d\n",
                   victim->name().c_str(), this->m_sid);

        return false;
    }

    if (m_kernel2ctas.count(victim->get_uid()) == 0)
        return false;

    // Preempt all the CTAs that are over the quota limit of victim
    unsigned num_ctas =
        m_kernel2ctas[victim->get_uid()].size() - victim->get_cta_quota();
    assert(num_ctas < m_kernel2ctas[victim->get_uid()].size());

    if (num_ctas == 0) {
        return false;
    }

    // mark preempted ctas
    unsigned start_tid, end_tid;
    std::vector<unsigned>::iterator start_cta_it, end_cta_it;

    // sort victim ctas list (ascending order)
    std::sort(m_kernel2ctas[victim->get_uid()].begin(),
              m_kernel2ctas[victim->get_uid()].end());

    unsigned int vic_padded_cta_size = victim->threads_per_cta();
    const unsigned int warp_size = m_config->warp_size;
    if (vic_padded_cta_size % warp_size)
        vic_padded_cta_size =
            ((vic_padded_cta_size / warp_size) + 1) * (warp_size);

    if (victim->allocate_from_top()) {
        start_cta_it = m_kernel2ctas[victim->get_uid()].end() - num_ctas;
        start_tid = m_occupied_cta_to_hwtid[*start_cta_it];

        end_cta_it = m_kernel2ctas[victim->get_uid()].end() - 1;
        end_tid = m_occupied_cta_to_hwtid[*end_cta_it] + vic_padded_cta_size;

    } else {
        start_cta_it = m_kernel2ctas[victim->get_uid()].begin();
        start_tid = m_occupied_cta_to_hwtid[*start_cta_it];

        end_cta_it = m_kernel2ctas[victim->get_uid()].begin() + num_ctas - 1;
        end_tid = m_occupied_cta_to_hwtid[*end_cta_it] + vic_padded_cta_size;
    }

    // FIXME: can increase the number of preempted cta to find contiguous tids
    // check if the range of tids belongs to victim kernel or simply idle
    for (unsigned tid = start_tid; tid < end_tid; ++tid) {
        assert(tid >= 0 && tid < m_warp_count * m_warp_size);
        if (m_occupied_hwtid.test(tid)) {
            // if the bit is set, the thread slot might be
            // 1. null if the slot was taken due to padded cta size
            // 2. not null in which case we need check whether the kernel taking
            // it is the victim kernel if not, we can't preempt this slot
            if (m_thread[tid] &&
                m_thread[tid]->get_kernel().get_uid() != victim->get_uid())
                return false;
        }
    }

    m_preempted_ctas = std::vector<unsigned>(start_cta_it, end_cta_it + 1);

    return true;
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count,
                                   unsigned stream_id) {
    if (m_memory_config->m_perf_sim_memcpy) {
        //       assert (dst_start_addr % 32 == 0);
        if (dst_start_addr % 32 == 0) {
            // Only update performance model if the destination address is
            // aligned
            for (unsigned counter = 0; counter < count; counter += 32) {
                const unsigned wr_addr = dst_start_addr + counter;
                addrdec_t raw_addr;
                mem_access_sector_mask_t mask;
                mask.set(wr_addr % 128 / 32);
                m_memory_config->m_address_mapping.addrdec_tlx(wr_addr,
                                                               &raw_addr);
                const unsigned partition_id =
                    raw_addr.sub_partition /
                    m_memory_config->m_n_sub_partition_per_memory_channel;
                m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
                    wr_addr, raw_addr.sub_partition, mask, stream_id);
            }
        }
    }
}

void gpgpu_sim::invalidate_cache(unsigned stream_id) {
    // Invalidate L2
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
        m_memory_sub_partition[i]->invalidateL2(stream_id);
    }

    // Invalidate L1D, L1I, L0I
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        m_cluster[i]->cache_invalidate(stream_id);
    }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
    /*
       You may want to use this function while running GPGPU-Sim in gdb.
       One way to do that is add the following to your .gdbinit file:

          define dp
             call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
          end

       Then, typing "dp 3" will show the contents of the pipeline for shader
       core 3.
    */

    printf("Dumping pipeline state...\n");
    if (!mask)
        mask = 0xFFFFFFFF;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (s != -1) {
            i = s;
        }
        if (mask & 1)
            m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
                i, stdout, 1, mask & 0x2E);
        if (s != -1) {
            break;
        }
    }
    if (mask & 0x10000) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            if (m != -1) {
                i = m;
            }
            printf("DRAM / memory controller %u:\n", i);
            if (mask & 0x100000)
                m_memory_partition_unit[i]->print_stat(stdout);
            if (mask & 0x1000000)
                m_memory_partition_unit[i]->visualize();
            if (mask & 0x10000000)
                m_memory_partition_unit[i]->print(stdout);
            if (m != -1) {
                break;
            }
        }
    }
    fflush(stdout);
}

const struct shader_core_config *gpgpu_sim::getShaderCoreConfig() {
    return m_shader_config;
}

const struct memory_config *gpgpu_sim::getMemoryConfig() {
    return m_memory_config;
}

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }
