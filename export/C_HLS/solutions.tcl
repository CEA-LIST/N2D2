################################################################################
# Author: Olivier BICHLER (olivier.bichler@cea.fr)
# (C) Copyright 2015 CEA LIST
################################################################################

# "solutions.tcl" is included into N2D2 auto-generated "run_hls.tcl"
# Command line example: vivado_hls -f run_hls.tcl

proc create_solution {sol} {
    set tm [clock format [clock second] -format %y.%m.%d-%Hh%M]
    open_solution -reset "${tm}_$sol"
    # Kintex UltraScale KU040 (on KCU105 evaluation board)
    set_part  {xcku060-ffva1156-2-e}
    create_clock -period 10

    file mkdir "hls/${tm}_$sol/src/"
    foreach f [concat Makefile [glob include/*.h] [glob src/*.c] [glob *.c] [glob *.tcl]] {
        file copy -force $f "hls/${tm}_$sol/src/"
    }

    #config_schedule -effort low
    #config_bind -effort low
}

proc set_base_directives {} {
    global cells

    dict for {name info} $cells {
        dict with info {
            if {$type == "Conv"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                set_directive_array_partition -dim 2 -type complete "$name" inputs
                set_directive_array_partition -dim 3 -type complete "$name" inputs

                # const WDATA_T weights_full[M_nbOutputs][M_nbChannels][M_kernelHeight][M_kernelWidth]
                set_directive_array_partition -dim 3 -type complete "$name" weights_full
                set_directive_array_partition -dim 4 -type complete "$name" weights_full

                # const int weights_map[M_nbOutputs][M_nbChannels]
                set_directive_array_partition -dim 2 -type complete "$name" weights_map

                # const WDATA_T weights_compact[M_nbKernels][M_kernelHeight][M_kernelWidth]
                set_directive_array_partition -dim 2 -type complete "$name" weights_compact
                set_directive_array_partition -dim 3 -type complete "$name" weights_compact
            } elseif {$type == "Pool" || $type == "Pool_UnitMap"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                set_directive_array_partition -dim 2 -type complete "$name" inputs
                set_directive_array_partition -dim 3 -type complete "$name" inputs

                if {$type == "Pool"} {
                    # const char mapping[M_nbOutputs][M_nbChannels]
                    set_directive_array_partition -dim 2 -type complete "$name" mapping
                }
            } elseif {$type == "Fc_2D"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                set_directive_array_partition -dim 2 -type complete "$name" inputs
                set_directive_array_partition -dim 3 -type complete "$name" inputs

                # const WDATA_T weights[M_nbOutputs][M_nbChannels_]
                set_directive_array_partition -dim 2 -type complete "$name" weights
            } elseif {$type == "Fc"} {
                # DATA_T inputs[M_nbChannels]
                set_directive_array_partition -dim 1 -type complete "$name" inputs

                # const WDATA_T weights[M_nbOutputs][M_nbChannels]
                set_directive_array_partition -dim 2 -type complete "$name" weights
            }
        }
    }

    set_directive_dataflow "network"
    #set_directive_interface -mode ap_fifo -depth 2 "network" env_data_in
}

proc set_base_pipelining {level} {
    global cells

    dict for {name info} $cells {
        dict with info {
            if {$type == "Conv"} {
                if {$level == 0} {
                    set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_O"
                } elseif {$level == 1} {
                    set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_OUTPUT"
                } elseif {$level == 2} {
                    set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_CHANNEL"
                }
            } elseif {$type == "Pool" || $type == "Pool_UnitMap"} {
                if {$level == 0} {
                    set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_O"
                } elseif {$level == 1} {
                    set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_OUTPUT"
                } elseif {$level == 2} {
                    if {$type == "Pool"} {
                        set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_CHANNEL"
                    } else {
                        set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_S"
                    }
                }
            } elseif {$type == "Fc_2D" || $type == "Fc"} {
                if {$level == 0} {
                    set_directive_pipeline -II 1 "$name/LOOP_FCCELL_OUTPUT"
                } elseif {$level == 1 || $level == 2} {
                    set_directive_pipeline -II 1 "$name/LOOP_FCCELL_CHANNEL"
                }
            }
        }
    }

    if {$level == 0} {
        set_directive_pipeline -II 1 "fccell_output_max/LOOP_FCCELL_OUTPUT_MAX"
    }
}

proc set_directives {max_latency auto_partition} {
    global cells

    # Options
    set partial_unrolling true
    set top_loop_unrolling true

    # Auto check if ENABLE_X_Y_LOOPS is enabled
    set n2d2_hls_filename "include/n2d2_hls.h"
    set enable_x_y_loops_pattern {^\s*#define\s+ENABLE_X_Y_LOOPS\s*$}
    set n2d2_hls_file [open $n2d2_hls_filename r]
    set enable_x_y_loops [regexp -lineanchor -all -- $enable_x_y_loops_pattern [read $n2d2_hls_file [file size $n2d2_hls_filename]]]

    if {$auto_partition} {
        config_array_partition -throughput_driven -include_ports
    }

    dict for {name info} $cells {
        dict with info {
            # Channel info
            if {$type == "Pool_UnitMap"} {
                set nb_channels 1
            } else {
                set nb_channels [lindex $channel_dims 0]
            }

            set channels_width [lindex $channel_dims 1]
            set channels_height [lindex $channel_dims 2]
            set channels_size [expr $channels_width*$channels_height]

            # Output info
            set nb_outputs [lindex $output_dims 0]
            set outputs_width [lindex $output_dims 1]
            set outputs_height [lindex $output_dims 2]
            set outputs_size [expr $outputs_width*$outputs_height]

            # Kernel info
            set kernel_width [lindex $kernel_dims 0]
            set kernel_height [lindex $kernel_dims 1]
            set kernel_size [expr $kernel_width*$kernel_height]

            # Number of operations
            set max_latency_output_map $max_latency
            set max_latency_output_map_oy $max_latency
            set max_latency_output_map_ox [expr int(floor(1.0*$max_latency_output_map_oy/$outputs_height))]
            set max_latency_output [expr max(1, int(floor(1.0*$max_latency_output_map/$outputs_size)))]
            set max_latency_channel [expr int(floor(1.0*$max_latency_output/$nb_outputs))]
            set max_latency_kernel [expr int(floor(1.0*$max_latency_channel/$nb_channels))]
            set max_latency_kernel_sy [expr int(floor(1.0*$max_latency_channel/$nb_channels))]
            set max_latency_kernel_sx [expr int(floor(1.0*$max_latency_kernel_sy/$kernel_height))]

            set target_ii_output_map [expr int(floor(1.0*$max_latency/$outputs_size))]
            set target_ii_output_map_oy [expr int(floor(1.0*$max_latency/$outputs_height))]
            set target_ii_output_map_ox [expr int(floor(1.0*$target_ii_output_map_oy/$outputs_width))]
            set target_ii_output [expr int(floor(1.0*$target_ii_output_map/$nb_outputs))]
            set target_ii_channel [expr int(floor(1.0*$target_ii_output/$nb_channels))]
            set target_ii_kernel [expr int(floor(1.0*$target_ii_channel/$channels_size))]
            set target_ii_kernel_sy [expr int(floor(1.0*$target_ii_channel/$kernel_height))]
            set target_ii_kernel_sx [expr int(floor(1.0*$target_ii_kernel_sy/$kernel_width))]

            set trip_count_output_map $outputs_size
            set trip_count_output_map_oy $outputs_height
            set trip_count_output_map_ox $outputs_width
            set trip_count_output $nb_outputs
            set trip_count_channel $nb_channels
            set trip_count_kernel $kernel_size
            set trip_count_kernel_sy $kernel_height
            set trip_count_kernel_sx $kernel_width

            set unroll_factor_output_map [expr int(ceil(1.0*$trip_count_output_map/$max_latency))]
            set unroll_factor_output_map_oy [expr int(ceil(1.0*$trip_count_output_map_oy/$max_latency))]
            set unroll_factor_output_map_ox [expr int(ceil(1.0*$trip_count_output_map_ox/$max_latency_output_map_ox))]
            set unroll_factor_output [expr int(ceil(1.0*$trip_count_output/$max_latency_output))]

            puts "$name:"
            if {$enable_x_y_loops} {
                puts "    max_latency_oy = $max_latency_output_map_oy; trip cnt = $trip_count_output_map_oy;\
                    target II = $target_ii_output_map_oy; unroll = $unroll_factor_output_map_oy"
                puts "    max_latency_ox = $max_latency_output_map_ox; trip cnt = $trip_count_output_map_ox;\
                    target II = $target_ii_output_map_ox; unroll = $unroll_factor_output_map_ox"
            } else {
                puts "    max_latency_o_map = $max_latency_output_map; trip cnt = $trip_count_output_map;\
                    target II = $target_ii_output_map; unroll = $unroll_factor_output_map"
            }
            puts "    max_latency_output = $max_latency_output; trip cnt = $trip_count_output; target II = $target_ii_output;\
                unroll = $unroll_factor_output"
            puts "    max_latency_channel = $max_latency_channel; trip cnt = $trip_count_channel; target II = $target_ii_channel"
            if {$enable_x_y_loops} {
                puts "    max_latency_sy = $max_latency_kernel_sy; trip cnt = $trip_count_kernel_sy; \
                    target II = $target_ii_kernel_sy"
                puts "    max_latency_sx = $max_latency_kernel_sx; trip cnt = $trip_count_kernel_sx; \
                    target II = $target_ii_kernel_sx"
            } else {
                puts "    max_latency_kernel = $max_latency_kernel; trip cnt = $trip_count_kernel; target II = $target_ii_kernel"
            }

            if {$type == "Conv"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                # const WDATA_T weights_full[M_nbOutputs][M_nbChannels][M_kernelHeight][M_kernelWidth]
                # const WDATA_T weights_compact[M_nbKernels][M_kernelHeight][M_kernelWidth]
                # const int weights_map[M_nbOutputs][M_nbChannels]

                if {$enable_x_y_loops && $target_ii_kernel_sx > 0} {
                    set_directive_pipeline -II "$target_ii_kernel_sx" "$name/LOOP_CONVCELL_SX"
                } elseif {$target_ii_kernel_sy > 0 || $target_ii_kernel > 0} {
                    if {$enable_x_y_loops} {
                        set_directive_pipeline -II "$target_ii_kernel_sy" "$name/LOOP_CONVCELL_SY"
                    } else {
                        set_directive_pipeline -II "$target_ii_kernel" "$name/LOOP_CONVCELL_S"
                    }
                } elseif {$target_ii_channel > 0} {
                    set_directive_pipeline -II "$target_ii_channel" "$name/LOOP_CONVCELL_CHANNEL"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$kernel_size/$target_ii_channel/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $kernel_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" weights_compact
                            set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_height" "$name" weights_full
                        }

                        if {$min_nb_reads_width > 1} {
                            set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" weights_compact
                            set_directive_array_partition -dim 4 -type cyclic -factor "$min_nb_reads_width" "$name" weights_full
                        }
                    }
                } elseif {$target_ii_output > 0} {
                    set_directive_pipeline -II "$target_ii_output" "$name/LOOP_CONVCELL_OUTPUT"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$nb_channels*$kernel_size/$target_ii_output/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $kernel_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        set split_kernel_width [expr min($min_nb_reads_width, $kernel_width)]
                        set split_kernel_height [expr min($min_nb_reads_height, $kernel_height)]
                        set split_kernel_channel [expr int(ceil(1.0*$min_nb_reads/($split_kernel_width*$split_kernel_height)))]

                        for {set i 2} {$min_nb_reads_height > $channels_height} {incr i} {
                            set min_nb_reads_width [expr min($min_nb_reads, $i*$kernel_width)]
                            set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]
                        }

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width (split_kernel_width = $split_kernel_width)"
                        puts "        min_nb_reads_height = $min_nb_reads_height (split_kernel_height = $split_kernel_height)"

                        if {$min_nb_reads_height > 1} {
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            set_directive_array_partition -dim 2 -type cyclic -factor "$split_kernel_height" "$name" weights_compact
                            set_directive_array_partition -dim 3 -type cyclic -factor "$split_kernel_height" "$name" weights_full
                        }

                        if {$min_nb_reads_width > 1} {
                            if {$min_nb_reads_width > [expr $channels_width/2]} {
                                set_directive_array_partition -dim 3 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            }

                            set_directive_array_partition -dim 3 -type cyclic -factor "$split_kernel_width" "$name" weights_compact
                            set_directive_array_partition -dim 4 -type cyclic -factor "$split_kernel_width" "$name" weights_full
                        }

                        if {$split_kernel_channel > 1} {
                            set_directive_array_partition -dim 1 -type cyclic -factor "$split_kernel_channel" "$name" \
                                weights_compact
                            set_directive_array_partition -dim 2 -type cyclic -factor "$split_kernel_channel" "$name" \
                                weights_full
                        }

                        set_directive_array_partition -dim 2 -type complete "$name" weights_map
                    }
                } elseif {$enable_x_y_loops && $target_ii_output_map_ox > 0} {
                    if {$partial_unrolling && $unroll_factor_output < $trip_count_output} {
                        set_directive_unroll -factor "$unroll_factor_output" "$name/LOOP_CONVCELL_OUTPUT"
                        set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_OUTPUT"
                    } else {
                        set_directive_pipeline -II "$target_ii_output_map_ox" "$name/LOOP_CONVCELL_OX"
                    }
                } elseif {$target_ii_output_map_oy > 0 || $target_ii_output_map > 0} {
                    if {$enable_x_y_loops} {
                        if {$partial_unrolling && $unroll_factor_output_map_ox < $trip_count_output_map_ox} {
                            set_directive_unroll -factor "$unroll_factor_output_map_ox" "$name/LOOP_CONVCELL_OX"
                            set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_OX"
                        } else {
                            set_directive_pipeline -II "$target_ii_output_map" "$name/LOOP_CONVCELL_OY"
                        }
                    } else {
                        if {$partial_unrolling && $unroll_factor_output < $trip_count_output} {
                            set_directive_unroll -factor "$unroll_factor_output" "$name/LOOP_CONVCELL_OUTPUT"
                            set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_OUTPUT"
                        } else {
                            set_directive_pipeline -II "$target_ii_output_map" "$name/LOOP_CONVCELL_O"
                        }
                    }

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$nb_outputs*$nb_channels*$kernel_size/$target_ii_output_map/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $channels_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            if {$min_nb_reads_height > [expr $channels_height/2]} {
                                set_directive_array_partition -dim 2 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            }
                        }

                        if {$min_nb_reads_width > 1} {
                            if {$min_nb_reads_width > [expr $channels_width/2]} {
                                set_directive_array_partition -dim 3 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            }
                        }

                        set_directive_array_partition -dim 0 -type complete "$name" weights_map
                        set_directive_array_partition -dim 2 -type complete "$name" weights_compact
                        set_directive_array_partition -dim 3 -type complete "$name" weights_compact
                        set_directive_array_partition -dim 3 -type complete "$name" weights_full
                        set_directive_array_partition -dim 4 -type complete "$name" weights_full
                    }
                } else {
                    if {$top_loop_unrolling} {
                        if {$unroll_factor_output_map < $trip_count_output_map} {
                            set_directive_unroll -factor "$unroll_factor_output_map" "$name/LOOP_CONVCELL_O"
                            set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_O"
                        } else {
                            set_directive_unroll "$name/LOOP_CONVCELL_O"
                        }
                    } else {
                        set_directive_pipeline -II 1 "$name/LOOP_CONVCELL_O"
                    }

                    if {!$auto_partition} {
                        set_directive_array_partition -dim 0 -type complete "$name" inputs
                        set_directive_array_partition -dim 0 -type complete "$name" weights_map
                        set_directive_array_partition -dim 0 -type complete "$name" weights_compact
                        set_directive_array_partition -dim 0 -type complete "$name" weights_full
                    }
                }
            } elseif {$type == "Pool" || $type == "Pool_UnitMap"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                # const char mapping[M_nbOutputs][M_nbChannels]

                if {$enable_x_y_loops && $target_ii_kernel_sx > 0} {
                    set_directive_pipeline -II "$target_ii_kernel_sx" "$name/LOOP_POOLCELL_SX"
                } elseif {$target_ii_kernel_sy > 0 || $target_ii_kernel > 0} {
                    if {$enable_x_y_loops} {
                        set_directive_pipeline -II "$target_ii_kernel_sy" "$name/LOOP_POOLCELL_SY"
                    } else {
                        set_directive_pipeline -II "$target_ii_kernel" "$name/LOOP_POOLCELL_S"
                    }
                } elseif {$target_ii_channel > 0 && $type == "Pool"} {
                    set_directive_pipeline -II "$target_ii_channel" "$name/LOOP_POOLCELL_CHANNEL"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$kernel_size/$target_ii_channel/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $kernel_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        for {set i 2} {$min_nb_reads_height > $channels_height} {incr i} {
                            set min_nb_reads_width [expr min($min_nb_reads, $i*$kernel_width)]
                            set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]
                        }

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                        }

                        if {$min_nb_reads_width > 1} {
                            set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                        }
                    }
                } elseif {$target_ii_output > 0} {
                    set_directive_pipeline -II "$target_ii_output" "$name/LOOP_POOLCELL_OUTPUT"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$nb_channels*$kernel_size/$target_ii_output/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $channels_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            if {$min_nb_reads_height > [expr $channels_height/2]} {
                                set_directive_array_partition -dim 2 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            }
                        }

                        if {$min_nb_reads_width > 1} {
                            if {$min_nb_reads_width > [expr $channels_width/2]} {
                                set_directive_array_partition -dim 3 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            }
                        }

                        if {$type == "Pool"} {
                            set_directive_array_partition -dim 2 -type complete "$name" mapping
                        }
                    }
                } elseif {$enable_x_y_loops && $target_ii_output_map_ox > 0} {
                    if {$partial_unrolling && $unroll_factor_output < $trip_count_output} {
                        set_directive_unroll -factor "$unroll_factor_output" "$name/LOOP_POOLCELL_OUTPUT"
                        set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_OUTPUT"
                    } else {
                        set_directive_pipeline -II "$target_ii_output_map_ox" "$name/LOOP_POOLCELL_OX"
                    }
                } elseif {$target_ii_output_map_oy > 0 || $target_ii_output_map > 0} {
                    if {$enable_x_y_loops} {
                        if {$partial_unrolling && $unroll_factor_output_map_ox < $trip_count_output_map_ox} {
                            set_directive_unroll -factor "$unroll_factor_output_map_ox" "$name/LOOP_POOLCELL_OX"
                            set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_OX"
                        } else {
                            set_directive_pipeline -II "$target_ii_output_map" "$name/LOOP_POOLCELL_OY"
                        }
                    } else {
                        if {$partial_unrolling && $unroll_factor_output < $trip_count_output} {
                            set_directive_unroll -factor "$unroll_factor_output" "$name/LOOP_POOLCELL_OUTPUT"
                            set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_OUTPUT"
                        } else {
                            set_directive_pipeline -II "$target_ii_output_map" "$name/LOOP_POOLCELL_O"
                        }
                    }

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$nb_outputs*$nb_channels*$kernel_size/$target_ii_output_map/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $channels_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            if {$min_nb_reads_height > [expr $channels_height/2]} {
                                set_directive_array_partition -dim 2 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            }
                        }

                        if {$min_nb_reads_width > 1} {
                            if {$min_nb_reads_width > [expr $channels_width/2]} {
                                set_directive_array_partition -dim 3 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            }
                        }

                        if {$type == "Pool"} {
                            set_directive_array_partition -dim 0 -type complete "$name" mapping
                        }
                    }
                } else {
                    if {$top_loop_unrolling} {
                        if {$unroll_factor_output_map < $trip_count_output_map} {
                            set_directive_unroll -factor "$unroll_factor_output_map" "$name/LOOP_POOLCELL_O"
                            set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_O"
                        } else {
                            set_directive_unroll "$name/LOOP_POOLCELL_O"
                        }
                    } else {
                        set_directive_pipeline -II 1 "$name/LOOP_POOLCELL_O"
                    }

                    if {!$auto_partition} {
                        set_directive_array_partition -dim 0 -type complete "$name" inputs

                        if {$type == "Pool"} {
                            set_directive_array_partition -dim 0 -type complete "$name" mapping
                        }
                    }
                }
            } elseif {$type == "Fc_2D"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                # const WDATA_T weights[M_nbOutputs][M_nbChannels_]

                if {$target_ii_kernel > 0} {
                    set_directive_pipeline -II "$target_ii_kernel" "$name/LOOP_FCCELL_I"
                } elseif {$target_ii_channel > 0} {
                    set_directive_pipeline -II "$target_ii_channel" "$name/LOOP_FCCELL_CHANNEL"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$channels_size/$target_ii_channel/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $channels_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                        }

                        if {$min_nb_reads_width > 1} {
                            set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                        }

                        set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads" "$name" weights
                    }
                } elseif {$target_ii_output > 0} {
                    set_directive_pipeline -II "$target_ii_output" "$name/LOOP_FCCELL_OUTPUT"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$nb_channels*$channels_size/$target_ii_output/2.0))]
                        set min_nb_reads_width [expr min($min_nb_reads, $channels_width)]
                        set min_nb_reads_height [expr int(ceil(1.0*$min_nb_reads/$min_nb_reads_width))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"
                        puts "        min_nb_reads_width = $min_nb_reads_width"
                        puts "        min_nb_reads_height = $min_nb_reads_height"

                        if {$min_nb_reads_height > 1} {
                            if {$min_nb_reads_height > [expr $channels_height/2]} {
                                set_directive_array_partition -dim 2 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads_height" "$name" inputs
                            }
                        }

                        if {$min_nb_reads_width > 1} {
                            if {$min_nb_reads_width > [expr $channels_width/2]} {
                                set_directive_array_partition -dim 3 -type complete "$name" inputs
                            } else {
                                set_directive_array_partition -dim 3 -type cyclic -factor "$min_nb_reads_width" "$name" inputs
                            }
                        }

                        set_directive_array_partition -dim 2 -type complete "$name" weights
                    }
                } else {
                    if {$top_loop_unrolling} {
                        if {$unroll_factor_output_map < $trip_count_output_map} {
                            set_directive_unroll -factor "$unroll_factor_output_map" "$name/LOOP_FCCELL_OUTPUT"
                            set_directive_pipeline -II 1 "$name/LOOP_FCCELL_OUTPUT"
                        } else {
                            set_directive_unroll "$name/LOOP_FCCELL_OUTPUT"
                        }
                    } else {
                        set_directive_pipeline -II 1 "$name/LOOP_FCCELL_OUTPUT"
                    }

                    if {!$auto_partition} {
                        set_directive_array_partition -dim 0 -type complete "$name" inputs
                        set_directive_array_partition -dim 0 -type complete "$name" weights
                    }
                }
            } elseif {$type == "Fc"} {
                # DATA_T inputs[M_nbChannels]
                # const WDATA_T weights[M_nbOutputs][M_nbChannels]

                if {$target_ii_channel > 0} {
                    set_directive_pipeline -II "$target_ii_channel" "$name/LOOP_FCCELL_CHANNEL"

                    if {!$auto_partition} {
                        set min_nb_reads [expr int(ceil(1.0*$channels_size/$target_ii_channel/2.0))]

                        puts "    memory split:"
                        puts "        min_nb_reads = $min_nb_reads"

                        if {$min_nb_reads > 1} {
                            set_directive_array_partition -dim 1 -type cyclic -factor "$min_nb_reads" "$name" inputs
                            set_directive_array_partition -dim 2 -type cyclic -factor "$min_nb_reads" "$name" weights
                        }
                    }
                } elseif {$target_ii_output > 0} {
                    set_directive_pipeline -II "$target_ii_output" "$name/LOOP_FCCELL_OUTPUT"

                    if {!$auto_partition} {
                        set_directive_array_partition -dim 0 -type complete "$name" inputs
                        set_directive_array_partition -dim 2 -type complete "$name" weights
                    }
                } else {
                    if {$top_loop_unrolling} {
                        if {$unroll_factor_output_map < $trip_count_output_map} {
                            set_directive_unroll -factor "$unroll_factor_output_map" "$name/LOOP_FCCELL_OUTPUT"
                            set_directive_pipeline -II 1 "$name/LOOP_FCCELL_OUTPUT"
                        } else {
                            set_directive_unroll "$name/LOOP_FCCELL_OUTPUT"
                        }
                    } else {
                        set_directive_pipeline -II 1 "$name/LOOP_FCCELL_OUTPUT"
                    }

                    if {!$auto_partition} {
                        set_directive_array_partition -dim 0 -type complete "$name" inputs
                        set_directive_array_partition -dim 0 -type complete "$name" weights
                    }
                }
            }

            # No inter dependence (write order can be arbitrary)
            #set_directive_dependence -variable outputs -type inter -direction WAW -dependent false "$name"
        }

        if {!$auto_partition} {
            set_directive_inline -region -recursive "$name"
        }
    }

    if {$auto_partition} {
        # If we use the directive config_array_partition -throughput_driven, the same array cannot be read and write in different
        # functions, because this may result in two different set_directive_array_partition directives for the same array (the
        # write constrains on one layer may be different from the read constrains of the next layer), which currently terminate
        # Vivado execution.
        set_directive_inline -region -recursive "network"
    }

    #set_directive_pipeline -II 1 "fccell_output_max/LOOP_FCCELL_OUTPUT_MAX"

    set_directive_dataflow "network"
    #set_directive_interface -mode ap_fifo -depth 2 "network" env_data_in
}

proc generate_solution {} {
    #csim_design
    csynth_design
    #cosim_design -trace_level none -rtl vhdl -tool auto
    export_design -evaluate vhdl -format ipxact
}

################################################################################
# sol_most_aggressive
# Min. possible latency and interval, huge utilization
################################################################################

proc sol_most_aggressive {} {
    global cells

    create_solution "sol_most_aggressive"
    set_base_directives
    set_base_pipelining 0

    dict for {name info} $cells {
        dict with info {
            if {$name == "Conv" || $name == "Pool" || $type == "Pool_UnitMap" || $name == "Fc_2D"} {
                # DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth]
                set_directive_array_partition -dim 1 -type complete "$name" inputs
            }
        }
    }

    generate_solution
}


################################################################################
# sol_aggressive
# Good latency and interval vs utilization tradeoff
################################################################################

proc sol_aggressive {} {
    create_solution "sol_aggressive"
    set_base_directives
    set_base_pipelining 0
    generate_solution
}


################################################################################
# sol_aggressive_constrained
# Good latency and interval with constrained utilization
################################################################################

proc sol_aggressive_constrained {} {
    create_solution "sol_aggressive_constrained"
    set_base_directives
    set_base_pipelining 0
    # Example for a network with conv1, pool1, fc1 and fc2
    set_directive_allocation -limit 192 -type operation "conv1_upropagate" mul
    set_directive_allocation -limit 64 -type operation "pool1_propagate" icmp
    set_directive_allocation -limit 96 -type operation "fc1_propagate_2d" mul
    set_directive_allocation -limit 2 -type operation "fc2_propagate" mul
    generate_solution
}


################################################################################
# sol_relaxed
# Average latency and interval, with lower utilization
################################################################################

proc sol_relaxed {} {
    create_solution "sol_relaxed"
    set_base_directives
    set_base_pipelining 1
    generate_solution
}


################################################################################
# sol_relaxed_constrained
# Average latency and interval with lower constrained utilization
################################################################################

proc sol_relaxed_constrained {} {
    create_solution "sol_relaxed_constrained"
    set_base_directives
    set_base_pipelining 1
    # Example for a network with conv1, pool1, fc1 and fc2
    set_directive_allocation -limit 192 -type operation "conv1_upropagate" mul
    set_directive_allocation -limit 64 -type operation "pool1_propagate" icmp
    set_directive_allocation -limit 96 -type operation "fc1_propagate_2d" mul
    set_directive_allocation -limit 2 -type operation "fc2_propagate" mul
    generate_solution
}


################################################################################
# sol_compact
# Higher latency and interval with even lower utilization
################################################################################

proc sol_compact {} {
    create_solution "sol_compact"
    set_base_directives
    set_base_pipelining 2
    generate_solution
}


################################################################################
# sol_adaptative
################################################################################

proc sol_adaptative {} {
    create_solution "sol_adaptative"
    set_directives 10000 true
    generate_solution
}


################################################################################
# Design exploration
################################################################################

#sol_most_aggressive
#sol_aggressive
#sol_aggressive_constrained
#sol_relaxed
#sol_relaxed_constrained
#sol_compact
sol_adaptative
