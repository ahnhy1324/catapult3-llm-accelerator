package require ::quartus::project
package require ::quartus::flow

if {$argc != 4} {
    error "usage: synth_microbench.tcl <top> <lanes> <clock_mhz> <build_dir>"
}
set top [lindex $argv 0]
set lanes [lindex $argv 1]
set clock_mhz [lindex $argv 2]
set build_dir [file normalize [lindex $argv 3]]
set source_dir [file normalize [file dirname [info script]]]
set project "${top}_${lanes}_${clock_mhz}"
file mkdir $build_dir
cd $build_dir

project_new $project -overwrite
set_global_assignment -name FAMILY "Arria 10"
# Catapult3 Rev E non-standard marking. Quartus 25.1 resolves it to
# device 10AX115_JZ, GX family, speed grade 2.
set_global_assignment -name DEVICE 10AXF40GAE
set_global_assignment -name TOP_LEVEL_ENTITY $top
set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL
set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
set_global_assignment -name PHYSICAL_SYNTHESIS_REGISTER_RETIMING ON
set_global_assignment -name PHYSICAL_SYNTHESIS_COMBO_LOGIC ON
set_global_assignment -name AUTO_RAM_RECOGNITION ON
if {$top eq "bonsai_binary_g128_dot"} {
    set_global_assignment -name SYSTEMVERILOG_FILE [file join $source_dir bonsai_binary_g128_dot.sv]
} elseif {$top eq "bitnet_direct_ternary"} {
    set_global_assignment -name SYSTEMVERILOG_FILE [file join $source_dir bitnet_direct_ternary.sv]
} elseif {$top eq "bitnet_tl5"} {
    set_global_assignment -name SYSTEMVERILOG_FILE [file join $source_dir bitnet_tl5.sv]
} else {
    error "unsupported top: $top"
}
set_parameter -name LANES $lanes
set_parameter -name PIPE_DEPTH 2
if {$top eq "bonsai_binary_g128_dot"} {
    set_parameter -name GROUP_SIZE 128
    set_parameter -name ACT_W 8
    set_parameter -name ACC_W 24
    set_parameter -name SCALE_W 16
    set_parameter -name SCALE_FRAC 8
    set_parameter -name OUT_W 40
} else {
    set_parameter -name ACT_W 8
    set_parameter -name OUT_W 32
}
set_instance_assignment -name VIRTUAL_PIN ON -to *
set period_ns [expr {1000.0 / double($clock_mhz)}]
set sdc_path [file join $build_dir microbench.sdc]
set sdc [open $sdc_path w]
puts $sdc "create_clock -name clk -period $period_ns \[get_ports \{clk\}\]"
puts $sdc "derive_clock_uncertainty"
close $sdc
set_global_assignment -name SDC_FILE $sdc_path
project_close

project_open $project
execute_flow -compile
project_close
