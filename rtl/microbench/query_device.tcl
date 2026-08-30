package require ::quartus::project
set part [lindex $argv 0]
foreach property {family family_variant device package pin_count speed_grade hssi_speed_grade default_voltage} {
    if {[catch {set value [get_part_info -$property $part]} error]} {
        puts "$property=UNAVAILABLE:$error"
    } else {
        puts "$property=$value"
    }
}
