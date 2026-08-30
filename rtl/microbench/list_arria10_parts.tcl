package require ::quartus::project
foreach part [get_part_list -family "Arria 10"] {
    puts $part
}
