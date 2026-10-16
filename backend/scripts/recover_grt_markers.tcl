read_db $::env(FQL_REPARSE_ODB)
source [file join [file dirname [info script]] write_grt_marker_status.tcl]
write_fql_grt_marker_status $::env(FQL_REPARSE_REPORT_DIR) saved_final_global_route_odb
exit
