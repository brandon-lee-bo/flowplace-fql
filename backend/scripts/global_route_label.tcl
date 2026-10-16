# Run the author's unmodified global-routing optimization, then collect reports
# while its routing parasitics are still live. No detailed routing is run here.
source $::env(SCRIPTS_DIR)/global_route.tcl
source [file join [file dirname [info script]] write_grt_marker_status.tcl]
write_fql_grt_marker_status $::env(REPORTS_DIR) live_final_global_route_state
report_metrics 5 "fql_global_route_label"
