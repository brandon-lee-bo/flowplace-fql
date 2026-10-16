# Run the author's unmodified global-routing optimization, then collect reports
# while its routing parasitics are still live. No detailed routing is run here.
source $::env(SCRIPTS_DIR)/global_route.tcl
report_metrics 5 "fql_global_route_label"
