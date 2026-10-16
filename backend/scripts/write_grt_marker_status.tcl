# Read-only check of the final OpenDB congestion markers. OpenROAD omits its
# congestion report entirely when the "Global route" marker category is absent.
proc write_fql_grt_marker_status {report_dir evidence_source} {
  set category [[ord::get_db_block] findMarkerCategory "Global route"]
  set report_path [file join $report_dir fql_final_congestion.rpt]
  if {$category eq "NULL"} {
    set present false
    set count 0
  } else {
    set present true
    set count [$category getMarkerCount]
    $category writeTR $report_path
  }
  set status_file [open [file join $report_dir fql_grt_marker_status.json] w]
  puts $status_file [format {{"schema":1,"evidence_source":"%s","category_present":%s,"marker_count":%d}} $evidence_source $present $count]
  close $status_file
  puts "FQL_FINAL_GRT_MARKERS category_present=$present marker_count=$count source=$evidence_source"
}
