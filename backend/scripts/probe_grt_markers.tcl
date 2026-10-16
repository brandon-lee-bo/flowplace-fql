read_db /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/results/bp_fe_grt_v1_row0000/orfs_work/results/nangate45/bp_fe/eval/5_1_grt.odb
set block [ord::get_db_block]
puts "FQL_BLOCK $block"
puts "FQL_MARKER_COMMANDS [lsort [info commands odb::*MarkerCategory*]]"
set cat [$block findMarkerCategory "Global route"]
puts "FQL_ZERO_CATEGORY <$cat>"
read_db /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/results/bp_fe_grt_v1_row0003/orfs_work/results/nangate45/bp_fe/eval/5_1_grt.odb
set block [ord::get_db_block]
set cat [$block findMarkerCategory "Global route"]
puts "FQL_NONZERO_CATEGORY <$cat>"
foreach method {getMarkerCount getMarkers getAllMarkers getCategories} {
 puts "FQL_METHOD $method [catch {$cat $method} res] <$res>"
}
exit
