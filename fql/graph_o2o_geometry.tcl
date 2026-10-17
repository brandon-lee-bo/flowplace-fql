# Read-only audit of the saved final GRT database; no optimization commands.
read_db $::env(GO2O_AUDIT_ODB)
set block [ord::get_db_block]
set core [$block getCoreArea]
puts "GO2O_CORE\t[$block getDbUnitsPerMicron]\t[$core xMin]\t[$core yMin]\t[$core xMax]\t[$core yMax]"
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  if {![$master isBlock]} { continue }
  set box [$inst getBBox]
  puts "GO2O_MACRO\t[$inst getName]\t[$master getName]\t[$inst getOrient]\t[$inst getPlacementStatus]\t[$box xMin]\t[$box yMin]\t[$box xMax]\t[$box yMax]"
}
