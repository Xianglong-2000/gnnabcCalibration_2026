# CellModeller-output-processing
This script converts CellModeller pickle files (output) into CSV files, similar to CellProfiler’s CSV output. </br>
More importantly, it calculates neighbor information based on the definition that two objects are neighbors if their expanded pixel boundaries touch. This aligns perfectly with CellProfiler's "MeasureObjectsNeighbors" module. 
</br>
<b>Note:</b> While CellModeller's "neighbors" module provides neighbor information, it differs from this definition, which is why this script has been implemented.
