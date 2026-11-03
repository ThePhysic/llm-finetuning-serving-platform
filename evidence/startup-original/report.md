# Measured paired backend startup

3 fresh Python children each loaded the same immutable base package twice. The second load followed real generation and unload in the same child.

First-load median: **20.4554 seconds**. Same-process warm-reload median: **14.7571 seconds**.

First load includes cold heavy Python imports and CUDA context initialization. Both conditions use locally cached weights with uncontrolled OS cache. These observations do not establish disk-cold startup or isolate weight I/O. The small paired sample supports descriptive reporting, not stable tail estimates. Raw load/unload/generation timings, CUDA memory, policy identity, versions and source hashes are in the repetition files and summary.json.
