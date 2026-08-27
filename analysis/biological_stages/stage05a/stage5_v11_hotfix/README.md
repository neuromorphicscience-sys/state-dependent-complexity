Stage 5 v1.1 hotfix

Confirmed Allen v1 bug:
- sweep inventory contains both stimulus_description and stimulus_name
- v1 selected stimulus_description first
- Noise rows therefore became C1NSSEED_* instead of Noise 1 / Noise 2
- v1 usable_noise=0 was a metadata-selection bug, not corrupt NWB data

v1.1:
- exact stimulus_name only
- explicit Noise 1 and Noise 2 classes
- repeated presentations averaged after stimulus-consistency audit
- 24-cell smoke test must pass >=90% before full extraction
- PFC 2022/2023 representation reconciliation by archive_id + cell_id
- no old subtype statistics are allowed to run silently on 12,714 SWC files
- explicit pipeline_status.json on failure / completion
- original aggressive RTX 5090 GPU lane reused after corrected Allen extraction

Package is a PATCH and expects the original v1 harness at:
  /data/coding/NeuralScience/biological_stage5_discovery_harness_v1
