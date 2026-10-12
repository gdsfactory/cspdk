# si220 cells from before MPW Run 50

GDS files that `cspdk/si220/gds/` shipped before the Run 50 update and that exist in no other reference file (or differ from it). cspdk no longer uses them; they are kept for reference.

- `CORNERSTONE MPW SOI 220nm GDSII Template.gds` — the older SOI 220 template (includes the old `Heater` and `Layer_Designations`).
- `SOI220nm_1550nm_TE_{RIB,STRIP}_MZI.gds`, `SOI220nm_{1310,1550}nm_TE_MZI_Modulator.gds`, `SOI220nm_1550nm_TE_Defect_Detector.gds`, `SOI220nm_1550nm_TE_RIB_EBL_Apodised_Grating_Coupler.gds` — cells Cornerstone dropped from the library in the 48th call.
- `SOI220nm_1550nm_TE_STRIP_Grating_Coupler.gds`, `Layer_Designations.gds`, `Cell0_Institution_Name.gds` — variants that differ from the copies in the templates.

The other pre-Run-50 cells were identical to the cells in `../CORNERSTONE-GDSII-Template_All-Platforms_Jan_2024-1.gds` and were removed.
