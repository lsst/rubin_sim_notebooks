# Generating a cloud database for `rubin_sim`

The notebooks should be run in this order:



| notebook | reads | writes |
| -------- | ----- | ------ |
| `scrape_clouds.ipynb` | (Blanco night reports) | clouds_2014-03-30_to_2022-09-04.txt |
| `combine_report_clouds.ipynb` | clouds_1975-01-01_to_2014-03-30.txt, clouds_2014-03-30_to_2022-09-04.txt| clouds_ctio_blanco.txt, clouds_ctio_blanco.h5|
| `build_night_events.ipynb`  | clouds_ctio_blanco.h5 | night_events.h5 |
| `cloud_satellite.ipynb`  | night_events.h5, clouds_ctio_blanco.h5, (GOES-13 data from SDS archive) | satellite_cloudy.txt |
| `fill_missing_clouds.ipynb`  | clouds_ctio_blanco.h5, satellite_cloudy.txt | clouds.h5, clouds.txt |
| `build_cloud_db.ipynb`  | clouds.h5 | clouds.db, clouds.db.gz|
| `examine_clouds.ipynb`  | | |