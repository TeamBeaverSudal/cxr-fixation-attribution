# Execution provenance

The camera-ready aggregate was produced on 2026-08-18 from implementation
commit `45ada8abc9b7e3ec98b37b4dcb58f0521c67cef8`.

| Analysis | Completed PBS job | Executed source SHA-256 |
|---|---|---|
| Primary five-seed comparison | `117008.ECE-util1` | `5f943df616986d406c95e9f096c7c1d0d2224aeeb2a5c280e769d6de99862e60` |
| Five partitions × five seeds | `117013[].ECE-util1` | `fc1947033e76e97c044a84a945cc1f8c21191ead4b1758e3998a5adb18cfe40b` |
| Matched other-patient record control | `117032.ECE-util1` | `c06f4b536c92d9dbaeca08f14db0e9bf06db5a8cb159c7e81f2dcafd9dc985de` |
| Five chains × five seeds at partial training fractions | `117041[].ECE-util1` | `df6c094bedc617466df27a3db2c3f45c14baffc677df9ecaea5ab8169dca70aa` |

The completed strict aggregate has SHA-256
`e8e54ebc4ad4d83cef88bfd18c487b94267c9b78c85d6982262bf6f48758a71a`.
The submitted camera-ready PDF reviewed against that aggregate has SHA-256
`0b6fe9d8d4eee8dbae6b4e62f4dd28b2be25b9add928eff97abdd98710273728`.

The files under `experiments/` are publication-oriented copies of the executed
analysis logic. Their imports were renamed to the descriptive module names in
this repository, so their byte hashes are not expected to equal the deployment
hashes above. The hashes identify the source that actually ran; the public
copies expose the corresponding computation for audit and rerun.

The earlier submitted analysis mixed five-seed descriptive rows with seed-0
paired inference and seed-0 secondary analyses. It was superseded by the
consistent estimator used here: seed-specific metrics are averaged per instance
before inference, every patient partition repeats all five seeds, and each
training-fraction chain averages all five optimizer seeds.

Raw REFLACX/MIMIC-CXR data, patient-level predictions, model checkpoints, and
qualitative radiographs remain restricted. This repository releases only code,
aggregate results, configuration logic, and nonidentifying provenance.
