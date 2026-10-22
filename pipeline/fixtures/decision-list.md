The importer now reads both archive formats.

Open questions, with my recommendations:

1. The empty archive: report it as a warning, with an exit code of 0. I agree with the plan.
2. The log folder: keep it next to the archive, as the other importer does.
3. The sample files: hand them to the team that owns the fixtures after this lands.
