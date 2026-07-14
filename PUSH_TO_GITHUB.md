# Push Instructions

The GitHub CLI was not available in the local environment, so this repository was prepared locally.

To publish:

1. Create a new **public** empty repository on GitHub, for example:
   `cancer-multiomics-alignment-domain-shift-benchmark`

2. From this directory, run:

```bash
git remote add origin https://github.com/YOUR_USER/cancer-multiomics-alignment-domain-shift-benchmark.git
git branch -M main
git push -u origin main
```

3. Replace `TO_BE_ADDED_AFTER_GITHUB_PUBLICATION` in `CITATION.cff` and the manuscript availability statement with the final repository URL and Zenodo DOI if archived.
