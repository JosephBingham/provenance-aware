# Input data

## ICIJ Offshore Leaks

Download the full CSV export from the ICIJ Offshore Leaks Database (https://offshoreleaks.icij.org, "Download
the data") and extract it into one folder; set `ICIJ_DIR` to that folder. The database is published by the
International Consortium of Investigative Journalists under the Open Database License (database) and Creative
Commons Attribution-ShareAlike (contents); see the ICIJ site for terms.

The paper uses the export whose marker file is `GENERATED_ON_20260729.txt`. ICIJ regenerates the export over
time; later exports may differ. SHA-256 of the files used:

```
d0ef10a2dacf11b62bb4290b0e2521405f965f36f11534ad50d42bf6e8ecdb2b  nodes-addresses.csv
b1baca3676d045e82e3e7bd701d472210f5cc99a39d6949fa2e203cb2f19cae7  nodes-entities.csv
287a655fca5a607be1929c52f081c848370769d3f7a681900fde7710b9e9fcfe  nodes-intermediaries.csv
c89898b1cf7d4d8b14d9e109a49286d5a53642b1dc21f6a945e7e63a6717f2fa  nodes-officers.csv
32f6532c8bb290233d67423de5d778cb5a55478e721d0a07e9c8a2991a1b82c0  nodes-others.csv
bf0d0b4e36e243b8de2a0aeef160ef5faf3b07a14754d9e7235a758196d85323  relationships.csv
```

Counts in this export: 814,344 entities, 771,315 officers, 3,339,267 relationships (2,820,285 with a source
attribute); 571,870 officers attached to the four leaks used, 14,878 of them name-proxy positives.

## OpenSanctions

`opensanctions/get_opensanctions.py` downloads the `default` collection (no account needed; free for non-commercial
use, commercial use needs a licence from OpenSanctions) and reduces it to a name-free extract. The paper uses version
`20261124065439-mxy` (published 2026-11-24 06:54 UTC):

```
entities.ftm.json  2,592,567,695 bytes  sha1 7d3950786a1109e2470ffc2e7d830ffbd66ff149
statements.csv     9,492,426,845 bytes  sha1 2498536dd249ebe1a77df43541d6d544b9bc7701
```

Both checksums match the published index. Bulk exports older than a few months need an OpenSanctions data delivery
token; the extract itself is not redistributed here.

## Elliptic

The Elliptic Bitcoin dataset (Weber et al., 2019) is available from Kaggle ("elliptic-data-set"); see the
supplement for how blocks of time steps are formed.
