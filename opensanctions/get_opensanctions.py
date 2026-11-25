#!/usr/bin/env python3
"""Download OpenSanctions bulk data and reduce it to a compact, name-free graph extract for the E2 experiment.

Run on your own machine (Python 3.8+, standard library only, no account or key needed):

    python3 get_opensanctions.py                      # latest 'default' collection, ~2.5 GB download
    python3 get_opensanctions.py --with-statements    # also per-source topic statements (~9 GB more download)

What it does
  1. Reads the collection's index.json and pins the exact published version (so the extract is citable).
  2. Downloads entities.ftm.json (and, with --with-statements, statements.csv) to <out>/raw/ with automatic
     resume after interruptions, and verifies the published checksum when the index provides one.
  3. Streams the file(s) into a compact extract (no names, no free text):
       nodes.tsv.gz     id, schema, datasets, topics, countries, target, first_seen       (Person, Company, ...)
       edges.tsv.gz     src, dst, schema, datasets                                          (ownership, directorship, ...)
       sanctions.tsv.gz entity id, sanction datasets, authority, program, start date        (Sanction records)
       topics_by_dataset.tsv.gz  entity, canonical id, dataset, topic, first_seen, last_seen (--with-statements)
       summary.json     counts by schema, per-dataset schema and topic counts, datasets per entity
       provenance.json  URLs, version, sizes, checksums, verification, run times, row counts
  4. Packs everything into <out>/opensanctions_e2_<version>.zip. If it is larger than --part-mb MB (default 25),
     it is split into numbered parts so each can be uploaded; upload all parts (they are rejoined byte for byte:
     `cat file.zip.part* > file.zip` on macOS/Linux, `copy /b file.zip.part00+file.zip.part01 file.zip` on Windows).
  5. Deletes the raw downloads unless --keep-raw.

Disk space: about 3 GB free (12 GB with --with-statements) during the run; the final package is ~100-300 MB.
Licence: OpenSanctions bulk data is free for non-commercial use (research included); cite OpenSanctions. Commercial
use needs a licence. The extract is for analysis only; do not republish it.

Options: --dataset NAME (default 'default'), --version latest|YYYYMMDD, --out DIR, --with-statements,
         --keep-raw, --part-mb N, --limit N (first N lines only, for a quick test), --selftest.
"""
import argparse, collections, csv, datetime, gzip, hashlib, io, json, os, platform, shutil, sys, time, urllib.error, \
    urllib.request, zipfile

BASE = os.environ.get("OS_BASE", "https://data.opensanctions.org/datasets")   # override only for testing
UA = {"User-Agent": "research-e2-extract/1.0 (non-commercial academic use)"}
NODE_SCHEMAS = {"Person", "Company", "Organization", "LegalEntity", "Address", "PublicBody"}
LABEL_SCHEMAS = {"Person", "Company", "Organization", "LegalEntity", "PublicBody"}
EDGE_PROPS = {  # FollowTheMoney relationship schema -> (source property, target property)
    "Ownership": ("owner", "asset"), "Directorship": ("director", "organization"),
    "Employment": ("employee", "employer"), "Membership": ("member", "organization"),
    "Representation": ("agent", "client"), "Family": ("person", "relative"),
    "Associate": ("person", "associate"), "UnknownLink": ("subject", "object"),
    "Occupancy": ("holder", "post"), "Succession": ("predecessor", "successor"),
}
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


# ----------------------------------------------------------------------------------------------- index and download
def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_index(dataset, version):
    tried = []
    for url in [f"{BASE}/{version}/{dataset}/index.json", f"{BASE}/{dataset}/{version}/index.json"]:
        try:
            return url, get_json(url)
        except (urllib.error.URLError, ValueError) as e:
            tried.append(f"{url}: {e}")
    sys.exit("Could not read the dataset index:\n  " + "\n  ".join(tried))


def find_resource(index, name, dataset, version):
    for r in index.get("resources") or []:
        if r.get("name") == name or str(r.get("url", "")).endswith("/" + name):
            return r
    return {"name": name, "url": f"{BASE}/{version}/{dataset}/{name}"}   # fall back to the documented URL pattern


def download(url, dest, expected_size=None, tries=50):
    """Resumable download: continues from the bytes already on disk after any interruption."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(dest) and (expected_size is None or os.path.getsize(dest) == expected_size):
        log(f"already downloaded: {dest}")
        return
    for attempt in range(1, tries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        hdr = dict(UA)
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                if have and r.status != 206:      # server ignored the range: start over
                    have = 0; mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = expected_size or (have + int(r.headers.get("Content-Length") or 0)) or None
                t0, got, last = time.time(), 0, 0.0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk); got += len(chunk)
                        if time.time() - last > 15:
                            last = time.time(); done = have + got
                            rate = got / max(1e-9, last - t0)
                            pct = f" ({100 * done / total:.1f}%)" if total else ""
                            log(f"  {os.path.basename(dest)}: {human(done)}{pct} at {human(rate)}/s")
            size = os.path.getsize(part)
            if expected_size and size < expected_size:
                raise IOError(f"incomplete: {size} of {expected_size} bytes")
            os.replace(part, dest); log(f"downloaded {dest} ({human(os.path.getsize(dest))})")
            return
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:      # range beyond end: file is complete
                os.replace(part, dest); return
            wait = min(60, 2 * attempt)
            log(f"  interrupted ({e}); resuming in {wait}s (attempt {attempt}/{tries})")
            time.sleep(wait)
    sys.exit(f"Download failed after {tries} attempts: {url}. Re-run the same command to resume.")


def file_hashes(path):
    h = {"sha1": hashlib.sha1(), "sha256": hashlib.sha256(), "md5": hashlib.md5()}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            for x in h.values():
                x.update(chunk)
    return {k: v.hexdigest() for k, v in h.items()}


def verify(resource, hashes, size):
    """Compare with whatever checksum the index publishes (sha1/sha256/md5, possibly prefixed 'algo:')."""
    out = {"size_bytes": size, **{f"local_{k}": v for k, v in hashes.items()}}
    if resource.get("size") is not None:
        out["size_matches_index"] = int(resource["size"]) == size
    for key in ["checksum", "sha256", "sha1", "md5"]:
        v = resource.get(key)
        if isinstance(v, str) and v:
            algo, _, digest = v.rpartition(":")
            digest = digest.lower()
            algo = algo or key if key != "checksum" else (algo or {40: "sha1", 64: "sha256", 32: "md5"}.get(len(digest)))
            if algo in hashes:
                out["published_checksum"] = v; out["checksum_matches"] = hashes[algo] == digest
                break
    return out


# ----------------------------------------------------------------------------------------------- extraction
def ids(values):
    out = []
    for v in values or []:
        if isinstance(v, dict):
            v = v.get("id")
        if isinstance(v, str) and v:
            out.append(v)
    return out


def clean(s):
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def extract_entities(path, out, limit=None):
    by_schema = collections.Counter(); n_ds = collections.Counter(); bad = 0; rows = collections.Counter()
    ds_schema = collections.defaultdict(collections.Counter); ds_topic = collections.defaultdict(collections.Counter)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fin, \
            gzip.open(f"{out}/nodes.tsv.gz", "wt", encoding="utf-8") as fn, \
            gzip.open(f"{out}/edges.tsv.gz", "wt", encoding="utf-8") as fe, \
            gzip.open(f"{out}/sanctions.tsv.gz", "wt", encoding="utf-8") as fs:
        fn.write("id\tschema\tdatasets\ttopics\tcountries\ttarget\tfirst_seen\n")
        fe.write("src\tdst\tschema\tdatasets\n")
        fs.write("entity\tdatasets\tauthority\tprogram\tstart_date\n")
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                e = json.loads(line)
            except ValueError:
                bad += 1; continue
            sch, eid = e.get("schema"), e.get("id"); props = e.get("properties") or {}
            dss = sorted(set(e.get("datasets") or [])); by_schema[sch] += 1
            if sch in NODE_SCHEMAS:
                topics = sorted(set(props.get("topics") or []))
                countries = sorted(set((props.get("country") or []) + (props.get("jurisdiction") or [])
                                       + (props.get("nationality") or [])))
                fn.write("\t".join([eid, sch, "|".join(dss), "|".join(topics), "|".join(countries),
                                    "1" if e.get("target") else "0", clean(e.get("first_seen") or "")]) + "\n")
                rows["nodes"] += 1
                for a in ids(props.get("addressEntity")):
                    fe.write("\t".join([eid, a, "addressEntity", "|".join(dss)]) + "\n"); rows["edges"] += 1
                if sch in LABEL_SCHEMAS:
                    n_ds[len(dss)] += 1
                    for d in dss:
                        ds_schema[d][sch] += 1
                        for t in topics:
                            ds_topic[d][f"{sch}:{t}"] += 1
            elif sch in EDGE_PROPS:
                a, b = EDGE_PROPS[sch]
                for s in ids(props.get(a)):
                    for t in ids(props.get(b)):
                        fe.write("\t".join([s, t, sch, "|".join(dss)]) + "\n"); rows["edges"] += 1
            elif sch == "Sanction":
                for ent in ids(props.get("entity")):
                    fs.write("\t".join([ent, "|".join(dss), clean("|".join(props.get("authority") or [])),
                                        clean("|".join(props.get("program") or [])),
                                        clean("|".join(props.get("startDate") or []))]) + "\n"); rows["sanctions"] += 1
            if i and i % 1_000_000 == 0:
                log(f"  entities: {i:,} lines")
    summary = {"lines_unparsed": bad, "entities_by_schema": by_schema.most_common(),
               "label_entities_by_number_of_datasets": sorted(n_ds.items()),
               "datasets": {d: {"schemas": dict(ds_schema[d]), "topics": dict(ds_topic[d].most_common(40))}
                            for d in sorted(ds_schema, key=lambda d: -sum(ds_schema[d].values()))}}
    return summary, dict(rows)


def extract_statements(path, out, limit=None):
    """Keep only topic statements: which source dataset asserted which topic for which entity."""
    n = kept = 0; per = collections.Counter()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fin, \
            gzip.open(f"{out}/topics_by_dataset.tsv.gz", "wt", encoding="utf-8") as fo:
        rd = csv.DictReader(fin); header = rd.fieldnames or []
        need = {"prop", "value", "dataset"}
        if not need <= set(header):
            log(f"statements.csv has unexpected columns {header}; skipped"); return {"header": header, "skipped": True}
        ent = "entity_id" if "entity_id" in header else header[0]
        can = "canonical_id" if "canonical_id" in header else ent
        fo.write("entity\tcanonical\tdataset\ttopic\tfirst_seen\tlast_seen\n")
        for row in rd:
            n += 1
            if limit and n > limit:
                break
            if row.get("prop") == "topics":
                fo.write("\t".join(clean(row.get(k) or "") for k in [ent, can, "dataset", "value", "first_seen", "last_seen"]) + "\n")
                kept += 1; per[(row.get("dataset"), row.get("value"))] += 1
            if n % 5_000_000 == 0:
                log(f"  statements: {n:,} rows, {kept:,} topic statements")
    return {"header": header, "rows_read": n, "topic_statements": kept,
            "topic_statements_by_dataset": [[d, t, c] for (d, t), c in per.most_common(400)]}


# ----------------------------------------------------------------------------------------------- packaging
def package(out, version, files, part_mb):
    zpath = os.path.join(out, f"opensanctions_e2_{version}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:   # members are already gzipped
        for f in files:
            if os.path.exists(os.path.join(out, f)):
                z.write(os.path.join(out, f), arcname=f)
    size = os.path.getsize(zpath); parts = [zpath]
    lim = max(1 << 20, int(part_mb * 1024 * 1024)) if part_mb >= 1 else max(1 << 16, int(part_mb * 1024 * 1024))
    if size > lim:
        parts = []
        with open(zpath, "rb") as f:
            k = 0
            while True:
                chunk = f.read(lim)
                if not chunk:
                    break
                p = f"{zpath}.part{k:02d}"; open(p, "wb").write(chunk); parts.append(p); k += 1
        os.remove(zpath)
    return zpath, size, parts


def selftest(tmp):
    """Run the extraction on a tiny synthetic file (no network)."""
    os.makedirs(tmp, exist_ok=True)
    ents = [
        {"id": "p1", "schema": "Person", "datasets": ["us_ofac_sdn", "eu_fsf"], "target": True, "first_seen": "2020-01-01",
         "properties": {"topics": ["sanction"], "nationality": ["ru"], "addressEntity": ["a1"]}},
        {"id": "c1", "schema": "Company", "datasets": ["gb_coh"], "properties": {"country": ["gb"]}},
        {"id": "a1", "schema": "Address", "datasets": ["us_ofac_sdn"], "properties": {}},
        {"id": "o1", "schema": "Ownership", "datasets": ["gb_coh"], "properties": {"owner": ["p1"], "asset": ["c1"]}},
        {"id": "s1", "schema": "Sanction", "datasets": ["us_ofac_sdn"],
         "properties": {"entity": ["p1"], "authority": ["OFAC"], "program": ["RUSSIA-EO14024"], "startDate": ["2022-03-01"]}},
    ]
    with open(f"{tmp}/entities.ftm.json", "w") as f:
        f.write("\n".join(json.dumps(e) for e in ents) + "\nnot json\n")
    with open(f"{tmp}/statements.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "entity_id", "canonical_id", "prop", "prop_type", "schema", "value", "dataset",
                                       "lang", "original_value", "first_seen", "last_seen"])
        w.writerow(["x1", "ofac-1", "p1", "topics", "topic", "Person", "sanction", "us_ofac_sdn", "", "", "2020-01-01", "2026-11-01"])
        w.writerow(["x2", "eu-9", "p1", "name", "name", "Person", 'Doe, "J"\nline2', "eu_fsf", "", "", "2021-01-01", "2026-11-01"])
        w.writerow(["x3", "eu-9", "p1", "topics", "topic", "Person", "sanction", "eu_fsf", "", "", "2021-01-01", "2026-11-01"])
    s, rows = extract_entities(f"{tmp}/entities.ftm.json", tmp)
    st = extract_statements(f"{tmp}/statements.csv", tmp)
    assert rows == {"nodes": 3, "edges": 2, "sanctions": 1}, rows
    assert s["lines_unparsed"] == 1 and st["topic_statements"] == 2, (s, st)
    nodes = gzip.open(f"{tmp}/nodes.tsv.gz", "rt").read().splitlines()
    assert nodes[1].split("\t")[:6] == ["p1", "Person", "eu_fsf|us_ofac_sdn", "sanction", "ru", "1"], nodes[1]
    _, size, parts = package(tmp, "selftest", ["nodes.tsv.gz", "edges.tsv.gz", "sanctions.tsv.gz", "topics_by_dataset.tsv.gz"], 1)
    print("selftest passed:", rows, f"package {size} bytes in {len(parts)} part(s)")


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="default"); ap.add_argument("--version", default="latest")
    ap.add_argument("--out", default="opensanctions_e2"); ap.add_argument("--with-statements", action="store_true")
    ap.add_argument("--keep-raw", action="store_true"); ap.add_argument("--part-mb", type=float, default=25)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(os.path.join(a.out, "_selftest")); return
    os.makedirs(a.out, exist_ok=True); raw = os.path.join(a.out, "raw")
    free = shutil.disk_usage(os.path.abspath(a.out)).free
    need = (12 if a.with_statements else 3) * 1024 ** 3
    if free < need and not a.limit:
        log(f"warning: {human(free)} free in {a.out}; about {human(need)} is needed")
    t_start = datetime.datetime.now(datetime.timezone.utc)
    idx_url, index = fetch_index(a.dataset, a.version)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    version = str(index.get("version") or index.get("last_export") or a.version)
    log(f"{a.dataset}: version {version}, updated {index.get('updated_at') or index.get('last_export')}")
    prov = {"dataset": a.dataset, "requested_version": a.version, "published_version": version, "index_url": idx_url,
            "index_updated_at": index.get("updated_at"), "index_last_change": index.get("last_change"),
            "run_started_utc": t_start.isoformat(), "python": platform.python_version(), "platform": platform.platform(),
            "limit": a.limit, "files": {}}
    names = ["entities.ftm.json"] + (["statements.csv"] if a.with_statements else [])
    for name in names:
        res = find_resource(index, name, a.dataset, a.version)
        url = res.get("url"); dest = os.path.join(raw, name)
        log(f"downloading {url} ({human(int(res['size'])) if res.get('size') else 'size unknown'})")
        download(url, dest, int(res["size"]) if res.get("size") else None)
        log(f"hashing {name}")
        ver = verify(res, file_hashes(dest), os.path.getsize(dest))
        prov["files"][name] = {"url": url, "index_entry": {k: v for k, v in res.items() if k != "url"}, **ver}
        if ver.get("checksum_matches") is False or ver.get("size_matches_index") is False:
            sys.exit(f"{name}: checksum or size does not match the index; delete {dest} and re-run")
    log("extracting entities"); summary, rows = extract_entities(os.path.join(raw, "entities.ftm.json"), a.out, a.limit)
    prov["rows"] = rows
    if a.with_statements:
        log("extracting topic statements")
        summary["statements"] = extract_statements(os.path.join(raw, "statements.csv"), a.out, a.limit)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    prov["run_finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    files = ["nodes.tsv.gz", "edges.tsv.gz", "sanctions.tsv.gz", "topics_by_dataset.tsv.gz", "summary.json",
             "provenance.json", "index.json"]
    zpath, size, parts = package(a.out, version.replace(":", "").replace("/", "_"), files, a.part_mb)
    if not a.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)
    print("\nDone. Upload these file(s):")
    for p in parts:
        print(f"  {p}  ({human(os.path.getsize(p))})")
    if len(parts) > 1:
        print(f"(The package, {human(size)}, was split into {len(parts)} parts of at most {a.part_mb} MB.)")
    print(f"Rows: {rows}. Provenance and checksums are in {os.path.join(a.out, 'provenance.json')}.")


if __name__ == "__main__":
    main()
