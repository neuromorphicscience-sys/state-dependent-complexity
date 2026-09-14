"""Validate the published revision's file hashes and quantitative-panel bindings.

Standard-library only. This checks metadata integrity, not raw model fitting,
figure regeneration, independent statistical replication or artwork eligibility.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, name):
    require(isinstance(name, str) and bool(name), 'Empty or non-string path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts
            and '\\' not in name and ':' not in name, 'Unsafe relative path')
    target = (root / name).resolve()
    require(target.is_relative_to(root.resolve()), 'Path escapes revision')
    require(target.is_file(), 'Missing file: ' + name)
    return target


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(root, name):
    return json.loads(safe_file(root, name).read_text(encoding='utf-8'))


def validate_bindings(index, full, crosswalks):
    datasets = index['datasets']
    by_id = {d['id']: d for d in datasets}
    require(len(by_id) == len(datasets) == index['unique_datasets'],
            'Duplicate dataset ID or inconsistent dataset count')
    names = full['crosswalks']
    require(len(names) == len(set(names)) and set(names) == set(crosswalks),
            'Crosswalk inventory mismatch')
    seen = {}
    for name in names:
        walk = crosswalks[name]
        rows = walk.get('records', walk.get('rows'))
        require(isinstance(rows, list) and bool(rows), 'Missing crosswalk rows')
        for row in rows:
            key = (row['figure'], row['panel'])
            require(key not in seen, 'Duplicate figure/panel: ' + repr(key))
            count = row['quantitative_microplots']
            require(type(count) is int and count > 0, 'Invalid microplot count')
            seen[key] = count
            require(bool(row['datasets']), 'Quantitative panel without datasets')
            for data in row['datasets']:
                ident = data['dataset_id']
                require(ident in by_id, 'Unresolved dataset ID: ' + ident)
                require(data['sha256'] == by_id[ident]['sha256'],
                        'Panel/index hash mismatch: ' + ident)
                if 'public_file' in data:
                    require(data['public_file'] == by_id[ident]['file'],
                            'Panel/index path mismatch: ' + ident)
    require(len(seen) == full['quantitative_panels'], 'Panel total mismatch')
    require(sum(seen.values()) == full['quantitative_microplots'],
            'Microplot total mismatch')
    figures = {fig for fig, panel in seen}
    require(figures == set(full['breakdown']) and len(figures) == full['figure_count'],
            'Figure inventory mismatch')
    for fig, expected in full['breakdown'].items():
        values = [n for (f, panel), n in seen.items() if f == fig]
        require(len(values) == expected['panels']
                and sum(values) == expected['microplots'],
                'Figure breakdown mismatch: ' + fig)
    require(index['display_items'] == {'main': 6, 'extended_data': 8, 'supplementary': 6},
            'Unexpected revision display inventory')
    require((len(by_id), len(seen), sum(seen.values()), len(figures)) == (296, 202, 246, 20),
            'Unexpected frozen revision counts')
    return dict(datasets=len(by_id), quantitative_panels=len(seen),
                quantitative_microplots=sum(seen.values()), figures=len(figures))


def verify(root):
    root = Path(root).resolve()
    checksums = read_json(root, 'SHA256SUMS.json')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'SHA256SUMS.json'}
    require(actual == set(checksums), 'Revision inventory/checksum mismatch')
    for name, expected in checksums.items():
        require(digest(safe_file(root, name)) == expected, 'File hash mismatch: ' + name)
    index = read_json(root, 'DATASET_INDEX.json')
    for data in index['datasets']:
        require(digest(safe_file(root, data['file'])) == data['sha256'],
                'Dataset content mismatch: ' + data['id'])
    full = read_json(root, 'panel_maps/full_quantitative_coverage.json')
    walks = {name: read_json(root, 'panel_maps/' + name) for name in full['crosswalks']}
    counts = validate_bindings(index, full, walks)
    return dict(passed=True, **counts, revision_checksums=len(checksums),
                scope='Published metadata and file integrity; no raw fits or statistical reanalysis',
                dataset_index_sha256=digest(root / 'DATASET_INDEX.json'),
                full_coverage_sha256=digest(root / 'panel_maps/full_quantitative_coverage.json'),
                script_sha256=digest(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output filename; previous reports are preserved.')
    report = verify(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
